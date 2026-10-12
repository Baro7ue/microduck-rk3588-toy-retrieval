# Moved 2026-10-11 from work/ball_follow_rpc.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Bounded board-local robotd transport; this module never enables or starts motion."""
from __future__ import annotations

import copy
from collections import deque
import json
import math
from pathlib import Path
import socket
import threading
import time
import traceback


MAX_BUFFER = 1024 * 1024


def _packet(message: dict) -> bytes:
    return (json.dumps(message, allow_nan=False) + '\n').encode()


def _lines(buffer: bytearray):
    while b'\n' in buffer:
        end = buffer.index(b'\n') + 1
        line = bytes(buffer[:end])
        del buffer[:end]
        yield line


def _check_stand(state: dict) -> None:
    gravity = state['safety']['gravity']
    if (state['policy'] != 'stand' or state['safety']['fallen'] is not False
            or len(gravity) != 3 or any(not math.isfinite(v) for v in gravity)
            or gravity[2] > -0.9
            or any(len(state['move'][field]) != 3 for field in ('requested', 'applied'))
            or any(not math.isfinite(v) or abs(v) > 1e-8
                   for field in ('requested', 'applied') for v in state['move'][field])):
        raise RuntimeError('initial state is not unfallen, upright, zero-velocity stand')


class RobotLink:
    def __init__(self, path, directory, io_timeout=0.15):
        self.path, self.directory = str(path), Path(directory)
        if not math.isfinite(io_timeout) or io_timeout <= 0:
            raise ValueError('io_timeout must be positive and finite')
        self.io_timeout = io_timeout
        self._condition = threading.Condition()
        self._move_lock = threading.Lock()
        self._stopped = threading.Event()
        self._state_socket = self._move_socket = self._log = self._thread = None
        self._state = self._first_state = self._received_at = self._last_t_ns = None
        self._pose_history = deque(maxlen=64)
        self._models = None
        self._error = None
        self._next_id = 2

    @property
    def models(self):
        with self._condition:
            return copy.deepcopy(self._models)

    @property
    def error(self):
        with self._condition:
            return self._error

    def _record_error(self):
        detail = traceback.format_exc()
        with self._condition:
            self._error = (self._error or '') + detail
            self._condition.notify_all()

    def _connect(self, timeout=None):
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(self.io_timeout if timeout is None else timeout)
        try:
            client.connect(self.path)
        except Exception:
            client.close()
            raise
        return client

    def start(self):
        try:
            if self._state_socket is not None or self._stopped.is_set():
                raise RuntimeError('RobotLink cannot be started twice')
            self.directory.mkdir(parents=True, exist_ok=True)
            self._log = (self.directory / 'states.jsonl').open('xb')
            self._state_socket = self._connect()
            self._state_socket.sendall(_packet({'jsonrpc': '2.0', 'id': 1,
                'method': 'robot.subscribe', 'params': {'hz': 20}}))
            self._move_socket = self._connect()
            self._thread = threading.Thread(target=self._receive_states, daemon=True)
            self._thread.start()
        except Exception:
            self._record_error()
            self.close()
            raise

    def _receive_states(self):
        buffer = bytearray()
        try:
            while not self._stopped.is_set():
                try:
                    data = self._state_socket.recv(min(65536, MAX_BUFFER - len(buffer)))
                except socket.timeout:
                    continue  # Polling does not refresh a state deadline.
                except OSError:
                    if self._stopped.is_set():
                        break
                    raise
                if not data:
                    if self._stopped.is_set():
                        break
                    if buffer:
                        self._log.write(buffer)
                        self._log.flush()
                    raise EOFError('robotd state connection EOF')
                received = time.monotonic()
                buffer.extend(data)
                for line in _lines(buffer):
                    try:
                        message = json.loads(line)
                    except Exception:
                        self._log.write(line)
                        self._log.flush()
                        raise
                    if message.get('method') == 'robot.state':
                        logged = {**message, 'received_monotonic_s': received}
                        self._log.write((json.dumps(logged) + '\n').encode())
                    else:
                        self._log.write(line)
                    self._log.flush()
                    with self._condition:
                        if message.get('id') == 1:
                            if 'error' in message or not message.get('result', {}).get('accepted'):
                                raise RuntimeError(f'robot.subscribe refused: {message}')
                            self._models = message['result']
                        elif message.get('method') == 'robot.state':
                            state = message['params']
                            tick = state['t_ns']
                            if isinstance(tick, bool) or not isinstance(tick, int):
                                raise ValueError('robot.state t_ns must be an integer')
                            if self._last_t_ns is not None and tick < self._last_t_ns:
                                raise RuntimeError('robot.state sensor t_ns moved backward')
                            self._state = state
                            if self._first_state is None:
                                self._first_state = state
                            if self._last_t_ns is None or tick > self._last_t_ns:
                                self._last_t_ns, self._received_at = tick, received
                                self._pose_history.append((tick, received, copy.deepcopy(state)))
                        self._condition.notify_all()
                if len(buffer) >= MAX_BUFFER:
                    raise ValueError('robotd state message exceeds 1 MiB buffer limit')
        except Exception:
            self._record_error()
        finally:
            try:
                self._log.close()
            except Exception:
                self._record_error()

    def snapshot(self):
        with self._condition:
            return copy.deepcopy(self._state), self._received_at

    def camera_pose_at(self, frame_at, max_skew=.1):
        """Latest causal state near frame receipt; this does not measure exposure time."""
        try:
            valid = (not isinstance(frame_at, bool) and math.isfinite(frame_at)
                     and not isinstance(max_skew, bool) and math.isfinite(max_skew)
                     and max_skew > 0)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            return None
        with self._condition:
            selected = next(((tick, received, state)
                             for tick, received, state in reversed(self._pose_history)
                             if 0 <= frame_at-tick/1e9 <= max_skew
                             and 0 <= frame_at-received <= max_skew), None)
        if selected is None:
            return None
        tick, received, state = selected
        from ball_follow_camera import normalize_quat, world_camera_quat
        quat = world_camera_quat(state)
        if quat is None:
            return None
        trunk_quat = normalize_quat(state['imu']['quat'])
        return {'quat': list(quat), 'trunk_quat': list(trunk_quat),
                'sensor_t_ns': tick,
                'received_monotonic_s': received,
                'sensor_to_frame_s': frame_at-tick/1e9,
                'receipt_to_frame_s': frame_at-received}

    def wait_ready(self, timeout=2.0):
        deadline = time.monotonic() + timeout
        try:
            with self._condition:
                while True:
                    if self._error:
                        raise RuntimeError(self._error)
                    if self._stopped.is_set():
                        raise RuntimeError('RobotLink is closed')
                    if self._models is not None and (
                            self._models.get('walk') != 'walk_17500.onnx'
                            or self._models.get('stand') != 'velstand.onnx'):
                        raise RuntimeError(f'unexpected robotd models: {self._models}')
                    if self._first_state is not None:
                        _check_stand(self._first_state)
                    now = time.monotonic()
                    if self._models is not None and self._received_at is not None \
                            and now - self._received_at <= 0.3:
                        _check_stand(self._state)
                        return copy.deepcopy(self._state)
                    remaining = deadline - now
                    if remaining <= 0:
                        raise TimeoutError('no fresh validated stand state before readiness deadline')
                    self._condition.wait(min(remaining, self.io_timeout))
        except Exception:
            self._record_error()
            raise

    def move(self, vx, vy, vyaw):
        try:
            with self._move_lock:
                if self._move_socket is None:
                    raise OSError('robot.move connection is closed')
                self._move_socket.sendall(_packet({'jsonrpc': '2.0', 'method': 'robot.move',
                    'params': {'vx': vx, 'vy': vy, 'vyaw': vyaw}}))
        except Exception:
            self._record_error()
            self.close_moves()
            raise

    def _send_request(self, client, method, params, deadline):
        with self._condition:
            request_id = self._next_id
            self._next_id += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'{method}: reply deadline exceeded')
        client.settimeout(min(self.io_timeout, remaining))
        client.sendall(_packet({'jsonrpc': '2.0', 'id': request_id,
                               'method': method, 'params': params}))
        buffer = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f'{method}: reply deadline exceeded')
            client.settimeout(min(self.io_timeout, remaining))
            try:
                data = client.recv(min(65536, MAX_BUFFER - len(buffer)))
            except socket.timeout:
                continue
            if not data:
                raise EOFError(f'{method}: EOF before matching reply')
            buffer.extend(data)
            for line in _lines(buffer):
                if time.monotonic() > deadline:
                    raise TimeoutError(f'{method}: reply deadline exceeded')
                reply = json.loads(line)
                if reply.get('id') != request_id:
                    continue
                if 'error' in reply:
                    raise RuntimeError(f'{method}: {reply["error"]}')
                if 'result' not in reply:
                    raise ValueError(f'{method}: matching reply has no result')
                if time.monotonic() > deadline:
                    raise TimeoutError(f'{method}: reply deadline exceeded')
                return reply['result']
            if len(buffer) >= MAX_BUFFER:
                raise ValueError('robotd reply exceeds 1 MiB buffer limit')

    def request(self, method, params):
        deadline = time.monotonic() + 0.3
        try:
            with self._connect(min(self.io_timeout, 0.3)) as client:
                return self._send_request(client, method, params, deadline)
        except Exception:
            self._record_error()
            raise

    def stop_moves(self):
        """After the sender has joined, stop in-order on its own stream, then close it."""
        deadline = time.monotonic() + 0.3
        try:
            with self._move_lock:
                if self._move_socket is None:
                    raise OSError('robot.move connection is closed')
                self._move_socket.settimeout(min(self.io_timeout, 0.3))
                self._move_socket.sendall(_packet({'jsonrpc': '2.0', 'method': 'robot.move',
                    'params': {'vx': 0.0, 'vy': 0.0, 'vyaw': 0.0}}))
                return self._send_request(self._move_socket, 'robot.stop', {}, deadline)
        except Exception:
            self._record_error()
            raise
        finally:
            self.close_moves()

    def close_moves(self):
        with self._move_lock:
            client, self._move_socket = self._move_socket, None
            if client is not None:
                try:
                    client.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                client.close()

    def close(self):
        self._stopped.set()
        with self._condition:
            self._condition.notify_all()
        self.close_moves()
        if self._state_socket is not None:
            try:
                self._state_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=max(0.5, self.io_timeout * 2))
            if self._thread.is_alive():
                try:
                    raise RuntimeError('robotd state thread did not stop within join bound')
                except RuntimeError:
                    self._record_error()
                    raise
        elif self._log is not None:
            try:
                self._log.close()
            except Exception:
                self._record_error()
                raise
        if self._state_socket is not None:
            self._state_socket.close()
