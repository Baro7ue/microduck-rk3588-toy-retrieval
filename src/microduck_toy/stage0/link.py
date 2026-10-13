"""Stage 0 connections: robotd's control socket and the probe body's test port.

`Robot` keeps an every-tick (50 Hz) state subscription (trimmed copy in state.jsonl) and re-sends the
current move and head intent at 20 Hz, so robotd's 500 ms deadman never fires mid-test.
`Probe` is the sim-only truth/test port of probe_body.py. Both log what they send to
events.jsonl, with the same time.monotonic() clock the probe writes into truth.csv.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import socket
import threading
import time

SEND_HZ = 20.0
# Dropped from the logged copy of robot.state: about 9 KB per message at 50 Hz.
UNLOGGED = ('skeleton', 'frames')
HEADS = {
    'H0': (0.0, 0.0),
    'H1': (0.0, 0.6),
    'H2': (0.0, 1.0),
    'H3': (-0.4, 0.8),
}
UPRIGHT = ('stand', 'walk')


class Events:
    def __init__(self, path: Path):
        self.file = path.open('a')
        self.lock = threading.Lock()

    def log(self, event: str, **fields) -> None:
        with self.lock:
            self.file.write(json.dumps({'mono': time.monotonic(), 'event': event, **fields}) + '\n')
            self.file.flush()

    def close(self) -> None:
        self.file.close()


def _line(message: dict) -> bytes:
    return (json.dumps(message, allow_nan=False) + '\n').encode()


class Probe:
    def __init__(self, host: str = '127.0.0.1', port: int = 7811, events: Events | None = None):
        self.sock = socket.create_connection((host, port), timeout=5.0)
        self.stream = self.sock.makefile('rwb')
        self.events = events
        self.lock = threading.Lock()

    def call(self, op: str, **params) -> dict:
        with self.lock:
            self.stream.write(_line({'op': op, **params}))
            self.stream.flush()
            answer = json.loads(self.stream.readline())
        if 'error' in answer:
            raise RuntimeError(f'probe {op}: {answer["error"]}')
        if self.events and op not in ('truth', 'info'):
            self.events.log(f'probe.{op}', **params, reply=answer)
        return answer

    def truth(self) -> dict:
        row = self.call('truth')['row']
        if row is None:
            raise RuntimeError('probe has no truth yet')
        return row

    def mark(self, label: str) -> None:
        self.call('mark', label=label)

    def close(self) -> None:
        try:
            self.call('record', path=None)
        finally:
            self.sock.close()


class Robot:
    def __init__(self, path: str, directory: Path, events: Events):
        self.path, self.events = path, events
        self.cond = threading.Condition()
        self.state = None
        self.received = None
        self.models = None
        self.error = None
        self.labels: list[tuple[float, str]] = []
        self.intent = [0.0, 0.0, 0.0]
        self.head_intent = HEADS['H0']
        self.stopped = threading.Event()
        self.log = (directory / 'state.jsonl').open('a')
        self.sub = self._connect()
        # No `hz`: every tick. hz=50 decimates against tick jitter and delivers about 25 Hz.
        self.sub.sendall(_line({'jsonrpc': '2.0', 'id': 1, 'method': 'robot.subscribe',
                                'params': {}}))
        self.out = self._connect()
        self._next_id = 100
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.sender = threading.Thread(target=self._send, daemon=True)
        self.sender.start()

    def _connect(self) -> socket.socket:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(3.0)
        client.connect(self.path)
        return client

    # ── state stream ─────────────────────────────────────────────────────

    def _read(self) -> None:
        stream = self.sub.makefile('rb')
        try:
            while not self.stopped.is_set():
                try:
                    line = stream.readline()
                except socket.timeout:
                    continue
                if not line:
                    raise EOFError('robotd closed the state stream')
                received = time.monotonic()
                message = json.loads(line)
                if message.get('id') == 1:
                    if 'error' in message or not message.get('result', {}).get('accepted'):
                        raise RuntimeError(f'robot.subscribe refused: {message}')
                    with self.cond:
                        self.models = message['result']
                    continue
                if message.get('method') != 'robot.state':
                    continue
                state = message['params']
                trimmed = {k: v for k, v in state.items() if k not in UNLOGGED}
                self.log.write(json.dumps({'mono': received, **trimmed}) + '\n')
                with self.cond:
                    if not self.labels or self.labels[-1][1] != state['policy']:
                        self.labels.append((received, state['policy']))
                    self.state, self.received = state, received
                    self.cond.notify_all()
        except Exception as error:
            if not self.stopped.is_set():
                with self.cond:
                    self.error = f'{type(error).__name__}: {error}'
                    self.cond.notify_all()
        finally:
            self.log.flush()

    def snapshot(self) -> dict:
        with self.cond:
            if self.error:
                raise RuntimeError(self.error)
            if self.state is None or time.monotonic() - self.received > 0.5:
                raise TimeoutError('no fresh robot.state in the last 0.5 s')
            return self.state

    def wait(self, predicate, timeout: float, what: str):
        """Wait for a state the predicate accepts; returns it, or raises TimeoutError."""
        deadline = time.monotonic() + timeout
        with self.cond:
            while True:
                if self.error:
                    raise RuntimeError(self.error)
                if self.state is not None and predicate(self.state):
                    return self.state
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'{what}: not reached within {timeout} s')
                self.cond.wait(min(remaining, 0.1))

    # ── intents ─────────────────────────────────────────────────────────

    def _send(self) -> None:
        period = 1.0 / SEND_HZ
        while not self.stopped.is_set():
            with self.cond:
                vx, vy, vyaw = self.intent
                neck, head = self.head_intent
            try:
                self.out.sendall(_line({'jsonrpc': '2.0', 'method': 'robot.move',
                                        'params': {'vx': vx, 'vy': vy, 'vyaw': vyaw}}))
                self.out.sendall(_line({'jsonrpc': '2.0', 'method': 'robot.head',
                                        'params': {'neck_pitch': neck, 'head_pitch': head,
                                                   'head_yaw': 0.0, 'head_roll': 0.0}}))
            except OSError as error:
                if not self.stopped.is_set():
                    with self.cond:
                        self.error = f'move connection: {error}'
                return
            self.stopped.wait(period)

    def move(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> None:
        values = [float(vx), float(vy), float(vyaw)]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('non-finite move')
        with self.cond:
            changed = values != self.intent
            self.intent = values
        if changed:
            self.events.log('move', vx=values[0], vy=values[1], vyaw=values[2])

    def head(self, pose: str | tuple[float, float]) -> None:
        neck, head = HEADS[pose] if isinstance(pose, str) else pose
        with self.cond:
            self.head_intent = (float(neck), float(head))
        self.events.log('head', pose=pose if isinstance(pose, str) else None,
                        neck_pitch=neck, head_pitch=head)

    def request(self, method: str, params: dict, timeout: float = 3.0) -> dict:
        with self.cond:
            request_id = self._next_id
            self._next_id += 1
        with self._connect() as client:
            client.settimeout(timeout)
            client.sendall(_line({'jsonrpc': '2.0', 'id': request_id, 'method': method,
                                  'params': params}))
            stream = client.makefile('rb')
            while True:
                line = stream.readline()
                if not line:
                    raise EOFError(f'{method}: connection closed before its reply')
                reply = json.loads(line)
                if reply.get('id') == request_id:
                    break
        self.events.log('rpc', method=method, params=params, reply=reply)
        if 'error' in reply:
            raise RuntimeError(f'{method}: {reply["error"]}')
        return reply['result']

    def close(self) -> None:
        try:
            self.move(0.0, 0.0, 0.0)
            time.sleep(2.0 / SEND_HZ)
        finally:
            self.stopped.set()
            for client in (self.out, self.sub):
                try:
                    client.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                client.close()
            self.reader.join(timeout=1.0)
            self.sender.join(timeout=1.0)
            self.log.close()
