"""robotd client for Stage 1 and later: the state stream, the intents, and the fall guard.

    robot = Robot(socket_path, run_dir, events)
    robot.move(0.25, 0, 0); robot.head('H2'); robot.mouth(1.0)
    robot.request('robot.do', {'skill': 'ground_pick'})
    state = robot.wait(lambda s: s['policy'] == 'stand', 2.0, 'stand')

Same wire as Stage 0's link.Robot: one connection subscribed to every robot.state tick (50 Hz,
`params: {}`), one connection re-sending the current intents at 20 Hz so robotd's 500 ms deadman
never fires mid-test. Added for Stage 1:

* robot.mouth (0 closed .. 1 open) is an intent like move and head. robotd drives the mouth only
  from it; no policy moves the mouth (robotd/src/main.rs:3073-3079).
* The fall guard. When robot.state reports safety.fallen, the move intent is zeroed at once and
  the zero is sent on the same tick, not at the next 20 Hz resend. robotd still smooths the command
  (ema += 0.2 * (target - ema) per tick, robotd/src/main.rs:2525-2528), so the walk slot keeps
  driving for a few ticks; zeroing early is all a client can do. Only safety.fallen is used: a real
  robot has no truth. Each fall is kept in `falls` with the time it was seen.

Nothing here reads the simulator; the same code runs against robotd on the board.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import socket
import threading
import time

SEND_HZ = 20.0
UNLOGGED = ('skeleton',)            # about 9 KB a tick; frames (camera pose) is kept
HEADS = {
    'H0': (0.0, 0.0),
    'H1': (0.0, 0.6),
    'H2': (0.0, 1.0),
    'H3': (-0.4, 0.8),
}
UPRIGHT = ('stand', 'walk')


class Events:
    """events.jsonl: every command and op, on the time.monotonic() clock."""

    def __init__(self, path: Path):
        self.file = Path(path).open('a')
        self.lock = threading.Lock()

    def log(self, event: str, **fields) -> None:
        with self.lock:
            self.file.write(json.dumps({'mono': time.monotonic(), 'event': event, **fields}) + '\n')
            self.file.flush()

    def close(self) -> None:
        self.file.close()


def _line(message: dict) -> bytes:
    return (json.dumps(message, allow_nan=False) + '\n').encode()


def _notification(method: str, params: dict) -> bytes:
    return _line({'jsonrpc': '2.0', 'method': method, 'params': params})


class Robot:
    def __init__(self, path: str, directory: Path | None, events: Events | None,
                 fall_guard: bool = True, on_state=None):
        self.path, self.events = path, events
        self.fall_guard = fall_guard
        self.on_state = on_state          # called as on_state(state, received) for every tick
        self.cond = threading.Condition()
        self.send_lock = threading.Lock()
        self.state = None
        self.received = None
        self.models = None
        self.error = None
        self.labels: list[tuple[float, str]] = []
        self.falls: list[dict] = []
        self.was_fallen = False
        self.intent = [0.0, 0.0, 0.0]
        self.head_intent = HEADS['H0']
        self.mouth_intent = 0.0
        self.stopped = threading.Event()
        self.log = (Path(directory) / 'state.jsonl').open('a') if directory else None
        self.sub = self._connect()
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

    def _event(self, event: str, **fields) -> None:
        if self.events is not None:
            self.events.log(event, **fields)

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
                fallen = bool(state.get('safety', {}).get('fallen'))
                if fallen and not self.was_fallen:
                    self._on_fall(state, received)
                self.was_fallen = fallen
                if self.log is not None:
                    trimmed = {k: v for k, v in state.items() if k not in UNLOGGED}
                    self.log.write(json.dumps({'mono': received, **trimmed}) + '\n')
                with self.cond:
                    if not self.labels or self.labels[-1][1] != state['policy']:
                        self.labels.append((received, state['policy']))
                    self.state, self.received = state, received
                    self.cond.notify_all()
                if self.on_state is not None:
                    self.on_state(state, received)
        except Exception as error:
            if not self.stopped.is_set():
                with self.cond:
                    self.error = f'{type(error).__name__}: {error}'
                    self.cond.notify_all()
        finally:
            if self.log is not None:
                self.log.flush()

    def _on_fall(self, state: dict, seen: float) -> None:
        record = {'mono': seen, 't': state.get('t'), 'policy': state.get('policy'),
                  'zeroed': False}
        if self.fall_guard:
            with self.cond:
                self.intent = [0.0, 0.0, 0.0]
            try:
                with self.send_lock:
                    self.out.sendall(_notification('robot.move',
                                                   {'vx': 0.0, 'vy': 0.0, 'vyaw': 0.0}))
                record['zeroed'] = True
            except OSError as error:
                record['error'] = str(error)
        self.falls.append(record)
        self._event('fall_guard', **record)

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

    def next_states(self, count: int, timeout: float = 2.0) -> list[dict]:
        """The next `count` ticks after this call (a tick missed between wake-ups is skipped)."""
        out = []
        deadline = time.monotonic() + timeout
        with self.cond:
            last_t = self.state['t'] if self.state is not None else None
            while len(out) < count:
                if self.error:
                    raise RuntimeError(self.error)
                if self.state is not None and self.state['t'] != last_t:
                    out.append(self.state)
                    last_t = self.state['t']
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'{count} robot.state ticks not received in {timeout} s')
                self.cond.wait(min(remaining, 0.1))
        return out

    # ── intents ─────────────────────────────────────────────────────────

    def _send(self) -> None:
        period = 1.0 / SEND_HZ
        while not self.stopped.is_set():
            with self.cond:
                vx, vy, vyaw = self.intent
                neck, head = self.head_intent
                mouth = self.mouth_intent
            try:
                with self.send_lock:
                    self.out.sendall(_notification('robot.move', {'vx': vx, 'vy': vy, 'vyaw': vyaw}))
                    self.out.sendall(_notification('robot.head', {
                        'neck_pitch': neck, 'head_pitch': head, 'head_yaw': 0.0,
                        'head_roll': 0.0}))
                    self.out.sendall(_notification('robot.mouth', {'open': mouth}))
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
            # Send now as well: a pulse's start and end must not wait for the 20 Hz resend.
            with self.send_lock:
                self.out.sendall(_notification('robot.move', {'vx': values[0], 'vy': values[1],
                                                              'vyaw': values[2]}))
            self._event('move', vx=values[0], vy=values[1], vyaw=values[2])

    def head(self, pose: str | tuple[float, float]) -> None:
        neck, head = HEADS[pose] if isinstance(pose, str) else pose
        with self.cond:
            self.head_intent = (float(neck), float(head))
        self._event('head', pose=pose if isinstance(pose, str) else None,
                    neck_pitch=neck, head_pitch=head)

    def mouth(self, fraction: float) -> None:
        value = min(1.0, max(0.0, float(fraction)))
        with self.cond:
            changed = value != self.mouth_intent
            self.mouth_intent = value
        if changed:
            with self.send_lock:
                self.out.sendall(_notification('robot.mouth', {'open': value}))
            self._event('mouth', open=value)

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
        self._event('rpc', method=method, params=params, reply=reply)
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
            if self.log is not None:
                self.log.close()
