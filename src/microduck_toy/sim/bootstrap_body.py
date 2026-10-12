# Moved 2026-10-11 from work/runtime_bootstrap_body.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Explicit simulation startup support; the pinned body and robotd sources are unchanged.

Support ends once, after robotd reports its first stand/walk inference tick.
Never use the supported interval as evidence of active balancing.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time

from mjlab_microduck.sim import body_server as sim

DEFAULT_SOCKET = os.environ.get('MD_STARTUP_SOCKET', '/private/tmp/md-runtime-ethan/robotd.sock')


class StartupBody(sim.Body):
    def __init__(self, *args, startup_socket: str = DEFAULT_SOCKET, **kwargs):
        super().__init__(*args, **kwargs)
        self.startup_socket = startup_socket
        self.startup_started = False
        self.support_ended = False
        self.bootstrap_error = None

    def set_torque(self, on: bool) -> None:
        start_watch = False
        # Preserve the original torque/gain semantics, with startup holding changed atomically.
        with self.world.lock:
            self.torque_on = bool(on)
            if on:
                self.released = self.support_ended
                self.world.data.ctrl[self.actuator_slice] = self.world.data.qpos[self.qpos_adr]
                if not self.startup_started:
                    self.startup_started = True
                    start_watch = True
            self._apply_torque()
        if start_watch:
            print('== startup assistance: hold HOME during homing; wait for a real policy tick',
                  flush=True)
            threading.Thread(target=self._watch_policy, daemon=True).start()

    def _watch_policy(self) -> None:
        started = time.monotonic()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(2.0)
                client.connect(self.startup_socket)
                client.sendall((json.dumps({'jsonrpc': '2.0', 'id': 1,
                    'method': 'robot.subscribe', 'params': {'hz': 50}}) + '\n').encode())
                with client.makefile('rb') as stream:
                    while time.monotonic() - started < 5.0:
                        line = stream.readline()
                        if not line:
                            raise RuntimeError('robotd startup state stream closed')
                        packet = json.loads(line)
                        if packet.get('id') == 1:
                            if 'error' in packet or not packet.get('result', {}).get('accepted'):
                                raise RuntimeError(f'robotd refused startup subscription: {packet}')
                            continue
                        if packet.get('method') != 'robot.state':
                            continue
                        state = packet['params']
                        if state['policy'] in {'stand', 'walk'}:
                            # The daemon reports the active inference phase after attempting its
                            # target write. Bus-write failures remain visible in robotd's log.
                            with self.world.lock:
                                self.support_ended = True
                                self.released = True
                            print('== startup assistance ended: '
                                  + json.dumps({'policy': state['policy'],
                                                'support_s': time.monotonic() - started,
                                                'daemon_t': state['t'],
                                                'gravity': state['safety']['gravity']}), flush=True)
                            return
                    raise RuntimeError('no stand/walk policy tick within 5 s; stop assisted simulation')
        except Exception as error:
            self.bootstrap_error = str(error)
            print(f'== startup assistance failed: {error}', flush=True)


class StartupWorld(sim.World):
    def step(self, times: int = 1) -> None:
        for body in self.bodies:
            error = getattr(body, 'bootstrap_error', None)
            if error:
                raise RuntimeError(error)
        super().step(times)


def main() -> None:
    # Local entry point only; monkey-patching never writes either pinned repository.
    sim.Body = StartupBody
    sim.World = StartupWorld
    print('== simulation startup assistance enabled; count balancing only after support ends',
          flush=True)
    sim.main()


if __name__ == '__main__':
    main()
