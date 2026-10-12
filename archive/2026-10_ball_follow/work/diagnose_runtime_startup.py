"""A/B startup diagnostic in isolated headless worlds, never the user's live socket."""
from __future__ import annotations

import datetime as dt
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time

from mjlab_microduck.sim import body_server as sim
from runtime_bootstrap_body import StartupBody, StartupWorld

PROJECT = Path('/Users/ethan/Documents/RK3588 Microduck')
REPO = PROJECT / 'task/microduck'
ORT = PROJECT / 'task/microduck_r1/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.24.4.dylib'


def call(path: str, method: str, params: dict) -> dict:
    assert path.startswith('/private/tmp/md-boot-')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(2.0)
        client.connect(path)
        client.sendall((json.dumps({'jsonrpc': '2.0', 'id': 1,
                                   'method': method, 'params': params}) + '\n').encode())
        with client.makefile('rb') as stream:
            reply = json.loads(stream.readline())
    if 'error' in reply:
        raise RuntimeError(reply)
    return reply['result']


def run_arm(directory: Path, assisted: bool, walk_probe: bool = False) -> dict:
    mode = 'startup_support' if assisted else 'original'
    arm_dir = directory / mode
    arm_dir.mkdir()
    world = (StartupWorld if assisted else sim.World)(sim.DEFAULT_SCENE)
    body = (StartupBody if assisted else sim.Body)(world, 0)
    body.place(None, sim.HOME_TRUNK_Z, 0.0)
    world.bodies.append(body)
    sim.mujoco.mj_forward(world.model, world.data)
    server = sim.Server(('127.0.0.1', 0), sim.Handler)
    server.body = body
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    stop = threading.Event()

    def physics() -> None:
        next_step = time.perf_counter()
        while not stop.is_set():
            world.step(4)
            next_step += 0.020
            stop.wait(max(0.0, next_step - time.perf_counter()))

    physics_thread = threading.Thread(target=physics, daemon=True)
    physics_thread.start()
    process = None
    client = None
    stream = None
    attempted = False
    report = {'mode': mode, 'assistance': assisted, 'policy_seen': False,
              'post_release_s': 0.0, 'fallen': False}
    with tempfile.TemporaryDirectory(prefix='md-boot-', dir='/private/tmp') as state_dir, \
            (arm_dir / 'robotd.log').open('w') as log, \
            (arm_dir / 'states.jsonl').open('w') as trace:
        sock_path = state_dir + '/robotd.sock'
        if assisted:
            body.startup_socket = sock_path
        try:
            env = {**os.environ, 'ORT_DYLIB_PATH': str(ORT), 'DUCK_RUNTIME_DIR': state_dir}
            process = subprocess.Popen([str(REPO / 'target/debug/robotd'), '--sim',
                f'127.0.0.1:{port}', '--params', str(PROJECT / 'work/runtime_robotd.toml'),
                '--socket', sock_path], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 8.0
            while not Path(sock_path).exists():
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('isolated robotd did not start')
                time.sleep(0.05)
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(2.0)
            client.connect(sock_path)
            stream = client.makefile('rb')
            client.sendall((json.dumps({'jsonrpc': '2.0', 'id': 1,
                'method': 'robot.subscribe', 'params': {'hz': 50}}) + '\n').encode())
            while True:
                packet = json.loads(stream.readline())
                if packet.get('method') == 'robot.state':
                    break
            attempted = True
            report['enable'] = call(sock_path, 'robot.enable', {'on': True})
            started = time.monotonic()
            release_time = None
            labels = []
            while time.monotonic() - started < 14.0:
                line = stream.readline()
                if not line:
                    raise RuntimeError('isolated state stream closed')
                packet = json.loads(line)
                if packet.get('method') != 'robot.state':
                    continue
                state = packet['params']
                elapsed = time.monotonic() - started
                if not labels or labels[-1]['label'] != state['policy']:
                    labels.append({'elapsed_s': elapsed, 'label': state['policy']})
                if state['policy'] in {'stand', 'walk'} and release_time is None:
                    report['policy_seen'] = True
                    release_time = time.monotonic()
                    report['policy_start_s'] = elapsed
                with world.lock:
                    trunk_z = float(world.data.qpos[body.trunk + 2])
                    supported = not body.released
                    sim_time = float(world.data.time)
                row = {'elapsed_s': elapsed, 'supported': supported, 'sim_time': sim_time,
                       'trunk_z': trunk_z, 'state': state}
                trace.write(json.dumps(row) + '\n')
                report['elapsed_s'] = elapsed
                report['last'] = row
                if release_time is not None:
                    report['post_release_s'] = time.monotonic() - release_time
                if state['safety']['fallen']:
                    report['fallen'] = True
                    break
                if report['post_release_s'] >= (2.0 if walk_probe else 10.0):
                    if walk_probe:
                        # Retire this reader so the walking probe is the only state subscriber.
                        stream.close()
                        stream = None
                        client.close()
                        client = None
                        gait_log = arm_dir / 'walk_probe.log'
                        with gait_log.open('w') as output:
                            trial = subprocess.run([
                                str(PROJECT / 'task/microduck_r1/.venv/bin/python'), '-u',
                                str(PROJECT / 'work/probe_runtime_walk.py'), '--socket', sock_path],
                                cwd=PROJECT, stdout=output, stderr=subprocess.STDOUT, timeout=25.0)
                        report['walk_probe_returncode'] = trial.returncode
                        report['walk_probe_log'] = str(gait_log)
                        with world.lock:
                            report['walk_final_trunk_z'] = float(world.data.qpos[body.trunk + 2])
                    break
            report['labels'] = labels
        finally:
            if attempted:
                try:
                    report['disable'] = call(sock_path, 'robot.enable', {'on': False})
                except Exception as error:
                    report['disable_error'] = str(error)
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if stream is not None:
                stream.close()
            if client is not None:
                client.close()
            stop.set()
            physics_thread.join(timeout=2.0)
            server.shutdown()
            server.server_close()
    (arm_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    compact = {k: report.get(k) for k in ('mode', 'policy_seen', 'policy_start_s',
                                         'post_release_s', 'fallen', 'elapsed_s', 'labels',
                                         'walk_probe_returncode', 'walk_probe_log')}
    print(json.dumps(compact, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--walk-probe', action='store_true')
    args = parser.parse_args()
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory = PROJECT / 'task/evidence/G2' / f'startup_diagnostic_{stamp}'
    directory.mkdir()
    print(f'evidence: {directory}', flush=True)
    results = [run_arm(directory, True, walk_probe=True)] if args.walk_probe else [
        run_arm(directory, False), run_arm(directory, True)]
    (directory / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
