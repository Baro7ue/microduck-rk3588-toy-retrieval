# Moved 2026-10-11 from work/measure_board_runtime_resources.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Read-only Linux robotd health and process-resource sampling; no motion RPCs."""
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import signal
import socket
import struct
import time
import traceback

ROOT = Path(os.environ.get('MD_BOARD_ROOT', '/home/linaro/microduck')) / 'run'
SOCKET = str(ROOT / 'robotd.sock')


def proc_sample(pid):
    text = Path(f'/proc/{pid}/stat').read_text()
    fields = text.rsplit(')', 1)[1].split()
    return (int(fields[11]) + int(fields[12]), int(fields[19]),
            int(fields[21]) * os.sysconf('SC_PAGE_SIZE') / 1024**2)


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024
    raise RuntimeError('MemAvailable missing from /proc/meminfo')


def health(client, reader, request_id):
    started = time.monotonic()
    client.sendall((json.dumps({'jsonrpc': '2.0', 'id': request_id,
        'method': 'robot.health', 'params': {}}) + '\n').encode())
    line = reader.readline()
    if not line:
        raise RuntimeError('robotd health connection closed')
    response = json.loads(line)
    finished = time.monotonic()
    if response.get('id') != request_id or 'error' in response:
        raise RuntimeError(response)
    return response, (started + finished) / 2, (finished - started) * 1000


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Signal {signum}')


def main(duration):
    directory = ROOT / ('resources_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True)
    print(f'资源证据目录：{directory}', flush=True)
    summary = {'socket': SOCKET, 'duration_s': duration, 'outcome': 'failed',
        'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'cpu_percent_scale': '100% = one fully occupied CPU core',
        'health_rpc_rtt_note': 'Health request round-trip time, not inference latency'}
    client = reader = None
    samples = []
    status = 1
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if duration <= 0:
            raise ValueError('duration must be positive')
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(3)
        client.connect(SOCKET)
        pid, uid, gid = struct.unpack('3i', client.getsockopt(socket.SOL_SOCKET,
            socket.SO_PEERCRED, struct.calcsize('3i')))
        summary.update(pid=pid, uid=uid, gid=gid, exe=os.readlink(f'/proc/{pid}/exe'))
        _, identity, _ = proc_sample(pid)
        reader = client.makefile('rb')
        began = time.monotonic()
        previous = None
        next_print = 0
        with (directory / 'health.jsonl').open('w') as raw, \
                (directory / 'resources.csv').open('w', newline='') as table:
            columns = ['elapsed_s', 'monotonic_s', 'pid', 'healthy', 'control_loop_ticks',
                'control_loop_missed', 'control_loop_hz', 'cpu_temp_c',
                'proc_cpu_percent', 'rss_mib', 'system_mem_available_mib', 'health_rpc_rtt_ms']
            writer = csv.DictWriter(table, fieldnames=columns)
            writer.writeheader()
            while time.monotonic() - began < duration:
                response, stamp, rtt = health(client, reader, len(samples) + 1)
                raw.write(json.dumps({'monotonic_s': stamp, 'pid': pid,
                    'health_rpc_rtt_ms': rtt, 'response': response}) + '\n')
                raw.flush()
                cpu_ticks, current_identity, rss = proc_sample(pid)
                proc_stamp = time.monotonic()
                if current_identity != identity:
                    raise RuntimeError('robotd PID was reused; process identity changed')
                cpu = None if previous is None else 100 * (cpu_ticks - previous[0]) / \
                    os.sysconf('SC_CLK_TCK') / (proc_stamp - previous[1])
                previous = cpu_ticks, proc_stamp
                result = response['result']
                loop = result['control_loop']
                row = dict(zip(columns, [stamp - began, stamp, pid, result['healthy'],
                    loop['ticks'], loop['missed'], loop['achieved_hz'], result.get('cpu_temp_c'),
                    cpu, rss, available_memory(), rtt]))
                writer.writerow(row)
                table.flush()
                samples.append(row)
                if row['elapsed_s'] >= next_print:
                    cpu_label = '-' if cpu is None else f'{cpu:.2f}'
                    print(f"{row['elapsed_s']:.0f} 秒 | {loop['achieved_hz']} Hz | "
                          f"CPU {cpu_label} % | RSS {rss:.2f} MiB | 温度 {row['cpu_temp_c']} °C",
                          flush=True)
                    next_print += 60
                delay = began + len(samples) - time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, max(0, began + duration - time.monotonic())))
        summary['outcome'], status = 'completed', 0
    except BaseException:
        error = traceback.format_exc()
        (directory / 'error.txt').write_text(error)
        print(error, flush=True)
    finally:
        if reader is not None:
            reader.close()
        if client is not None:
            client.close()
        summary['samples'] = len(samples)
        if samples:
            summary['measured_elapsed_s'] = samples[-1]['monotonic_s'] - samples[0]['monotonic_s']
            summary['control_loop_ticks_delta'] = samples[-1]['control_loop_ticks'] - samples[0]['control_loop_ticks']
            summary['control_loop_missed_delta'] = samples[-1]['control_loop_missed'] - samples[0]['control_loop_missed']
            for name in ('proc_cpu_percent', 'rss_mib', 'cpu_temp_c', 'health_rpc_rtt_ms'):
                values = [row[name] for row in samples if row[name] is not None]
                if values:
                    summary[name] = {'mean': sum(values) / len(values), 'max': max(values)}
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f"资源采样结果：{summary['outcome']}；{len(samples)} 个样本；{directory}", flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=620)
    raise SystemExit(main(parser.parse_args().duration))
