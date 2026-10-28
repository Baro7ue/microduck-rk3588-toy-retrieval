"""Stage 4 board load sampler: runs ON the RK3588, standard library only (sample_board_load.zsh copies it).

    taskset -c 0-3 nice -n 5 python3 -u board_load.py --out DIR [--period 1.0] [--duration S]

Read-only. Once a period: every core's busy share from /proc/stat, MemAvailable, the CPU share (of one
core) and resident memory of robotd and of the Stage 3 brain (found by their command lines; the brain is
a new process each episode), the NPU load per core (`sudo -n cat /sys/kernel/debug/rknpu/load`, if
allowed, as Stage 2's 2F meter) and the SoC temperature. Rows go to DIR/load.csv as they are taken, with
the board's clock (t_unix); Ctrl-C or --duration ends it and writes DIR/summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

NPU_LOAD = '/sys/kernel/debug/rknpu/load'
BRAIN = 'microduck_toy.stage3.brain'
PAGE_MB = os.sysconf('SC_PAGE_SIZE') / 2 ** 20 if hasattr(os, 'sysconf') else 4096 / 2 ** 20
HZ = os.sysconf('SC_CLK_TCK') if hasattr(os, 'sysconf') else 100


def cores() -> dict[str, tuple[int, int]]:
    """{'cpu': (busy, total), 'cpu0': ..., ...} jiffies from /proc/stat."""
    out = {}
    for line in Path('/proc/stat').read_text().splitlines():
        name, *fields = line.split()
        if not re.fullmatch(r'cpu\d*', name):
            continue
        v = [int(x) for x in fields]
        idle = v[3] + (v[4] if len(v) > 4 else 0)
        out[name] = (sum(v) - idle, sum(v))
    return out


def mem_available_mb() -> float | None:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024
    return None


def find(kind: str) -> list[int]:
    """PIDs of robotd or of the brain, by command line (never this sampler)."""
    out = []
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            argv = (d / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        text = b' '.join(argv).decode(errors='replace')
        if kind == 'robotd' and argv and os.path.basename(argv[0].decode(errors='replace')) == 'robotd':
            out.append(int(d.name))
        elif kind == 'brain' and BRAIN in text and 'board_load.py' not in text:
            out.append(int(d.name))
    return sorted(out)


def proc(pid: int) -> tuple[int, float] | None:
    """(utime + stime jiffies, resident MB) of one process."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        rss = int(Path(f'/proc/{pid}/statm').read_text().split()[1]) * PAGE_MB
    except (OSError, IndexError, ValueError):
        return None
    return int(fields[11]) + int(fields[12]), rss


def npu() -> list[int] | None:
    try:
        r = subprocess.run(['sudo', '-n', 'cat', NPU_LOAD], capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        return None
    found = sorted((int(c), int(v)) for c, v in re.findall(r'Core(\d+):\s*(\d+)%', r.stdout))
    return [v for _, v in found] if r.returncode == 0 and found else None


def temp_c() -> float | None:
    try:
        return int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000.0
    except (OSError, ValueError):
        return None


class Procs:
    """CPU share of one core and RSS for a kind of process, from one sample to the next."""

    def __init__(self, kind: str):
        self.kind, self.prev = kind, {}

    def sample(self, wall: float) -> dict:
        now = {pid: p for pid in find(self.kind) if (p := proc(pid)) is not None}
        cpu = [100.0 * (now[pid][0] - self.prev[pid][0]) / HZ / wall for pid in now if pid in self.prev]
        self.prev = now
        return {f'{self.kind}_procs': len(now),
                f'{self.kind}_cpu_pct_core': round(sum(cpu), 2) if cpu else None,
                f'{self.kind}_rss_mb': round(sum(p[1] for p in now.values()), 1) if now else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--period', type=float, default=1.0)
    ap.add_argument('--duration', type=float, default=0.0, help='seconds; 0 = until Ctrl-C')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    use_npu = npu() is not None
    names = sorted(cores(), key=lambda n: (n != 'cpu', int(n[3:] or -1)))
    cols = (['t_unix'] + [f'{n}_pct' if n != 'cpu' else 'cpu_all_pct' for n in names]
            + ['mem_avail_mb', 'robotd_procs', 'robotd_cpu_pct_core', 'robotd_rss_mb',
               'brain_procs', 'brain_cpu_pct_core', 'brain_rss_mb', 'npu0_pct', 'npu1_pct', 'npu2_pct', 'temp_c'])
    robotd, brain = Procs('robotd'), Procs('brain')
    prev, t_prev = cores(), time.monotonic()
    robotd.sample(1.0)
    brain.sample(1.0)
    n, start = 0, time.monotonic()
    print(f'板端负载采样：每 {a.period} s 一行，写到 {out / "load.csv"}；NPU 负载{"能" if use_npu else "读不到，不"}记。'
          f'按 Ctrl-C 结束。', flush=True)
    try:
        with (out / 'load.csv').open('w') as f:
            f.write(','.join(cols) + '\n')
            while a.duration <= 0 or time.monotonic() - start < a.duration:
                time.sleep(a.period)
                cur, t_cur = cores(), time.monotonic()
                wall = t_cur - t_prev
                row = {'t_unix': f'{time.time():.3f}', 'mem_avail_mb': round(mem_available_mb() or 0.0, 1),
                       'temp_c': temp_c()}
                for name in names:
                    db, dt = cur[name][0] - prev[name][0], cur[name][1] - prev[name][1]
                    row[f'{name}_pct' if name != 'cpu' else 'cpu_all_pct'] = round(100.0 * db / dt, 1) if dt > 0 else None
                row.update(robotd.sample(wall))
                row.update(brain.sample(wall))
                loads = npu() if use_npu else None
                for i, v in enumerate((loads or [])[:3]):
                    row[f'npu{i}_pct'] = v
                f.write(','.join('' if row.get(c) is None else str(row[c]) for c in cols) + '\n')
                f.flush()
                prev, t_prev, n = cur, t_cur, n + 1
    except KeyboardInterrupt:
        pass
    summary = {'rows': n, 'period_s': a.period, 'npu_load': use_npu, 'columns': cols,
               'pid': os.getpid(), 'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(f'已结束：{n} 行。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
