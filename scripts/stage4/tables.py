"""Stage 4 result tables (plan claude-docs/stage4_plan_20261028/, sections 1, 2 and 5).

    scripts/stage4/run_tables.zsh [--runs DIR] [--out DIR] [--onef RUN_DIR] [--dev]
    (= python scripts/stage4/tables.py ..., with env.sh's PYTHONPATH)

Reads the Stage 4 batch folders (runs/batch_*, scripts/stage3/run_batch.zsh) and the 4D 1F run, and
writes the report's tables and figures into results/stage4/ (default):

    stage4_facts.json   every number below, each with the folders it came from; Stage 5 reads only this
    episodes.csv        one row per counted episode (all groups)
    failures.csv        T3: every counted episode that did not deliver, with an automatic cause
    t1_main.csv t2_per_target.csv t4_detection.csv t5_deployment.csv t6_last_segment.csv t7_timing.csv
    figures/            F1 failure traces, F2 robot's-eye frames, F3 detection latency, F4 park error

The rules are the ones the plan fixed before any formal seed ran (section 2):
  * a group is the batch folders with the same target, backend and brain --eyes (yolo unless the batch
    passed `-- --eyes colour`) on the formal layouts (configs/control/stage1_layouts.json, --formal);
    several folders of a group are merged;
  * only an episode folder the runner counts as done is a result (batch.status); attempts that did not
    finish are counted, never scored; an episode whose attempts ended in a brain error
    REPEATED_ERROR_MIN times or more counts as a failure, end 'error';
  * if two folders hold a result for the same episode, the earlier one counts;
  * delivered rate with a Wilson 95% interval; two groups on the same episodes: McNemar's exact test;
  * the 4D run is the earliest complete 1F run (30 + 30 starts) after the freeze whose sim kept real time.
Every output is deterministic (no clock, repo-relative paths), so the Mac and the cloud must write the
same JSON and CSV files from the same folders. Labels are English for the report.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import shlex
import sys

from microduck_toy.paths import repo_root, runs_dir
from microduck_toy.stage1 import metrics
from microduck_toy.stage3 import batch, episode

FORMAL_LAYOUTS = 'configs/control/stage1_layouts.json'
DEV_LAYOUTS = 'configs/control/stage3_layouts.json'
FREEZE_UTC = '20261028T030300Z'            # tag stage3-freeze (cc0931a)
GROUPS = {                                 # name: step, target, backend, eyes, seeds (None = every formal seed)
    'board_int8': ('4A', 'board', 'int8', 'yolo', None),
    'board_fp16': ('4B', 'board', 'fp16', 'yolo', (1000, 1004)),
    'board_cpu': ('4B', 'board', 'cpu', 'yolo', (1000, 1004)),
    'mac_cpu': ('4C', 'mac', 'cpu', 'yolo', None),
    'mac_colour': ('4E', 'mac', 'cpu', 'colour', None),   # run_batch.zsh ... -- --eyes colour
}
LABELS = {'board_int8': 'RK3588, YOLO INT8 (NPU)', 'board_fp16': 'RK3588, YOLO FP16 (NPU)',
          'board_cpu': 'RK3588, YOLO FP32 (CPU)', 'mac_cpu': 'Mac, YOLO FP32 (CPU)',
          'mac_colour': 'Mac, colour threshold (toys) + YOLO FP32 (tray)'}
PAIRS = (('board_int8', 'mac_cpu'), ('mac_colour', 'mac_cpu'), ('board_fp16', 'board_int8'),
         ('board_cpu', 'board_int8'))
HEADLINE = 'board_int8'
HEADLINE_MIN = 42                          # plan section 2: >= 42/60 delivered (70%)
REPEATED_ERROR_MIN = 3
BRAIN_ERROR = '大脑出错'                    # batch.status's reason for a brain error
ONEF_RUN = re.compile(r'^(?P<utc>\d{8}T\d{6}Z)_s1f_')
ONEF_STARTS = 30
LAT_KEYS = ('e2e', 'preprocess', 'inference', 'postprocess', 'receive', 'wait')


# ── statistics ───────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    """Wilson score interval for k successes in n (plan section 5)."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(k: int, n: int) -> dict:
    return {'k': k, 'n': n, 'rate': round(k / n, 4) if n else None, 'wilson95': wilson(k, n)}


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p: b and c are the discordant pairs (A only, B only)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def mean_sd(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None]
    m = sum(values) / len(values)
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return [round(m, 4), round(sd, 4)]


def median(values: list[float]) -> float | None:
    v = sorted(values)
    if not v:
        return None
    k = len(v) // 2
    return round(v[k] if len(v) % 2 else (v[k - 1] + v[k]) / 2, 3)


# ── reading batch folders ──────────────────────────────────────────────────

def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root().resolve()))
    except ValueError:
        return str(path)


def expected_episodes(group: str, layouts: str) -> list[tuple[int, str]]:
    seeds = GROUPS[group][4]
    have = sorted(l['seed'] for l in episode.load_layouts(layouts))
    chosen = [s for s in have if seeds is None or seeds[0] <= s <= seeds[1]]
    return [(s, t) for s in chosen for t in episode.TARGETS]


def batch_folders(runs: Path) -> list[Path]:
    return sorted(d for d in runs.iterdir() if d.is_dir() and d.name.startswith('batch_')
                  and (d / 'batch.json').exists())


def eyes_of(extra: str | None) -> str:
    """The brain's --eyes in a batch's extra episode options (run_batch.zsh -- ...); yolo when not given."""
    words = shlex.split(extra or '')
    eyes = 'yolo'
    for i, w in enumerate(words):
        if w == '--eyes' and i + 1 < len(words):
            eyes = words[i + 1]
        elif w.startswith('--eyes='):
            eyes = w.split('=', 1)[1]
    return eyes


def group_of(settings: dict, layouts: str, formal: bool) -> str | None:
    """The group a batch belongs to; the layouts file is matched by its sha256, however it was typed."""
    if settings.get('layouts_sha256') != batch.file_sha256(layouts) or bool(settings.get('formal')) != formal:
        return None
    eyes = eyes_of(settings.get('extra'))
    for name, (_, target, backend, want, _) in GROUPS.items():
        if settings.get('target') == target and settings.get('backend') == backend and eyes == want:
            return name
    return None


def episode_row(group: str, ep_dir: Path) -> dict:
    """One counted episode: the scorer's result for part e plus the run's health, vision and timing."""
    s = json.loads((ep_dir / 'summary.json').read_text())
    r = next(r for r in s['results'] if r.get('part') == 'e')
    trial = next((t for t in s.get('trials') or [] if t.get('part', 'e') == 'e'), {})
    lat = (trial.get('brain') or {}).get('latency') or {}
    health = s.get('sim_health') or {}
    vis = s.get('vision_loop') or {}
    park = [[round(f, 4), round(l, 4)] for f, l in r.get('park_err_m') or []]
    closes = r.get('closes') or []
    label = r.get('label') or f'e_{r["seed"]}_{r["target"]}'
    trace = ep_dir / 'episodes' / label / 'trace.png'
    frames = ep_dir / 'episodes' / label / 'frames'
    row = {'group': group, 'seed': r['seed'], 'target': r['target'], 'outcome': r['outcome'],
           'delivered': r['outcome'] == 'delivered',
           'delivered_strict': bool(r.get('delivered_strict')) and r['outcome'] == 'delivered',
           'elapsed_s': r.get('elapsed_s'), 'closes': len(closes), 'grasped': r.get('grasped', 0),
           'wrong_grabs': r.get('wrong_grabs', 0), 'falls': len(r.get('falls') or []),
           'getup_s': [f.get('up_after_s') for f in r.get('falls') or []],
           'tof_stops': len(r.get('tof_stops') or []),
           'false_tof_stops': sum(bool(t.get('false')) for t in r.get('tof_stops') or []),
           'park_err_m': park, 'close_results': [c.get('result') for c in closes],
           'held_toys': r.get('held_toys') or [], 'others_in_tray': r.get('others_in_tray') or [],
           'target_from_tray_m': r.get('target_from_tray_m'),
           'tray_contact': bool(r.get('tray_contact')), 'states_s': r.get('states_s') or {},
           'handover': r.get('handover'), 'confirm_s': r.get('confirm_s'),
           'latency_ms': {k: {q: (lat.get(k) or {}).get(q) for q in ('p50', 'p99')}
                          for k in LAT_KEYS if lat.get(k)},
           'latency_frames': lat.get('frames'),
           'robotd': {k: (r.get('robotd') or {}).get(k) for k in ('ticks_expected', 'missed', 'hz_min')}
           if r.get('robotd') else None,
           'sim_stalled': health.get('stalled'), 'sim_odom_ratio': (health.get('odom_ratio') or {}).get('ratio'),
           'vision': {k: vis.get(k) for k in ('recall_targets', 'precision_targets', 'wrong_class_within_1_6m')}
           if vis.get('frames') else None,
           'started_utc': (s.get('meta') or {}).get('started_utc'),
           'folder': rel(ep_dir), 'trace': rel(trace) if trace.exists() else None,
           'frames': rel(frames) if frames.is_dir() else None}
    return row


def error_row(group: str, seed: int, target: str, attempts: list[dict]) -> dict:
    return {'group': group, 'seed': seed, 'target': target, 'outcome': 'error', 'delivered': False,
            'delivered_strict': False, 'elapsed_s': None, 'closes': 0, 'grasped': 0, 'wrong_grabs': 0,
            'falls': 0, 'getup_s': [], 'tof_stops': 0, 'false_tof_stops': 0, 'park_err_m': [],
            'close_results': [], 'held_toys': [], 'others_in_tray': [], 'target_from_tray_m': None,
            'tray_contact': False, 'states_s': {}, 'handover': None, 'confirm_s': None, 'latency_ms': {},
            'latency_frames': None, 'robotd': None, 'sim_stalled': None, 'sim_odom_ratio': None,
            'vision': None, 'started_utc': None, 'folder': attempts[0]['folder'], 'trace': None,
            'frames': None, 'error_attempts': [a['why'] for a in attempts]}


def collect(runs: Path, layouts: str, formal: bool) -> dict:
    """Per group: the batch folders, the counted episodes and everything that was not counted."""
    out = {}
    for d in batch_folders(runs):
        doc = json.loads((d / 'batch.json').read_text())
        name = group_of(doc['settings'], layouts, formal)
        if name is None:
            continue
        g = out.setdefault(name, {'folders': [], 'done': {}, 'duplicates': [], 'attempts': {}})
        g['folders'].append({'folder': rel(d), 'code_sha256': doc['settings'].get('code_sha256'),
                             'layouts_sha256': doc['settings'].get('layouts_sha256'),
                             'extra': doc['settings'].get('extra'),
                             'commits': sorted({c.get('commit') for c in doc['invocations'] if c.get('commit')}),
                             'dirty': any(c.get('dirty') for c in doc['invocations']),
                             'invocations': len(doc['invocations']), 'planned': doc['settings']['episodes']})
        for item in doc['settings']['episodes']:
            seed, target = item.split(':')
            key = (int(seed), target)
            ep_dir = d / 'episodes' / f'{seed}_{target}'
            if ep_dir.exists():
                st, why = batch.status(ep_dir)
                if st == 'done':
                    row = episode_row(name, ep_dir)
                    old = g['done'].get(key)
                    if old is None or (row['started_utc'] or '') < (old['started_utc'] or ''):
                        if old is not None:
                            g['duplicates'].append(old['folder'])
                        g['done'][key] = row
                    else:
                        g['duplicates'].append(row['folder'])
                else:
                    g['attempts'].setdefault(key, []).append({'folder': rel(ep_dir), 'status': st, 'why': why})
        for f in batch.failed_attempts(d):
            seed, target = f['episode'].split('_')
            g['attempts'].setdefault((int(seed), target), []).append(
                {'folder': rel(d / 'episodes' / f['folder']), 'status': f['status'], 'why': f['why']})
    return out


# ── group summaries ─────────────────────────────────────────────────────────

def counted_rows(name: str, g: dict, expected: list[tuple[int, str]]) -> tuple[list[dict], list[dict]]:
    rows, missing = [], []
    for key in expected:
        if key in g['done']:
            rows.append(g['done'][key])
            continue
        tries = g['attempts'].get(key, [])
        errors = [a for a in tries if a['why'].startswith(BRAIN_ERROR)]
        if len(errors) >= REPEATED_ERROR_MIN:
            rows.append(error_row(name, key[0], key[1], errors))
        else:
            missing.append({'episode': f'{key[0]}_{key[1]}', 'attempts': len(tries)})
    return rows, missing


def outcome_stats(rows: list[dict]) -> dict:
    n = len(rows)
    ends: dict[str, int] = {}
    for r in rows:
        ends[r['outcome']] = ends.get(r['outcome'], 0) + 1
    timed = [r['elapsed_s'] for r in rows if r['elapsed_s'] is not None]
    park = [p for r in rows for p in r['park_err_m']]
    first = [r['park_err_m'][0] for r in rows if r['park_err_m']]
    closes = sum(r['closes'] for r in rows)
    grasped = sum(r['grasped'] for r in rows)
    out = {'episodes': n,
           'delivered': rate(sum(r['delivered'] for r in rows), n),
           'delivered_strict': rate(sum(r['delivered_strict'] for r in rows), n),
           'wrong_toy': sum(r['outcome'] == 'wrong_toy' for r in rows),
           'ends': dict(sorted(ends.items())),
           'mean_s': round(sum(timed) / len(timed), 1) if timed else None,
           'falls': sum(r['falls'] for r in rows),
           'getup_s_max': max((s for r in rows for s in r['getup_s'] if s is not None), default=None),
           'tof_stops': sum(r['tof_stops'] for r in rows),
           'false_tof_stops': sum(r['false_tof_stops'] for r in rows),
           'tray_contact': sum(r['tray_contact'] for r in rows),
           'closes': closes, 'grasped': grasped,
           'success_per_close': round(grasped / closes, 4) if closes else None,
           'reached_pick': sum(r['closes'] > 0 for r in rows),
           'held': sum(r['grasped'] > 0 for r in rows),
           'wrong_grabs': sum(r['wrong_grabs'] for r in rows),
           'park_err_cm': {'n': len(park),
                           'forward_mean_sd': [None if v is None else round(v * 100, 2) for v in mean_sd([p[0] for p in park])],
                           'lateral_mean_sd': [None if v is None else round(v * 100, 2) for v in mean_sd([p[1] for p in park])],
                           'rms': round(100 * math.sqrt(sum(a * a + b * b for a, b in park) / len(park)), 2) if park else None,
                           'first_close_lateral_mean_sd': [None if v is None else round(v * 100, 2)
                                                           for v in mean_sd([p[1] for p in first])]}}
    return out


def timing_stats(rows: list[dict]) -> dict:
    timed = [r for r in rows if r['states_s']]
    names = sorted({k for r in timed for k in r['states_s']})
    out = {'state_s_mean': {k: round(sum(r['states_s'].get(k, 0.0) for r in timed) / len(timed), 2)
                            for k in names} if timed else {},
           'latency_ms': {}}
    for k in LAT_KEYS:
        p50 = [r['latency_ms'][k]['p50'] for r in rows if k in r['latency_ms'] and r['latency_ms'][k]['p50'] is not None]
        p99 = [r['latency_ms'][k]['p99'] for r in rows if k in r['latency_ms'] and r['latency_ms'][k]['p99'] is not None]
        if p50:
            out['latency_ms'][k] = {'episodes': len(p50), 'p50_median': median(p50), 'p99_median': median(p99),
                                    'p99_worst': max(p99) if p99 else None}
    return out


def board_stats(rows: list[dict]) -> dict | None:
    ticks = [r['robotd'] for r in rows if r.get('robotd')]
    if not ticks:
        return None
    d = episode.stage2_deploy()
    total = sum(t.get('ticks_expected') or 0 for t in ticks)
    missed = sum(t.get('missed') or 0 for t in ticks)
    rate0 = d['stage0_robotd']['missed'] / d['stage0_robotd']['ticks']
    p99 = [r['latency_ms']['e2e']['p99'] for r in rows if 'e2e' in r['latency_ms']
           and r['latency_ms']['e2e']['p99'] is not None]
    limit = float(d['e2e_p99_ms_max'])
    return {'robotd_ticks': total, 'robotd_missed': missed, 'robotd_missed_allowed': max(1, math.ceil(rate0 * total)),
            'robotd_hz_min': min((t['hz_min'] for t in ticks if t.get('hz_min')), default=None),
            'e2e_p99_ms_worst': round(max(p99), 3) if p99 else None, 'e2e_p99_ms_limit': limit,
            'episodes_over_limit': sum(v > limit for v in p99), 'latency_logs': len(p99)}


def vision_stats(rows: list[dict]) -> dict | None:
    vis = [r['vision'] for r in rows if r.get('vision')]
    if not vis:
        return None

    def pool(key):
        k = sum((v.get(key) or {}).get('k') or 0 for v in vis)
        n = sum((v.get(key) or {}).get('n') or 0 for v in vis)
        return rate(k, n)
    return {'episodes': len(vis), 'recall_targets': pool('recall_targets'),
            'precision_targets': pool('precision_targets'),
            'wrong_class_within_1_6m': sum(v.get('wrong_class_within_1_6m') or 0 for v in vis)}


def health_stats(rows: list[dict]) -> dict:
    st = [r['sim_stalled'] for r in rows if r['sim_stalled'] is not None]
    od = [r['sim_odom_ratio'] for r in rows if r['sim_odom_ratio'] is not None]
    return {'stalled_max': max(st) if st else None,
            'odom_ratio_min': min(od) if od else None, 'odom_ratio_max': max(od) if od else None}


def summarise_group(name: str, g: dict, layouts: str, dev: bool) -> tuple[dict, list[dict]]:
    expected = (sorted(set(g['done']) | set(g['attempts'])) if dev else expected_episodes(name, layouts))
    rows, missing = counted_rows(name, g, expected)
    attempts = [a for key in sorted(g['attempts']) for a in g['attempts'][key]]
    by_status: dict[str, int] = {}
    for a in attempts:
        by_status[a['status']] = by_status.get(a['status'], 0) + 1
    per_target = {}
    for t in episode.TARGETS:
        mine = [r for r in rows if r['target'] == t]
        if mine:
            per_target[t] = {'delivered': rate(sum(r['delivered'] for r in mine), len(mine)),
                             'wrong_toy': sum(r['outcome'] == 'wrong_toy' for r in mine),
                             'ends': outcome_stats(mine)['ends']}
    shas = sorted({f['code_sha256'] for f in g['folders'] if f['code_sha256']})
    out = {'step': GROUPS[name][0], 'label': LABELS[name], 'target': GROUPS[name][1], 'backend': GROUPS[name][2],
           'eyes': GROUPS[name][3],
           'folders': [{k: v for k, v in f.items() if k != 'planned'} for f in g['folders']],
           'code_sha256': shas, 'expected': len(expected), 'counted': len(rows), 'not_counted': missing,
           'attempts_not_counted': {'total': len(attempts), 'by_status': dict(sorted(by_status.items())),
                                    'list': attempts},
           'repeated_errors': [f"{r['seed']}_{r['target']}" for r in rows if r['outcome'] == 'error'],
           'duplicate_results_ignored': sorted(g['duplicates']),
           **outcome_stats(rows), 'per_target': per_target, 'timing': timing_stats(rows),
           'board': board_stats(rows), 'vision_loop': vision_stats(rows), 'sim_health': health_stats(rows)}
    return out, rows


def paired(rows_a: list[dict], rows_b: list[dict]) -> dict:
    a = {(r['seed'], r['target']): r['delivered'] for r in rows_a}
    b = {(r['seed'], r['target']): r['delivered'] for r in rows_b}
    both = sorted(set(a) & set(b))
    only_a = sum(a[k] and not b[k] for k in both)
    only_b = sum(b[k] and not a[k] for k in both)
    return {'pairs': len(both), 'a_delivered': sum(a[k] for k in both), 'b_delivered': sum(b[k] for k in both),
            'a_only': only_a, 'b_only': only_b, 'mcnemar_p': mcnemar(only_a, only_b),
            'a_only_episodes': [f'{s}_{t}' for s, t in both if a[(s, t)] and not b[(s, t)]],
            'b_only_episodes': [f'{s}_{t}' for s, t in both if b[(s, t)] and not a[(s, t)]]}


# ── failures (T3) ───────────────────────────────────────────────────────────

def cm(v) -> str:
    return f'{100 * v:+.1f}'


def cause(r: dict) -> str:
    """A first, automatic line for T3; the person reading the trace writes the real one."""
    o = r['outcome']
    if o == 'error':
        return f'brain error in {len(r.get("error_attempts", []))} attempts: {r["error_attempts"][0][:80]}'
    if o == 'grasp_fail':
        parks = ', '.join(f'({cm(f)}, {cm(l)})' for f, l in r['park_err_m'])
        return (f'{r["closes"]} closes, none held; park error (forward, lateral) cm: {parks or "-"}'
                + ('; + lateral = parked right of the toy' if r['park_err_m'] else ''))
    if o == 'drop_missed':
        dx = r['target_from_tray_m'] or [None, None, None]
        where = f'{dx[0]:+.3f}, {dx[1]:+.3f} m from the tray centre' if dx[0] is not None else 'outside'
        return f'held, dropped outside the tray ({where})' + ('; touched the rim' if r['tray_contact'] else '')
    if o == 'wrong_toy':
        held = [t for t in r['held_toys'] if t != r['target']]
        return f'took the wrong toy: {", ".join(held + r["others_in_tray"]) or "?"}'
    if o == 'search_fail':
        h = r.get('handover') or {}
        return 'never confirmed the target' + (' (handed over to a distractor)' if h.get('distractor') else '')
    if o == 'timeout':
        last = max(r['states_s'], key=r['states_s'].get) if r['states_s'] else '?'
        return f'180 s ran out; most time in {last}'
    return o


def failures(rows_by_group: dict[str, list[dict]], notes: dict) -> list[dict]:
    out = []
    for name, rows in rows_by_group.items():
        for r in rows:
            if r['delivered']:
                continue
            key = f'{name}/{r["seed"]}_{r["target"]}'
            out.append({'group': name, 'seed': r['seed'], 'target': r['target'], 'outcome': r['outcome'],
                        'closes': r['closes'], 'falls': r['falls'], 'auto_cause': cause(r),
                        'note': notes.get(key, ''), 'trace': r['trace'], 'folder': r['folder']})
    return out


# ── 4D: the 1F rerun ────────────────────────────────────────────────────────

def onef_candidates(runs: Path) -> list[Path]:
    out = []
    for d in sorted(runs.iterdir()):
        m = ONEF_RUN.match(d.name)
        if not (m and d.is_dir() and m['utc'] > FREEZE_UTC and (d / 'summary.json').exists()):
            continue
        try:
            res = json.loads((d / 'summary.json').read_text()).get('results') or {}
        except ValueError:
            continue
        if isinstance(res, dict) and all(len(res.get(p) or []) == ONEF_STARTS for p in ('f_main', 'f_blind')):
            out.append(d)
    return out


def onef_health(run: Path) -> dict:
    rows = metrics.read_truth(run / 'truth.csv') if (run / 'truth.csv').exists() else []
    if not rows:
        return {'ok': False, 'why': 'no truth.csv'}
    h = episode.sim_health(rows, run / 'state.jsonl')
    return {k: h.get(k) for k in ('ok', 'stalled', 'steps')} | {'odom_ratio': (h.get('odom_ratio') or {}).get('ratio')}


def onef_part(results: list[dict]) -> dict:
    picks = [pk for r in results for pk in r['picks'] if pk.get('ran')]
    park = [pk['park_err_m'] for r in results for pk in r['picks'] if pk.get('park_err_m')]
    grasped = sum(pk['grasped'] for pk in picks)
    first = [r for r in results if any(pk.get('ran') for pk in r['picks'])]
    return {'starts': len(results), 'closes': len(picks), 'grasped': grasped,
            'success_per_close': round(grasped / len(picks), 4) if picks else None,
            'first_close_held': rate(sum(next(pk for pk in r['picks'] if pk.get('ran'))['grasped'] for r in first),
                                     len(results)),
            'held_within_3': rate(sum(bool(r['held_within_3']) for r in results), len(results)),
            'park_err_cm': {'n': len(park),
                            'forward_mean_sd': [None if v is None else round(v * 100, 2) for v in mean_sd([p[0] for p in park])],
                            'lateral_mean_sd': [None if v is None else round(v * 100, 2) for v in mean_sd([p[1] for p in park])],
                            'rms': round(100 * math.sqrt(sum(a * a + b * b for a, b in park) / len(park)), 2) if park else None},
            'mean_start_s': round(sum(r['start_s'] for r in results) / len(results), 1) if results else None,
            'falls': sum(bool(r.get('fell')) for r in results)}


def last_segment(runs: Path, chosen: Path | None) -> dict:
    tried = []
    candidates = [chosen] if chosen else onef_candidates(runs)
    for run in candidates:
        health = onef_health(run)
        tried.append({'run': rel(run), 'sim_health': health})
        if health.get('ok'):
            s = json.loads((run / 'summary.json').read_text())
            res = s['results']
            main = {r['i']: r for r in res['f_main']}
            blind = {r['i']: r for r in res['f_blind']}
            both = sorted(set(main) & set(blind))
            b_only = sum(bool(blind[i]['held_within_3']) and not main[i]['held_within_3'] for i in both)
            m_only = sum(bool(main[i]['held_within_3']) and not blind[i]['held_within_3'] for i in both)
            return {'run': rel(run), 'commit': (s.get('meta') or {}).get('repo_commit'), 'sim_health': health,
                    'tried': tried, 'eyes': 'the arena body\'s segmentation boxes (Stage 1 plan 1A), not YOLO',
                    'inching': onef_part(res['f_main']), 'blind': onef_part(res['f_blind']),
                    'paired_held_within_3': {'pairs': len(both), 'blind_only': b_only, 'inching_only': m_only,
                                             'mcnemar_p': mcnemar(b_only, m_only)},
                    'starts': [{'i': i, 'toy': main[i]['toy'], 'inching_held_within_3': bool(main[i]['held_within_3']),
                                'blind_held_within_3': bool(blind[i]['held_within_3']),
                                'inching_closes': main[i]['closes'], 'blind_closes': blind[i]['closes']}
                               for i in both]}
    return {'run': None, 'tried': tried}


# ── offline rows (Stage 2) for T4 and T5 ─────────────────────────────────────

def stage2_facts() -> dict:
    return json.loads((repo_root() / 'configs' / 'vision' / 'stage2_facts.json').read_text())


def detection_table(groups: dict) -> list[dict]:
    acc = stage2_facts()['accuracy_test_far']
    names = {'coco': 'pretrained YOLO11n (COCO)', 'colour': 'colour threshold (HSV)',
             'fp32_cloud': 'fine-tuned YOLO11n FP32', 'int8_board': 'fine-tuned YOLO11n INT8 (board NPU)'}
    out = []
    for key, label in names.items():
        a = acc[key]
        out.append({'detector': label, 'source': 'offline, Stage 2 test + test_far (2100 frames)',
                    'targets_map50': a['targets_map50'], 'search_radius_m': a['search_radius_m'],
                    'wrong_grab_near': a['wrong_grab_near']})
    for name in ('mac_cpu', 'mac_colour'):
        g = groups.get(name)
        if not g:
            continue
        v = g.get('vision_loop') or {}
        henry = g['per_target'].get('henry', {}).get('delivered', {})
        out.append({'detector': LABELS[name], 'source': f'closed loop, {g["counted"]} formal episodes',
                    'delivered': g['delivered'], 'wrong_toy': g['wrong_toy'], 'henry_delivered': henry,
                    'recall_targets': (v.get('recall_targets') or {}).get('rate'),
                    'precision_targets': (v.get('precision_targets') or {}).get('rate'),
                    'wrong_class_within_1_6m': v.get('wrong_class_within_1_6m')})
    return out


def subset(rows: list[dict], seeds: tuple[int, int]) -> list[dict]:
    return [r for r in rows if seeds[0] <= r['seed'] <= seeds[1]]


def deployment_table(groups: dict, rows_by_group: dict, load: dict) -> list[dict]:
    dep = stage2_facts()['deploy_board']['backends']
    out = []
    for key in ('cpu', 'fp16', 'int8'):
        b = dep[key]
        out.append({'backend': key, 'source': 'offline 2F (standing and turning, 1000 frames, no brain)',
                    'e2e_ms_p50_p99': b['e2e_ms_p50_p99'], 'inference_ms_p50_p99': b['inference_ms_p50_p99'],
                    'preprocess_ms_p50_p99': b['preprocess_ms_p50_p99'],
                    'system_cpu_pct': b['system_cpu_pct'], 'robotd_missed': b['robotd_missed']})
    seeds = GROUPS['board_fp16'][4]
    for name in ('board_cpu', 'board_fp16', 'board_int8'):
        rows = rows_by_group.get(name)
        if not rows:
            continue
        rows = subset(rows, seeds)
        lat = timing_stats(rows)['latency_ms']
        bd = board_stats(rows) or {}
        out.append({'backend': GROUPS[name][2], 'source': f'closed loop, seeds {seeds[0]}-{seeds[1]} ({len(rows)} episodes)',
                    'delivered': rate(sum(r['delivered'] for r in rows), len(rows)),
                    'e2e_ms_p50_median': (lat.get('e2e') or {}).get('p50_median'),
                    'e2e_ms_p99_worst': (lat.get('e2e') or {}).get('p99_worst'),
                    'inference_ms_p50_median': (lat.get('inference') or {}).get('p50_median'),
                    'preprocess_ms_p50_median': (lat.get('preprocess') or {}).get('p50_median'),
                    'robotd_missed': bd.get('robotd_missed'), 'robotd_ticks': bd.get('robotd_ticks'),
                    'load': load.get(name)})
    return out


# ── board load samples (scripts/stage4/sample_board_load.zsh) ───────────────

def load_runs(runs: Path) -> list[Path]:
    return sorted(d for d in runs.iterdir() if d.is_dir() and d.name.endswith('_board_load')
                  and (d / 'meta.json').exists())


def read_load(run: Path) -> list[dict]:
    """The sampler's rows, with the board clock moved onto the Mac's (meta.json offset_s)."""
    meta = json.loads((run / 'meta.json').read_text())
    off = float(meta.get('offset_s') or 0.0)
    path = next(iter(sorted(run.rglob('load.csv'))), None)
    if path is None:
        return []
    out = []
    with path.open(newline='') as f:
        for raw in csv.DictReader(f):
            row = {'t': float(raw['t_unix']) + off}
            for k, v in raw.items():
                if k != 't_unix' and v not in ('', None):
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            out.append(row)
    return out


def utc_seconds(stamp: str) -> float:
    import datetime as dt
    return dt.datetime.strptime(stamp, '%Y%m%dT%H%M%SZ').replace(tzinfo=dt.timezone.utc).timestamp()


def load_by_group(runs: Path, rows_by_group: dict) -> dict:
    """Mean per-core CPU, NPU load and memory over each board group's episodes (sample inside an
    episode's [start, start + elapsed + 15 s])."""
    samples = [s for run in load_runs(runs) for s in read_load(run)]
    if not samples:
        return {}
    out = {}
    for name, rows in rows_by_group.items():
        if GROUPS[name][1] != 'board':
            continue
        wins = [(utc_seconds(r['started_utc']), utc_seconds(r['started_utc']) + (r['elapsed_s'] or 0) + 15.0)
                for r in rows if r.get('started_utc')]
        mine = [s for s in samples if any(a <= s['t'] <= b for a, b in wins)]
        if not mine:
            continue
        cols = sorted({k for s in mine for k in s if k.startswith(('cpu', 'npu', 'mem', 'robotd', 'brain', 'temp'))})
        out[name] = {'samples': len(mine),
                     **{c: round(sum(s[c] for s in mine if isinstance(s.get(c), float)) /
                                 max(1, sum(isinstance(s.get(c), float) for s in mine)), 2) for c in cols}}
    return out


# ── writing ─────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(cols)
    for r in rows:
        w.writerow(['' if r.get(c) is None else (json.dumps(r[c], ensure_ascii=False) if isinstance(r[c], (dict, list))
                    else r[c]) for c in cols])
    path.write_text(buf.getvalue())


def fmt_rate(r: dict | None) -> str:
    if not r or not r.get('n'):
        return '-'
    lo, hi = r['wilson95']
    return f'{r["k"]}/{r["n"]} = {100 * r["rate"]:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]'


def build(runs: Path, out: Path, onef: Path | None, dev: bool) -> dict:
    layouts = DEV_LAYOUTS if dev else FORMAL_LAYOUTS
    found = collect(runs, layouts, formal=not dev)
    groups, rows_by_group = {}, {}
    for name in GROUPS:
        if name in found:
            groups[name], rows_by_group[name] = summarise_group(name, found[name], layouts, dev)
    comparisons = {f'{a}_vs_{b}': paired(rows_by_group[a], rows_by_group[b]) for a, b in PAIRS
                   if a in rows_by_group and b in rows_by_group}
    notes_path = out / 'failure_notes.json'
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    fails = failures(rows_by_group, notes)
    seg = last_segment(runs, onef) if not dev else {'run': None, 'tried': []}
    load = load_by_group(runs, rows_by_group)
    frozen = sorted({s for n in ('board_int8', 'board_fp16', 'board_cpu', 'mac_cpu') if n in groups
                     for s in groups[n]['code_sha256']})
    head = groups.get(HEADLINE)
    pending = [f'{n}: {len(g["not_counted"])} episodes without a result' for n, g in groups.items() if g['not_counted']]
    pending += [f'{n}: no batch folder' for n in ('board_int8', 'mac_cpu') if n not in groups]
    if seg['run'] is None and not dev:
        pending.append('4D: no complete healthy 1F run after the freeze')
    if len(frozen) > 1:
        pending.append(f'4A-4C ran {len(frozen)} different code versions: {frozen}')
    facts = {'about': 'Stage 4 results (plan claude-docs/stage4_plan_20261028/). Written by scripts/stage4/tables.py '
                      'from the batch folders and the 4D run; Stage 5 takes its numbers only from here. Rates carry '
                      'a Wilson 95% interval; paired comparisons McNemar\'s exact p (two-sided).',
             'mode': 'dev (Stage 3 layouts, for checking the tool)' if dev else 'formal',
             'layouts': layouts,
             'rules': {'counted': 'an episode folder the batch runner counts as done (stage3/batch.py status); '
                                  'attempts without a result are listed, never scored',
                       'repeated_error': f'>= {REPEATED_ERROR_MIN} attempts ending in a brain error = a failure, end error',
                       'duplicates': 'the earlier result counts',
                       'headline': f'{HEADLINE} delivered >= {HEADLINE_MIN}/60',
                       'last_segment': 'earliest complete 1F run (30 + 30 starts) after the freeze with sim health ok'},
             'code_sha256_frozen_groups': frozen,
             'headline': None if head is None else {
                 'group': HEADLINE, 'delivered': head['delivered'],
                 'pass': head['delivered']['k'] >= HEADLINE_MIN if not dev and head['counted'] == 60 else None},
             'groups': groups, 'comparisons': comparisons, 'last_segment': seg,
             'detection': detection_table(groups), 'deployment': deployment_table(groups, rows_by_group, load),
             'failures': {'n': len(fails), 'by_group': {n: sum(f['group'] == n for f in fails) for n in groups}},
             'pending': pending}

    out.mkdir(parents=True, exist_ok=True)
    (out / 'stage4_facts.json').write_text(json.dumps(facts, indent=1, ensure_ascii=False) + '\n')
    all_rows = [r for n in GROUPS for r in rows_by_group.get(n, [])]
    write_csv(out / 'episodes.csv', all_rows,
              ['group', 'seed', 'target', 'outcome', 'delivered', 'delivered_strict', 'elapsed_s', 'closes', 'grasped',
               'wrong_grabs', 'falls', 'tof_stops', 'park_err_m', 'tray_contact', 'sim_stalled', 'sim_odom_ratio',
               'latency_ms', 'robotd', 'folder', 'trace'])
    write_csv(out / 'failures.csv', fails,
              ['group', 'seed', 'target', 'outcome', 'closes', 'falls', 'auto_cause', 'note', 'trace', 'folder'])
    t1 = [{'group': n, 'label': g['label'], 'episodes': g['counted'], 'delivered': fmt_rate(g['delivered']),
           'delivered_strict': fmt_rate(g['delivered_strict']), 'wrong_toy': g['wrong_toy'], 'falls': g['falls'],
           'tof_stops': g['tof_stops'], 'mean_s': g['mean_s'], 'ends': g['ends']} for n, g in groups.items()]
    write_csv(out / 't1_main.csv', t1, ['group', 'label', 'episodes', 'delivered', 'delivered_strict', 'wrong_toy',
                                        'falls', 'tof_stops', 'mean_s', 'ends'])
    t2 = [{'group': n, 'target': t, 'delivered': fmt_rate(v['delivered']), 'wrong_toy': v['wrong_toy'], 'ends': v['ends']}
          for n, g in groups.items() for t, v in g['per_target'].items()]
    write_csv(out / 't2_per_target.csv', t2, ['group', 'target', 'delivered', 'wrong_toy', 'ends'])
    write_csv(out / 't4_detection.csv', facts['detection'],
              ['detector', 'source', 'targets_map50', 'search_radius_m', 'wrong_grab_near', 'delivered', 'wrong_toy',
               'henry_delivered', 'recall_targets', 'precision_targets', 'wrong_class_within_1_6m'])
    write_csv(out / 't5_deployment.csv', facts['deployment'],
              ['backend', 'source', 'e2e_ms_p50_p99', 'inference_ms_p50_p99', 'preprocess_ms_p50_p99', 'system_cpu_pct',
               'delivered', 'e2e_ms_p50_median', 'e2e_ms_p99_worst', 'inference_ms_p50_median',
               'preprocess_ms_p50_median', 'robotd_missed', 'robotd_ticks', 'load'])
    t6 = []
    if seg['run']:
        for k in ('inching', 'blind'):
            p = seg[k]
            t6.append({'method': k, 'starts': p['starts'], 'closes': p['closes'], 'success_per_close': p['success_per_close'],
                       'first_close_held': fmt_rate(p['first_close_held']), 'held_within_3': fmt_rate(p['held_within_3']),
                       'park_rms_cm': p['park_err_cm']['rms'], 'park_lateral_mean_sd_cm': p['park_err_cm']['lateral_mean_sd'],
                       'park_forward_mean_sd_cm': p['park_err_cm']['forward_mean_sd'], 'mean_start_s': p['mean_start_s'],
                       'falls': p['falls']})
    write_csv(out / 't6_last_segment.csv', t6, ['method', 'starts', 'closes', 'success_per_close', 'first_close_held',
                                               'held_within_3', 'park_rms_cm', 'park_lateral_mean_sd_cm',
                                               'park_forward_mean_sd_cm', 'mean_start_s', 'falls'])
    t7 = [{'group': n, 'state_s_mean': g['timing']['state_s_mean'], 'latency_ms': g['timing']['latency_ms'],
           'mean_s': g['mean_s']} for n, g in groups.items()]
    write_csv(out / 't7_timing.csv', t7, ['group', 'mean_s', 'state_s_mean', 'latency_ms'])
    figures(out / 'figures', rows_by_group, fails, seg, runs)
    return facts


# ── figures ─────────────────────────────────────────────────────────────────

def figures(fig_dir: Path, rows_by_group: dict, fails: list[dict], seg: dict, runs: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except ImportError:
        return []
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    root = repo_root()
    # F1: failure traces, 12 per sheet, per group
    for name in rows_by_group:
        mine = [f for f in fails if f['group'] == name and f['trace']]
        for sheet in range(0, len(mine), 12):
            part = mine[sheet:sheet + 12]
            cols = 4
            rows = math.ceil(len(part) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4), squeeze=False)
            for ax in axes.flat:
                ax.axis('off')
            for ax, f in zip(axes.flat, part):
                ax.imshow(mpimg.imread(root / f['trace']))
                ax.set_title(f'{f["seed"]} {f["target"]}: {f["outcome"]}', fontsize=8)
            fig.suptitle(f'F1 failed episodes, {LABELS[name]}', fontsize=10)
            fig.tight_layout()
            path = fig_dir / f'f1_failures_{name}_{sheet // 12 + 1}.png'
            fig.savefig(path, dpi=110)
            plt.close(fig)
            made.append(path.name)
    # F2: robot's-eye frames of failed episodes (the last 4 frames each, Mac runs keep them)
    sheets = [(f, sorted((root / r['frames']).glob('*.jpg'))[-4:]) for f in fails
              for r in rows_by_group[f['group']] if r['seed'] == f['seed'] and r['target'] == f['target'] and r['frames']]
    sheets = [(f, imgs) for f, imgs in sheets if imgs][:8]
    if sheets:
        fig, axes = plt.subplots(len(sheets), 4, figsize=(9, 2.6 * len(sheets)), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for (f, imgs), ax_row in zip(sheets, axes):
            for ax, img in zip(ax_row, imgs):
                ax.imshow(mpimg.imread(img))
                ax.set_title(f'{f["group"]} {f["seed"]} {f["target"]}: {img.stem}', fontsize=6)
        fig.suptitle("F2 robot's-eye frames before the end of failed episodes", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / 'f2_failure_frames.png', dpi=110)
        plt.close(fig)
        made.append('f2_failure_frames.png')
    # F3: detection latency per episode (p50 dots, p99 crosses) per group
    have = [(n, rows) for n, rows in rows_by_group.items() if any('e2e' in r['latency_ms'] for r in rows)]
    if have:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for i, (n, rows) in enumerate(have):
            p50 = [r['latency_ms']['e2e']['p50'] for r in rows if 'e2e' in r['latency_ms']]
            p99 = [r['latency_ms']['e2e']['p99'] for r in rows if 'e2e' in r['latency_ms']]
            ax.scatter([i - 0.12] * len(p50), p50, s=10, label='per-episode P50' if i == 0 else None, color='#2f5d9a')
            ax.scatter([i + 0.12] * len(p99), p99, s=10, marker='x', label='per-episode P99' if i == 0 else None,
                       color='#b06000')
        ax.axhline(66.0, color='#888', lw=0.8, ls='--')
        ax.text(0.99, 67, 'one frame at 15 fps (66 ms)', fontsize=7, ha='right', color='#555',
                transform=ax.get_yaxis_transform())
        ax.set_xlim(-0.6, len(have) - 0.4)
        ax.set_xticks(range(len(have)), [LABELS[n].replace(', ', '\n', 1) for n, _ in have], fontsize=7)
        ax.set_ylabel('detection end-to-end, ms')
        ax.set_title('F3 detection latency in the closed loop', fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / 'f3_latency.png', dpi=120)
        plt.close(fig)
        made.append('f3_latency.png')
    # F4: park error per close, grasped vs missed
    panels = [(LABELS[n], [(p, res == 'grasped') for r in rows for p, res in zip(r['park_err_m'], r['close_results'])])
              for n, rows in rows_by_group.items() if n in ('board_int8', 'mac_cpu')]
    if seg.get('run'):
        s = json.loads((root / seg['run'] / 'summary.json').read_text())['results']
        for part, title in (('f_main', '1F visual inching'), ('f_blind', '1F blind walk')):
            panels.append((title, [(pk['park_err_m'], pk['grasped']) for r in s[part] for pk in r['picks']
                                   if pk.get('park_err_m') and pk.get('ran')]))
    panels = [p for p in panels if p[1]]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.3), squeeze=False)
        for ax, (title, pts) in zip(axes[0], panels):
            for ok, colour, lab in ((True, '#1f7a32', 'held'), (False, '#c0392b', 'missed')):
                xs = [100 * p[1] for p, g in pts if g == ok]
                ys = [100 * p[0] for p, g in pts if g == ok]
                ax.scatter(xs, ys, s=12, color=colour, label=f'{lab} ({len(xs)})')
            ax.add_patch(plt.Circle((0, 0), 4.0, fill=False, ls='--', color='#888'))
            ax.axhline(0, color='#ccc', lw=0.6)
            ax.axvline(0, color='#ccc', lw=0.6)
            ax.set_xlim(-8, 8)
            ax.set_ylim(-8, 8)
            ax.set_aspect('equal')
            ax.set_xlabel('lateral, cm (+ = parked right of the toy)', fontsize=7)
            ax.set_ylabel('forward, cm', fontsize=7)
            ax.set_title(title, fontsize=8)
            ax.legend(fontsize=6)
        fig.suptitle('F4 where the toy was at each mouth close (dashed: 4 cm grasp tolerance)', fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / 'f4_park_error.png', dpi=120)
        plt.close(fig)
        made.append('f4_park_error.png')
    return made


def print_summary(facts: dict, out: Path) -> None:
    p = print
    p(f'阶段 4 汇总（{facts["mode"]}）：{rel(out)}')
    for name, g in facts['groups'].items():
        line = (f'  {g["step"]} {name}：计入 {g["counted"]}/{g["expected"]}，送达 {fmt_rate(g["delivered"])}，'
                f'认错 {g["wrong_toy"]}，摔倒 {g["falls"]}，平均 {g["mean_s"]} s')
        if g['board']:
            b = g['board']
            line += (f'；robotd 漏拍 {b["robotd_missed"]}/{b["robotd_ticks"]}（允许 {b["robotd_missed_allowed"]}），'
                     f'P99 最差 {b["e2e_p99_ms_worst"]} ms')
        p(line)
        if g['attempts_not_counted']['total']:
            p(f'    没算数的尝试 {g["attempts_not_counted"]["total"]} 次：{g["attempts_not_counted"]["by_status"]}')
    for k, c in facts['comparisons'].items():
        p(f'  配对 {k}：{c["pairs"]} 对，只有前者送达 {c["a_only"]}，只有后者送达 {c["b_only"]}，McNemar p = {c["mcnemar_p"]}')
    seg = facts['last_segment']
    if seg.get('run'):
        p(f'  4D {seg["run"]}：点动 3 次内叼住 {fmt_rate(seg["inching"]["held_within_3"])}，'
          f'盲走 {fmt_rate(seg["blind"]["held_within_3"])}，McNemar p = {seg["paired_held_within_3"]["mcnemar_p"]}')
    if facts['headline']:
        verdict = {True: '过', False: '没过', None: '不判（开发模式，或还没有 60 个回合）'}[facts['headline']['pass']]
        p(f'  主结果 {facts["headline"]["group"]}：{fmt_rate(facts["headline"]["delivered"])}，'
          f'及格线 {HEADLINE_MIN}/60：{verdict}')
    if facts['pending']:
        p('  还缺：' + '；'.join(facts['pending']))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='stage4 tables', description=__doc__.split('\n')[0])
    ap.add_argument('--runs', default=None, help='runs folder (default: env.sh MD_RUNS)')
    ap.add_argument('--out', default=None, help='output folder (default: results/stage4, or results/stage4_dev with --dev)')
    ap.add_argument('--onef', default=None, help='the 4D 1F run folder (default: the rule in the docstring)')
    ap.add_argument('--dev', action='store_true', help='Stage 3 dev batches instead (checks the tool on known numbers)')
    a = ap.parse_args(argv)
    runs = Path(a.runs) if a.runs else runs_dir()
    out = Path(a.out) if a.out else repo_root() / 'results' / ('stage4_dev' if a.dev else 'stage4')
    facts = build(runs, out, Path(a.onef) if a.onef else None, a.dev)
    print_summary(facts, out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
