"""Stage 3G batch runs: the JSON side of scripts/stage3/run_batch.zsh.

The zsh runner owns the loop, the processes and Ctrl+C; each episode is one
`episode 3e --episodes SEED:TARGET --out <batch>/episodes/SEED_TARGET` process. This module answers it:

    python -m microduck_toy.stage3.batch plan --layouts FILE [--seeds A-B] [--targets bus,henry,ketchup] [--formal]
        one SEED:TARGET per line, seed by seed (the order 3E runs them)
    python -m microduck_toy.stage3.batch begin BATCH_DIR --target T --layouts FILE --episodes S:T,... --backend B
                                               --code-sha256 SHA [--formal] [--extra=...] [--commit C --dirty yes|no]
        write BATCH_DIR/batch.json, or check that a resumed batch runs the same code, layouts and settings
    python -m microduck_toy.stage3.batch status EPISODE_DIR
        "done <outcome>" or "interrupted|infra|incomplete <why>" (only done episodes are skipped on resume)
    python -m microduck_toy.stage3.batch table BATCH_DIR
        every episode's line, the totals and the failed attempts; writes BATCH_DIR/summary.json
    python -m microduck_toy.stage3.batch formal-check [RUNS_DIR]
        the freeze check (plan 10): no Stage 3 run folder holds an episode on a formal seed (1000-1019)

An attempt that did not finish (Ctrl+C, a crash, a brain error, a sim that did not keep real time, a
missing log) is not a result: the runner renames its folder SEED_TARGET.<status><n> as evidence and
runs the episode again next time. The table counts those attempts, so a report can say how many there were.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

from microduck_toy.paths import repo_root, runs_dir
from microduck_toy.stage3 import episode

SETTINGS = ('target', 'layouts', 'layouts_sha256', 'episodes', 'backend', 'formal', 'code_sha256', 'extra')
FAILED = re.compile(r'^(?P<name>\d+_[a-z]+)\.(?P<status>interrupted|infra|incomplete)(?P<n>\d+)$')
STAGE3_RUN = re.compile(r'^\d{8}T\d{6}Z_s3')


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


# ── plan ────────────────────────────────────────────────────────────────────

def parse_seeds(text: str) -> tuple[int, int]:
    m = re.fullmatch(r'(\d+)(?:-(\d+))?', text.strip())
    if not m:
        raise SystemExit(f'--seeds {text!r}: write A-B or A, e.g. 3000-3001')
    a, b = int(m.group(1)), int(m.group(2) or m.group(1))
    if b < a:
        raise SystemExit(f'--seeds {text!r}: the last seed is before the first')
    return a, b


def plan(layouts: str, seeds: str | None, targets: str, formal: bool) -> list[str]:
    have = sorted(l['seed'] for l in episode.load_layouts(layouts))
    if seeds:
        a, b = parse_seeds(seeds)
        missing = [s for s in range(a, b + 1) if s not in have]
        if missing:
            raise SystemExit(f'seeds {missing} are not in {layouts} (it has {have[0]}-{have[-1]})')
        chosen = list(range(a, b + 1))
    else:
        chosen = have
    names = [t.strip() for t in targets.split(',') if t.strip()]
    bad = [t for t in names if t not in episode.TARGETS]
    if bad or not names or len(set(names)) != len(names):
        raise SystemExit(f'--targets {targets!r}: a list of {episode.TARGETS}, each once')
    episode.check_not_formal(chosen, formal)
    return [f'{s}:{t}' for s in chosen for t in names]


# ── begin ───────────────────────────────────────────────────────────────────

def file_sha256(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    return hashlib.sha256(p.read_bytes()).hexdigest()


def begin(batch_dir: Path, settings: dict, commit: str | None = None, dirty: bool | None = None) -> dict:
    """batch.json: the settings a batch started with; a resume must match them (the code too, by the
    sha256 of the bundle's SHA256SUMS), so one batch never mixes two versions of the brain."""
    settings = {**settings, 'layouts_sha256': file_sha256(settings['layouts'])}
    path = batch_dir / 'batch.json'
    call = {'utc': utc(), 'commit': commit, 'dirty': dirty}
    if path.exists():
        doc = json.loads(path.read_text())
        old = doc['settings']
        diff = [k for k in SETTINGS if old.get(k) != settings.get(k)]
        if diff:
            lines = [f'  {k}: 原来 {old.get(k)!r}，现在 {settings.get(k)!r}' for k in diff]
            raise SystemExit(f'{batch_dir} 是用不同的设置开始的，不能接着跑：\n' + '\n'.join(lines)
                             + '\n换一个 --out，或者把代码和设置改回原来的样子。')
        doc['invocations'].append(call)
    else:
        doc = {'about': 'Stage 3G batch (scripts/stage3/run_batch.zsh): one folder per episode under '
                        'episodes/; the same command again resumes it.',
               'settings': settings, 'created_utc': call['utc'], 'invocations': [call]}
    batch_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + '\n')
    return doc


# ── status ──────────────────────────────────────────────────────────────────

def status(ep_dir: Path) -> tuple[str, str]:
    """Is this episode folder a result? Only 'done' is; everything else is run again."""
    path = ep_dir / 'summary.json'
    if not path.exists():
        return 'incomplete', '没有 summary.json（回合进程没来得及保存就结束了）'
    try:
        s = json.loads(path.read_text())
    except ValueError:
        return 'incomplete', 'summary.json 读不了'
    meta = s.get('meta') or {}
    if meta.get('interrupted'):
        return 'interrupted', '按了 Ctrl-C'
    if meta.get('error'):
        return 'infra', f'运行出错：{meta["error"]}'
    if s.get('score_error'):
        return 'infra', f'计分出错：{s["score_error"]}'
    results = [r for r in s.get('results') or [] if r.get('part') == 'e']
    if len(results) != 1:
        return 'infra', f'{len(results)} 个回合结果（应该 1 个）'
    r = results[0]
    if r.get('brain_end') == 'error':
        trial = next(iter(s.get('trials') or []), {})
        return 'infra', f'大脑出错：{(trial.get("brain") or {}).get("error", "?")}'
    health = s.get('sim_health') or {}
    if health.get('ok') is False:
        ratio = (health.get('odom_ratio') or {}).get('ratio')
        return 'infra', f'仿真没按真实时间跑（超时拍 {health.get("stalled")}，里程计/真值 {ratio}）'
    if not (ep_dir / 'episodes' / r['label'] / 'brain.jsonl').exists():
        return 'infra', '缺大脑日志 brain.jsonl'
    if meta.get('target') == 'board':
        if r.get('latency_e2e_p99_ms') is None:
            return 'infra', '缺板端检测时延'
        if not r.get('robotd'):
            return 'infra', '缺板端 robotd 拍子日志 state.jsonl'
    return 'done', r['outcome']


# ── table ───────────────────────────────────────────────────────────────────

def width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)


def pad(text, n: int) -> str:
    text = '' if text is None else str(text)
    return text + ' ' * max(0, n - width(text))


def failed_attempts(batch_dir: Path) -> list[dict]:
    out = []
    root = batch_dir / 'episodes'
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        m = FAILED.match(d.name)
        if m and d.is_dir():
            st, why = status(d)
            out.append({'folder': d.name, 'episode': m['name'], 'status': m['status'], 'why': why})
    return out


def board_totals(rs: list[dict]) -> dict:
    """The 3F numbers over the batch (plan 9): robotd's missed ticks against the Stage 0 rate, P99."""
    v = episode.verdict_3f(rs, {})
    keep = ('robotd_ticks', 'robotd_missed', 'robotd_missed_allowed', 'robotd_hz_min',
            'e2e_p99_ms_worst_episode', 'latency_logs')
    return {k: v[k] for k in keep}


def table(batch_dir: Path, out=sys.stdout) -> dict:
    doc = json.loads((batch_dir / 'batch.json').read_text())
    planned = doc['settings']['episodes']
    board = doc['settings']['target'] == 'board'
    results, not_done = [], []
    for item in planned:
        name = item.replace(':', '_')
        d = batch_dir / 'episodes' / name
        st, why = status(d) if d.exists() else ('missing', '还没跑')
        if st != 'done':
            not_done.append({'episode': name, 'status': st, 'why': why})
            continue
        r = next(r for r in json.loads((d / 'summary.json').read_text())['results'] if r.get('part') == 'e')
        results.append(r)
    failed = failed_attempts(batch_dir)
    totals = episode.verdict_3e(results) if results else {}
    per_target = {}
    for t in sorted({item.split(':')[1] for item in planned}, key=episode.TARGETS.index):
        mine = [r for r in results if r['target'] == t]
        per_target[t] = {'done': len(mine), 'delivered': sum(r['outcome'] == 'delivered' for r in mine)}
    summary = {'batch': str(batch_dir), 'settings': doc['settings'], 'invocations': doc['invocations'],
               'planned': len(planned), 'done': len(results), 'not_done': not_done,
               'failed_attempts': failed, 'per_target': per_target, 'totals': totals,
               'board': board_totals(results) if board and results else None,
               'episodes': [{k: r.get(k) for k in ('seed', 'target', 'outcome', 'delivered_strict', 'elapsed_s',
                                                    'grasped', 'latency_e2e_p99_ms')}
                            | {'closes': len(r.get('closes') or []), 'falls': len(r.get('falls') or []),
                               'robotd_missed': (r.get('robotd') or {}).get('missed')} for r in results],
               'written_utc': utc()}
    (batch_dir / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False) + '\n')

    p = lambda *a: print(*a, file=out)
    p(f'批量结果：{batch_dir}')
    head = [('回合', 14), ('结果', 13), ('闭嘴', 6), ('摔倒', 6), ('用时 s', 8)]
    if board:
        head += [('检测 P99 ms', 12), ('robotd 漏拍', 10)]
    p(''.join(pad(h, w) for h, w in head))
    for r in results:
        el = r.get('elapsed_s')
        cells = [f'{r["seed"]} {r["target"]}', r['outcome'], len(r.get('closes') or []),
                 len(r.get('falls') or []), None if el is None else f'{el:.1f}']
        if board:
            cells += [r.get('latency_e2e_p99_ms'), (r.get('robotd') or {}).get('missed')]
        p(''.join(pad(c, w) for c, (_, w) in zip(cells, head)))
    if results:
        by = '，'.join(f'{t} {v["delivered"]}/{v["done"]}' for t, v in per_target.items())
        p(f'完成 {len(results)}/{len(planned)}；送达 {totals["delivered"]}（{by}），严格规则 {totals["delivered_strict"]}；'
          f'认错 {totals["wrong_toy"]}；平均 {totals["mean_s"]} s；摔倒 {totals["falls"]}；'
          f'ToF 急停 {totals["tof_stops"]}（误停 {totals["false_tof_stops"]}）')
        p('结束原因：' + '，'.join(f'{k} {v}' for k, v in sorted(totals['ends'].items())))
        if summary['board']:
            b = summary['board']
            p(f'板端：robotd 漏拍 {b["robotd_missed"]}/{b["robotd_ticks"]} 拍（按阶段 0 的比率允许 '
              f'{b["robotd_missed_allowed"]}），检测 P99 最差 {b["e2e_p99_ms_worst_episode"]} ms')
    else:
        p(f'完成 0/{len(planned)}')
    if failed:
        p(f'没算数的尝试 {len(failed)} 次（文件夹留着作证据）：')
        for f in failed:
            p(f'  {f["folder"]}：{f["why"]}')
    if not_done:
        p(f'还没有结果的回合 {len(not_done)} 个：' + '，'.join(n['episode'] for n in not_done))
    p(f'汇总：{batch_dir / "summary.json"}')
    return summary


# ── the freeze check ────────────────────────────────────────────────────────

def seeds_in(folder: Path) -> list[tuple[str, int]]:
    """Every (file, seed) a Stage 3 run folder's trials or batch settings name."""
    out = []
    for f in folder.rglob('trials.jsonl'):
        for line in f.read_text(errors='replace').splitlines():
            try:
                seed = json.loads(line).get('seed')
            except ValueError:
                continue
            if seed is not None:
                out.append((str(f), int(seed)))
    for f in folder.rglob('summary.json'):
        try:
            doc = json.loads(f.read_text())
        except ValueError:
            continue
        for rec in doc.get('trials') or []:
            if isinstance(rec, dict) and rec.get('seed') is not None:
                out.append((str(f), int(rec['seed'])))
    for f in folder.rglob('batch.json'):
        for item in json.loads(f.read_text())['settings'].get('episodes', []):
            out.append((str(f), int(item.split(':')[0])))
    return out


def formal_check(runs: Path) -> dict:
    formal = set(episode.formal_seeds())
    folders = sorted(d for d in runs.iterdir() if d.is_dir()
                     and (STAGE3_RUN.match(d.name) or d.name.startswith('batch_')))
    hits, seen = [], 0
    for d in folders:
        found = seeds_in(d)
        seen += len(found)
        hits += [{'file': f, 'seed': s} for f, s in found if s in formal]
    return {'runs_dir': str(runs), 'stage3_folders': len(folders), 'seeds_named': seen,
            'formal_seeds': f'{min(formal)}-{max(formal)}', 'formal_hits': hits, 'ok': not hits}


# ── command line ────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='stage3.batch', description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--layouts', default=str(episode.layouts_path().relative_to(repo_root())))
    p.add_argument('--seeds', default=None)
    p.add_argument('--targets', default=','.join(episode.TARGETS))
    p.add_argument('--formal', action='store_true')
    p = sub.add_parser('begin')
    p.add_argument('batch_dir')
    p.add_argument('--target', required=True, choices=('mac', 'board'))
    p.add_argument('--layouts', required=True)
    p.add_argument('--episodes', required=True, help='SEED:TARGET,... as plan printed them')
    p.add_argument('--backend', required=True)
    p.add_argument('--code-sha256', required=True, help="sha256 of the bundle's SHA256SUMS")
    p.add_argument('--formal', action='store_true')
    p.add_argument('--extra', default='', help='the extra episode options, as one string')
    p.add_argument('--commit', default=None)
    p.add_argument('--dirty', default=None, choices=('yes', 'no'))
    p = sub.add_parser('status')
    p.add_argument('episode_dir')
    p = sub.add_parser('table')
    p.add_argument('batch_dir')
    p = sub.add_parser('formal-check')
    p.add_argument('runs', nargs='?', default=None)
    a = ap.parse_args(argv)
    if a.cmd == 'plan':
        print('\n'.join(plan(a.layouts, a.seeds, a.targets, a.formal)))
    elif a.cmd == 'begin':
        settings = {'target': a.target, 'layouts': a.layouts, 'episodes': a.episodes.split(','),
                    'backend': a.backend, 'formal': a.formal, 'code_sha256': a.code_sha256, 'extra': a.extra}
        begin(Path(a.batch_dir), settings, a.commit, None if a.dirty is None else a.dirty == 'yes')
    elif a.cmd == 'status':
        print(' '.join(status(Path(a.episode_dir))))
    elif a.cmd == 'table':
        return 0 if not table(Path(a.batch_dir))['not_done'] else 3
    else:
        r = formal_check(Path(a.runs) if a.runs else runs_dir())
        if r['ok']:
            print(f'正式种子 {r["formal_seeds"]} 没有在阶段 3 用过：查了 {r["stage3_folders"]} 个阶段 3 运行文件夹，'
                  f'{r["seeds_named"]} 处种子记录。')
        else:
            print(f'发现正式种子 {len(r["formal_hits"])} 处：')
            for h in r['formal_hits'][:20]:
                print(f'  {h["seed"]}  {h["file"]}')
        return 0 if r['ok'] else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
