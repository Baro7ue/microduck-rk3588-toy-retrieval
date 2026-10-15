"""Stage 1A: the field works: the robot stands in it, the toys lie still, every layout loads.

    python -m microduck_toy.stage1.s1a_scene [--parts a_stand,a_layouts] [--quick]

Parts (plan section 3):
  a_stand    formal layout 0 with the bus as target; stand 10 s at H0, then 10 s at H2.
             Pass: no fall; every placed toy moves < 1 mm over each 10 s; robotd misses no more
             ticks than in Stage 0 (Mac 0/15045).
  a_layouts  every formal layout (configs/control/stage1_layouts.json) x 3 targets: load it, wait
             1.5 s. Pass: the robot stands, no toy moves > 1 mm between 0.5 s and 1.5 s, and the
             file equals what the seeds generate now.
The render parity check is a separate, robotd-free script: stage1/parity.py.
"""
from __future__ import annotations

import json
import sys
import time

from microduck_toy.paths import repo_root
from microduck_toy.scene import layout as layout_lib
from microduck_toy.scene import toys as toy_lib
from microduck_toy.stage1 import common, metrics

STAND_S = 10.0
STILL_M = 0.001
LAYOUTS_FILE = 'configs/control/stage1_layouts.json'


def load_layouts() -> dict:
    return json.loads((repo_root() / LAYOUTS_FILE).read_text())


def a_stand(s) -> None:
    doc = load_layouts()
    ep = layout_lib.episode(doc['layouts'][0], 'bus')
    s.reset(head='H0', robot=ep['robot'], toys=ep['toys'])
    for head in ('H0', 'H2'):
        s.robot.head(head)
        time.sleep(1.5)
        before = s.robot.snapshot()['loop']
        start = s.mark(f'a_stand_{head}')
        time.sleep(3.0 if s.args.quick else STAND_S)
        end = s.mark(f'a_stand_{head}_end')
        after = s.robot.snapshot()['loop']
        s.trial({'part': 'a_stand', 'head': head, 'start': start, 'end': end,
                 'toys': sorted(ep['toys']), 'loop_before': before, 'loop_after': after})
        print(f'a_stand {head} 完成', flush=True)


def a_layouts(s) -> None:
    doc = load_layouts()
    regenerated = layout_lib.formal_layouts(s.cfg['layout'], s.cfg['arena'])
    same = json.dumps(regenerated, sort_keys=True) == json.dumps(doc['layouts'], sort_keys=True)
    s.meta['layouts_match_seeds'] = same
    chosen = doc['layouts'][:3] if s.args.quick else doc['layouts']
    targets = [toy_lib.SHORT[n] for n in toy_lib.targets()]
    for lay in chosen:
        for target in (targets[:1] if s.args.quick else targets):
            ep = layout_lib.episode(lay, target)
            s.reset(head='H0', robot=ep['robot'], toys=ep['toys'], hold=1.0, settle=0.0)
            start = s.mark(f'a_layout_{lay["seed"]}_{target}')
            time.sleep(1.6)
            s.trial({'part': 'a_layouts', 'seed': lay['seed'], 'target': target, 'start': start,
                     'toys': sorted(ep['toys'])})
        print(f'布局 {lay["seed"]} 完成', flush=True)


PARTS = {'a_stand': a_stand, 'a_layouts': a_layouts}


def missed_ticks(record: dict) -> int | None:
    before, after = record.get('loop_before') or {}, record.get('loop_after') or {}
    if 'missed' in before and 'missed' in after:
        return int(after['missed']) - int(before['missed'])
    return None


def score(s, rows) -> dict:
    parts: dict[str, list] = {}
    for record in s.trials:
        out = {k: record.get(k) for k in ('head', 'seed', 'target')}
        if record['part'] == 'a_stand':
            span = metrics.window(rows, record['start'], record['end'])
            out['fell'] = metrics.fall_time(rows, record['start'], record['end']) is not None
            out['toy_moved_m'] = {t: metrics.displacement(span[0], span[-1], t)
                                  for t in record['toys']} if span else None
            out['missed_ticks'] = missed_ticks(record)
            out['loop_hz'] = (record.get('loop_after') or {}).get('hz')
            out['seconds'] = record['end'] - record['start']
        else:
            hold_end = record['start']           # reset returns when the hold ends
            early = metrics.at(rows, hold_end + 0.5)
            late = metrics.at(rows, hold_end + 1.5)
            out['standing'] = metrics.standing(late, s.refs)
            out['fell'] = metrics.fall_time(rows, hold_end, hold_end + 1.6) is not None
            out['toy_moved_m'] = {t: metrics.displacement(early, late, t) for t in record['toys']}
        parts.setdefault(record['part'], []).append(out)
    verdict = {}
    if 'a_stand' in parts:
        res = parts['a_stand']
        worst = max((max(r['toy_moved_m'].values()) for r in res if r['toy_moved_m']), default=None)
        missed = [r['missed_ticks'] for r in res if r['missed_ticks'] is not None]
        verdict['a_stand'] = {
            'falls': sum(r['fell'] for r in res), 'toy_moved_max_m': worst,
            'missed_ticks': sum(missed) if missed else None,
            'pass': None if s.args.quick else bool(
                not any(r['fell'] for r in res) and worst is not None and worst < STILL_M
                and missed and sum(missed) == 0),
        }
    if 'a_layouts' in parts:
        res = parts['a_layouts']
        worst = max((max(r['toy_moved_m'].values()) for r in res), default=None)
        verdict['a_layouts'] = {
            'loaded': len(res), 'standing': sum(r['standing'] for r in res),
            'falls': sum(r['fell'] for r in res), 'toy_moved_max_m': worst,
            'layouts_match_seeds': s.meta.get('layouts_match_seeds'),
            'pass': None if s.args.quick else bool(
                all(r['standing'] and not r['fell'] for r in res) and worst is not None
                and worst < STILL_M and s.meta.get('layouts_match_seeds')),
        }
    verdict['note'] = ('plan 1A: stand 10 s without falling or extra missed ticks; toys still '
                       '(< 1 mm); every formal layout loads standing with still toys; pass is null '
                       'for --quick')
    return {'verdict': verdict, 'parts': parts}


def write_layouts(argv) -> int:
    """--write-layouts: (re)generate configs/control/stage1_layouts.json from the seeds."""
    cfg = common.stage1_config()
    layouts = layout_lib.formal_layouts(cfg['layout'], cfg['arena'])
    bad = {lay['seed']: layout_lib.violations(lay, cfg['layout'], cfg['arena']) for lay in layouts}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise SystemExit(f'layouts break constraints: {bad}')
    doc = {'about': ('Stage 1 formal layouts (plan 1A): seeds formal_seed_base .. +formal_seeds-1 '
                     'of configs/control/stage1.json "layout", made by scene/layout.py. robot = '
                     'start [x, y, yaw]; slots = [x, y, yaw] of the target slot and the two '
                     'distractors. Each layout runs once per target: 20 x 3 = 60 episodes.'),
           'layouts': layouts}
    path = repo_root() / LAYOUTS_FILE
    rows = ',\n'.join('  ' + json.dumps(lay) for lay in layouts)
    path.write_text('{\n "about": ' + json.dumps(doc['about']) + ',\n "layouts": [\n' + rows
                    + '\n ]\n}\n')
    print(f'已写入 {path}（{len(layouts)} 个布局）')
    return 0


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if '--write-layouts' in args:
        return write_layouts(args)
    return common.main('a', __doc__, PARTS, score, argv)


if __name__ == '__main__':
    sys.exit(main())
