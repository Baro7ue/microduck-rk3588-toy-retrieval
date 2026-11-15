"""Stage 5 room demo (not a formal result): one room, 12 toys, the frozen brain and YOLO deliver
several targets one after another without a reset.

    python -m microduck_toy.stage5.demo [--seed N] [--targets bus,henry,ketchup]
        [--target mac|board] [--backend cpu|fp16|int8] [... the stage3 episode options]

Needs the arena body in room mode (scripts/sim/stage1/run_arena_body.zsh --camera --room, plus
--board for the board) and robotd s2. The room, the field, the tray and the search spots come from
configs/demo/room.json (deep-merged over the frozen stage1/stage3 configs, stage5/room.py); the
toys are laid out once from the demo seed. Each target is one brain run (on the Mac in this
process, on the board one ssh as in Stage 3F) started where the previous run's brain thought it
ended (its odometry/tray-look pose, no truth; if that brain died without a pose, the truth, marked
start_from_truth in trials.jsonl and demo.json). The drop side is the brain's own Stage 3 choice
(v2; room.json demo.drop_sides, if put back, gives each target its own side, brain --side). Between
runs nothing is reset: the toys already in the tray stay there. Judged on the simulator's truth like a Stage 3
episode (in the round tray: inside its rim), plus the toys a run knocked out of the tray and
what is still in it at the end; output in runs/<UTC>_s5_room_demo[_board]/ (demo.json + the usual
truth.csv, events, episodes/<n>_<target>/). Diagnostics beside the unchanged outcome (v4, the bowl):
per target delivered_by_drop (in the bowl and the brain ended 'dropped': the DROP state let it go,
not a fall or an error), opened_at_m (the trunk's true distance from the centre when the beak
opened), tray_touch_s (truth rows with the robot touching the tray, x 20 ms), tray_look_err_m (the
last tray look's centre against the truth) and left_clear / leave_result; totals delivered_by_drop,
touched_tray (runs with a touch) and left_clear.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

from microduck_toy.paths import runs_dir
from microduck_toy.scene import toys as toy_lib
from microduck_toy.stage1 import metrics
from microduck_toy.stage3 import brain as brain_lib, episode
from microduck_toy.stage5 import room

ROOM = 'configs/demo/room.json'


class Demo(episode.Runner):
    def __init__(self, args: argparse.Namespace):
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        board = args.target == 'board'
        if args.out is None:
            args.out = str(runs_dir() / f'{stamp}_s5_room_demo{"_board" if board else ""}')
        super().__init__('demo', args)
        self.doc = room.load()
        self.configs = brain_lib.load_configs(override=ROOM)
        self.cfg, self.s3 = self.configs['stage1'], self.configs['stage3']
        self.toys = room.toy_names(self.doc)
        missing = [t for t in self.toys if t not in (self.meta.get('toys') or {})]
        if missing:
            raise RuntimeError(f'the arena body has no {missing}: start run_arena_body.zsh with --room')
        want = {k: v for k, v in self.cfg['arena']['tray'].items() if not k.startswith('note')}
        if self.probe.call('info').get('room_tray') != want:
            raise RuntimeError('the arena body was built from another room.json (its tray differs): '
                               'restart run_arena_body.zsh --camera --room')
        self.meta.update(test='s5_room_demo', stage1_config=self.cfg, stage3_config=self.s3,
                         room=self.doc, config_override=ROOM)

    def run_target(self, n: int, target: str, pose, layout: dict | None, side: str | None = None) -> dict:
        label = f'{n}_{target}'
        truth = self.probe.truth()
        rec = {'part': 'demo', 'label': label, 'seed': self.args.seed, 'target': target,
               'layout': layout, 'until': 'delivered', 'grasps_from': self.grasp_count(),
               'side': side, 'start_told': [round(float(v), 5) for v in pose],
               'start_truth': [round(truth['x'], 5), round(truth['y'], 5), round(truth['yaw'], 5)],
               'start': self.mark(label), 'offset': 0.0}
        where = f'，放到托盘 {side} 边' if side else ''
        print(f'== {label}：从 ({pose[0]:+.2f}, {pose[1]:+.2f}) 出发找 {target}{where}', flush=True)
        if self.board:
            self.args.side = side                  # episode.board_command passes it on
            rec['brain'] = self.remote(label, target, pose, 'delivered', rec)
        else:
            b = self.brain(label, target, pose, until='delivered', side=side)
            n0 = len(self.eyes.timings)
            rec['brain'] = b.run()
            rec['brain']['latency'] = self.eyes.latency(n0)
        return self.finish(rec, label)


def row_near(rows: list[dict], mono: float, within_s: float = 0.1) -> dict | None:
    """The truth row nearest the Mac monotonic time mono (None without one within within_s)."""
    best = min(rows, key=lambda r: abs(r['mono'] - mono), default=None)
    return best if best is not None and abs(best['mono'] - mono) <= within_s else None


def drop_checks(rec: dict, rows: list[dict], centre) -> dict:
    """v4 diagnostics from the truth (not the pass bar): where the trunk was when the beak opened,
    how long the robot touched the tray during the run, how far off the last tray look was. Brain
    times are its own monotonic clock: + rec['offset'] is the Mac's (0 on the Mac)."""
    off = float(rec.get('offset') or 0.0)
    brain = rec['brain']
    out = {'opened_at_m': None, 'tray_touch_s': None, 'tray_look_err_m': None}
    if not rows:
        return out
    opened = (brain.get('drop') or {}).get('open')
    r = row_near(rows, float(opened) + off) if opened is not None else None
    if r is not None:
        out['opened_at_m'] = round(math.hypot(r['x'] - centre[0], r['y'] - centre[1]), 4)
    span = metrics.window(rows, rec['start'], rec.get('judge_mark', rec.get('end_mark', math.inf)))
    out['tray_touch_s'] = round(sum(1 for x in span if x.get('tray_contact')) * 0.02, 2)
    look = brain.get('tray') or {}
    r = row_near(rows, float(look['mono']) + off) if look.get('mono') is not None else None
    if r is not None and look.get('centre'):
        c, s = math.cos(r['yaw']), math.sin(r['yaw'])
        dx, dy = centre[0] - r['x'], centre[1] - r['y']
        out['tray_look_err_m'] = round(math.dist((c * dx + s * dy, -s * dx + c * dy), look['centre']), 4)
        out['tray_look_source'] = look.get('source')
    return out


def judge(rec: dict, rows: list[dict], toys: list[str], in_before: set, s3: dict, centre,
          tray: dict) -> dict:
    """What this run did to the tray (truth): the toys in it now and not before (new), and the
    ones in it before and not now (knocked out, e.g. hit by the new drop or kicked)."""
    row = rec['judge_row']
    in_now = {t for t in toys if room.in_tray(row, t, centre, tray, s3)}
    new = sorted(in_now - in_before)
    out = sorted(in_before - in_now)
    kinds = [toy_lib.kind_of(t) for t in new]
    held = sorted({g['candidate'] for g in rec.get('grasps', [])
                   if g.get('result') == 'grasped' and g.get('candidate')})
    falls = episode.falls_in(rows, rec['start'], rec.get('judge_mark', rec['end_mark']))
    ok = rec['target'] in kinds and all(k == rec['target'] for k in kinds)
    told, end_pose = rec['start_told'], rec['brain'].get('pose')
    end = rec['brain'].get('end')
    # Diagnostics beside the outcome, which is unchanged (in the tray counts, however it got there:
    # Mac 20261112T035812Z bus fell into it out of the beak, brain grasp_fail; v3 retakes take1
    # ketchup grasp_fail and take3 henry error were in the tray too): delivered_by_drop says the DROP
    # state let it go; drop_checks where and how.
    drop = rec['brain'].get('drop') or {}
    leave = drop.get('leave') or {}
    if ok:
        outcome = 'delivered'
    elif new:
        outcome = 'wrong_toy'
    elif end == 'dropped':
        outcome = 'drop_missed'               # Stage 3's name: dropped, but not into the tray
    else:
        outcome = end
    return {'outcome': outcome, 'side': rec.get('side'),
            'start_from_truth': bool(rec.get('start_from_truth')),
            'new_in_tray': new, 'knocked_out': out, 'in_tray': sorted(in_now), 'held': held,
            'falls': len(falls), 'brain_end': rec['brain'].get('end'),
            'delivered_by_drop': ok and end == 'dropped',
            'left_clear': leave.get('clear'), 'leave_result': leave.get('result'),
            **drop_checks(rec, rows, centre),
            'elapsed_s': rec['brain'].get('elapsed_s'), 'closes': rec['brain'].get('closes'),
            'start_pose_error_m': round(math.dist(told[:2], rec['start_truth'][:2]), 3),
            'end_pose': end_pose}


def parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: otherwise the runner's --target (mac|board) is taken as an abbreviation of
    # --targets (Ethan's first board run 2026-10-31: "['board'] are not target toys").
    p = argparse.ArgumentParser(prog='stage5.demo', description=__doc__.split('\n')[0], allow_abbrev=False)
    p.add_argument('--seed', type=int, default=None, help='toy layout (default: room.json demo.seed)')
    p.add_argument('--targets', default=None, help='e.g. bus,henry,ketchup (default: room.json demo.targets)')
    return p


def parse(argv=None) -> tuple[argparse.Namespace, argparse.Namespace]:
    """(the demo's own options, the stage3 episode options it passes through)."""
    own, rest = parser().parse_known_args(argv)
    return own, episode.parser().parse_args(['3e', *rest])


def main(argv=None) -> int:
    own, args = parse(argv)
    doc = room.load()
    args.seed = int(own.seed if own.seed is not None else doc['demo']['seed'])
    targets = (own.targets.split(',') if own.targets else list(doc['demo']['targets']))
    bad = [t for t in targets if t not in brain_lib.TARGETS]
    if bad:
        raise SystemExit(f'{bad} are not target toys {brain_lib.TARGETS}')
    if args.backend is None:
        args.backend = 'int8' if args.target == 'board' else 'cpu'
    if args.target == 'board' and (args.board_bundle is None or args.board_out is None):
        raise SystemExit('board runs come from scripts/stage5/run_room_demo.zsh --board')
    args.config_override = ROOM                         # episode.board_command passes it on
    sides = list(doc['demo'].get('drop_sides') or [])
    sides = [sides[k % len(sides)] if sides else None for k in range(len(targets))]
    lay = room.layout(doc, args.seed)
    demo = Demo(args)
    demo.meta.update(seed=args.seed, demo_targets=targets, drop_sides=sides, layout=lay)
    centre = demo.meta['tray_centre']
    tray = demo.cfg['arena']['tray']
    recs, status = [], 0
    t0 = time.monotonic()
    try:
        demo.start()
        demo.reset(head='H0', robot=lay['robot'], toys=lay['toys'])
        if not demo.board:
            demo.wait_settled(3.0)
        pose = list(lay['robot'])
        from_truth = False
        for n, target in enumerate(targets, 1):
            rec = demo.run_target(n, target, pose, lay if n == 1 else None, sides[n - 1])
            rec['start_from_truth'] = from_truth
            end = rec['brain'].get('pose')
            from_truth = end is None
            if from_truth:             # the brain died before it could say: carry on from the truth
                row = rec['judge_row']
                end = [row['x'], row['y'], row['yaw']]
            # Set before the trial line is written (v2: it used to be set after, so trials.jsonl never had it).
            rec['next_start_from_truth'] = from_truth
            recs.append(rec)
            demo.trial({k: v for k, v in rec.items() if k != 'layout'})
            pose = end
    except KeyboardInterrupt:
        demo.meta['interrupted'] = True
        status = 130
        print('已中断；保存已完成的部分。', flush=True)
    except Exception as error:
        demo.meta['error'] = f'{type(error).__name__}: {error}'
        status = 1
        print(f'出错：{demo.meta["error"]}', flush=True)
    demo.meta['wall_s'] = round(time.monotonic() - t0, 1)
    rows = demo.close()
    before: set = set()
    results = []
    for rec in recs:
        j = judge(rec, rows, demo.toys, before, demo.s3, centre, tray)
        before = set(j['in_tray'])
        results.append({'n': rec['label'], 'target': rec['target'], **j})
    delivered = sum(r['outcome'] == 'delivered' for r in results)
    # Still in the tray at the end: the last judge row (each run's own toys, after every later drop).
    at_end = sorted(before)
    kept = sum(1 for r in results if r['outcome'] == 'delivered'
               and any(t in at_end for t in r['new_in_tray']))
    summary = {'demo': {'seed': args.seed, 'targets': targets, 'drop_sides': sides,
                        'delivered': delivered, 'kept_at_end': kept, 'in_tray_at_end': at_end,
                        'knocked_out': sum(len(r['knocked_out']) for r in results),
                        'starts_from_truth': sum(r['start_from_truth'] for r in results),
                        # v4 diagnostics (not the pass bar): delivered by the DROP state, runs that
                        # touched the tray, backs clear of it
                        'delivered_by_drop': sum(bool(r.get('delivered_by_drop')) for r in results),
                        'touched_tray': sum(bool(r.get('tray_touch_s')) for r in results),
                        'left_clear': sum(r['left_clear'] is True for r in results),
                        'results': results, 'wall_s': demo.meta['wall_s'],
                        'brain_s': round(sum(r['elapsed_s'] or 0 for r in results), 1),
                        'falls': sum(r['falls'] for r in results)},
               'truth': metrics.continuity(rows)}
    demo.write_summary(summary)
    (demo.dir / 'demo.json').write_text(json.dumps(summary['demo'], indent=1))
    print(f'\n== 房间演示（布局 {args.seed}）：送到 {delivered}/{len(targets)}，'
          f'最后还在托盘里 {kept}/{len(targets)}', flush=True)
    for r in results:
        got = ', '.join(r['new_in_tray']) or '无'
        out = f'  撞出：{", ".join(r["knocked_out"])}' if r['knocked_out'] else ''
        print(f'  {r["n"]:10s} {r["side"] or "-":3s} {r["outcome"]:12s} 新进托盘：{got:18s} '
              f'用时 {r["elapsed_s"] or 0:6.1f} s  摔倒 {r["falls"]}  '
              f'起点估计误差 {r["start_pose_error_m"]:.2f} m{out}', flush=True)
        print(f'             DROP 放下 {"是" if r.get("delivered_by_drop") else "否"}  '
              f'开嘴时距盘心 {_cm(r.get("opened_at_m"))}  碰盘 {r.get("tray_touch_s")} s  '
              f'看盘误差 {_cm(r.get("tray_look_err_m"))}  离开 {r.get("leave_result")}', flush=True)
    print(f'  最后托盘里：{", ".join(at_end) or "无"}', flush=True)
    print(f'  结果：{demo.dir}/demo.json', flush=True)
    return status


def _cm(v) -> str:
    return '-' if v is None else f'{v * 100:.1f} cm'


if __name__ == '__main__':
    sys.exit(main())
