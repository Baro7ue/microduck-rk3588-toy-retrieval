"""Stage 4 result tables (scripts/stage4/tables.py) on made-up batch folders: the statistics, which
episodes count (plan section 2), merging folders, the paired comparisons, the 4D run choice, the board
load samples, and that two runs over the same folders write the same files."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from microduck_toy.stage3 import batch, episode

REPO = Path(__file__).resolve().parents[1]


def load_tables():
    spec = importlib.util.spec_from_file_location('stage4_tables', REPO / 'scripts' / 'stage4' / 'tables.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tables = load_tables()
FORMAL = tables.FORMAL_LAYOUTS
SEEDS = list(range(1000, 1020))


def write_episode(ep_dir: Path, seed: int, target: str, outcome='delivered', board=False, brain_end=None,
                  started='20261028T050000Z', park=((0.002, 0.01),), p99=45.0, health_ok=True, frames=False):
    brain_end = brain_end or ('dropped' if outcome in ('delivered', 'drop_missed') else outcome)
    label = f'e_{seed}_{target}'
    closes = [{'result': 'grasped' if i == len(park) - 1 and outcome != 'grasp_fail' else 'missed', 'toy': target}
              for i in range(len(park))]
    result = {'part': 'e', 'label': label, 'seed': seed, 'target': target, 'outcome': outcome,
              'brain_end': brain_end, 'delivered_strict': outcome == 'delivered', 'elapsed_s': 40.0,
              'closes': closes, 'grasped': sum(c['result'] == 'grasped' for c in closes), 'wrong_grabs': 0,
              'falls': [], 'tof_stops': [], 'states_s': {'SPIN': 5.0, 'LOOK': 3.0}, 'disturbed': [],
              'park_err_m': [list(p) for p in park], 'held_toys': [target] if outcome != 'grasp_fail' else [],
              'others_in_tray': [], 'target_from_tray_m': [0.2, 0.0, 0.01] if outcome == 'drop_missed' else [0, 0, 0],
              'latency_e2e_p99_ms': p99 if board else None}
    if board:
        result['robotd'] = {'ticks_expected': 2000, 'missed': 0, 'hz_min': 49.9}
    lat = {'frames': 300, 'e2e': {'p50': p99 - 10, 'p99': p99}, 'inference': {'p50': 7.0, 'p99': 14.0}}
    doc = {'meta': {'target': 'board' if board else 'mac', 'started_utc': started},
           'sim_health': {'ok': health_ok, 'stalled': 0.001 if health_ok else 0.2, 'odom_ratio': {'ratio': 0.99}},
           'vision_loop': {'frames': {'spin': 10, 'look': 5}, 'recall_targets': {'k': 9, 'n': 10, 'rate': 0.9},
                           'precision_targets': {'k': 10, 'n': 10, 'rate': 1.0}, 'wrong_class_within_1_6m': 0},
           'results': [result],
           'trials': [{'part': 'e', 'seed': seed, 'target': target,
                       'brain': {'end': brain_end, 'error': 'boom', 'latency': lat}}]}
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / 'summary.json').write_text(json.dumps(doc))
    (ep_dir / 'episodes' / label).mkdir(parents=True, exist_ok=True)
    (ep_dir / 'episodes' / label / 'brain.jsonl').write_text('{}\n')
    if frames:
        (ep_dir / 'episodes' / label / 'frames').mkdir()
    return ep_dir


def write_batch(runs: Path, name: str, target: str, backend: str, episodes: list[str], layouts=FORMAL,
                formal=True, sha='8295a307' + '0' * 56, extra='') -> Path:
    d = runs / name
    (d / 'episodes').mkdir(parents=True)
    settings = {'target': target, 'layouts': layouts, 'layouts_sha256': batch.file_sha256(layouts),
                'episodes': episodes, 'backend': backend, 'formal': formal, 'code_sha256': sha, 'extra': extra}
    (d / 'batch.json').write_text(json.dumps({'settings': settings, 'created_utc': '20261028T050000Z',
                                              'invocations': [{'utc': '20261028T050000Z', 'commit': 'cc0931a',
                                                               'dirty': False}]}))
    return d


def all_episodes(seeds=SEEDS) -> list[str]:
    return [f'{s}:{t}' for s in seeds for t in episode.TARGETS]


class StatisticsTests(unittest.TestCase):
    def test_wilson_matches_the_plan(self):
        self.assertEqual(tables.wilson(50, 60), [0.7197, 0.9069])
        self.assertEqual(tables.wilson(42, 60), [0.5749, 0.801])
        self.assertEqual(tables.wilson(13, 15), [0.6212, 0.9626])
        self.assertEqual(tables.wilson(15, 15)[1], 1.0)
        self.assertIsNone(tables.wilson(0, 0))

    def test_mcnemar_exact_two_sided(self):
        self.assertEqual(tables.mcnemar(6, 1), 0.125)
        self.assertEqual(tables.mcnemar(9, 0), 0.0039)
        self.assertEqual(tables.mcnemar(0, 0), 1.0)
        self.assertEqual(tables.mcnemar(3, 3), 1.0)
        self.assertEqual(tables.mcnemar(1, 6), tables.mcnemar(6, 1))


class GroupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self.tmp.name) / 'runs'
        self.runs.mkdir()
        self.out = Path(self.tmp.name) / 'out'

    def tearDown(self):
        self.tmp.cleanup()

    def board_batch(self, name='batch_board_int8_stage1_layouts', episodes=None, fail=()):
        d = write_batch(self.runs, name, 'board', 'int8', episodes or all_episodes())
        for item in episodes or all_episodes():
            seed, target = item.split(':')
            outcome = 'grasp_fail' if (int(seed), target) in fail else 'delivered'
            park = ((0.0, 0.03), (0.0, 0.031), (0.0, 0.029)) if outcome == 'grasp_fail' else ((0.0, 0.01),)
            write_episode(d / 'episodes' / f'{seed}_{target}', int(seed), target, outcome, board=True, park=park)
        return d

    def test_headline_counts_rate_interval_and_pass(self):
        fail = {(1000 + i, 'bus') for i in range(10)}
        self.board_batch(fail=fail)
        facts = tables.build(self.runs, self.out, None, dev=False)
        g = facts['groups']['board_int8']
        self.assertEqual((g['expected'], g['counted']), (60, 60))
        self.assertEqual(g['delivered'], {'k': 50, 'n': 60, 'rate': 0.8333, 'wilson95': [0.7197, 0.9069]})
        self.assertEqual(g['ends'], {'delivered': 50, 'grasp_fail': 10})
        self.assertEqual(g['per_target']['bus']['delivered']['k'], 10)
        self.assertTrue(facts['headline']['pass'])
        self.assertEqual(g['park_err_cm']['lateral_mean_sd'][0], round(100 * (50 * 0.01 + 10 * 0.09) / 80, 2))
        self.assertEqual(g['board']['robotd_missed_allowed'], 8)      # 6/90k x 120k ticks
        self.assertEqual(facts['failures']['n'], 10)
        with (self.out / 'failures.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertIn('3 closes, none held', rows[0]['auto_cause'])
        self.assertIn('pending', facts)
        self.assertIn('4D: no complete healthy 1F run after the freeze', facts['pending'])

    def test_attempts_without_a_result_are_listed_not_scored(self):
        d = self.board_batch()
        ep = d / 'episodes' / '1003_henry'
        ep.rename(d / 'episodes' / '1003_henry.interrupted1')
        write_episode(ep, 1003, 'henry', 'grasp_fail', board=True)              # the rerun failed: it counts
        bad = d / 'episodes' / '1004_bus'
        bad.rename(d / 'episodes' / '1004_bus.infra1')
        write_episode(bad, 1004, 'bus', 'delivered', board=True, health_ok=False)   # still not a result
        g = tables.build(self.runs, self.out, None, dev=False)['groups']['board_int8']
        self.assertEqual(g['counted'], 59)
        self.assertEqual(g['not_counted'], [{'episode': '1004_bus', 'attempts': 2}])
        self.assertEqual(g['attempts_not_counted']['by_status'], {'infra': 2, 'interrupted': 1})
        self.assertEqual(g['delivered']['k'], 58)
        self.assertIsNone(tables.build(self.runs, self.out, None, dev=False)['headline']['pass'])

    def test_three_brain_errors_make_a_failure_two_do_not(self):
        d = self.board_batch()
        for key, n in (('1005_ketchup', 3), ('1006_bus', 2)):
            seed, target = key.split('_')
            shutil.rmtree(d / 'episodes' / key)
            for k in range(1, n):                               # earlier attempts, renamed by the runner
                write_episode(d / 'episodes' / f'{key}.infra{k}', int(seed), target, 'error', board=True)
            write_episode(d / 'episodes' / key, int(seed), target, 'error', board=True)   # the last one
        g = tables.build(self.runs, self.out, None, dev=False)['groups']['board_int8']
        self.assertEqual(g['repeated_errors'], ['1005_ketchup'])
        self.assertEqual(g['ends'].get('error'), 1)
        self.assertEqual([m['episode'] for m in g['not_counted']], ['1006_bus'])
        self.assertEqual(g['counted'], 59)

    def test_folders_of_a_group_merge_and_the_earlier_duplicate_counts(self):
        first = all_episodes(SEEDS[:10])
        second = all_episodes(SEEDS[9:])
        a = write_batch(self.runs, 'batch_board_int8_stage1_layouts_1000-1009', 'board', 'int8', first)
        b = write_batch(self.runs, 'batch_board_int8_stage1_layouts_1009-1019', 'board', 'int8', second)
        for d, items, start in ((a, first, '20261028T050000Z'), (b, second, '20261028T060000Z')):
            for item in items:
                seed, target = item.split(':')
                outcome = 'grasp_fail' if (d is b and seed == '1009') else 'delivered'
                write_episode(d / 'episodes' / f'{seed}_{target}', int(seed), target, outcome, board=True, started=start)
        g = tables.build(self.runs, self.out, None, dev=False)['groups']['board_int8']
        self.assertEqual(g['counted'], 60)
        self.assertEqual(g['delivered']['k'], 60)          # 1009 from the earlier folder
        self.assertEqual(len(g['duplicate_results_ignored']), 3)
        self.assertEqual(len(g['folders']), 2)

    def test_dev_batches_and_other_backends_stay_out_of_the_formal_groups(self):
        self.board_batch()
        dev = write_batch(self.runs, 'batch_board_int8_stage3_layouts_3003-3004', 'board', 'int8',
                          ['3003:bus'], layouts=tables.DEV_LAYOUTS, formal=False)
        write_episode(dev / 'episodes' / '3003_bus', 3003, 'bus', 'grasp_fail', board=True)
        fp16 = write_batch(self.runs, 'batch_board_fp16_stage1_layouts_1000-1004', 'board', 'fp16',
                           all_episodes(SEEDS[:5]))
        for item in all_episodes(SEEDS[:5]):
            seed, target = item.split(':')
            write_episode(fp16 / 'episodes' / f'{seed}_{target}', int(seed), target,
                          'grasp_fail' if target == 'henry' else 'delivered', board=True, p99=50.0)
        facts = tables.build(self.runs, self.out, None, dev=False)
        self.assertEqual(sorted(facts['groups']), ['board_fp16', 'board_int8'])
        self.assertEqual(facts['groups']['board_fp16']['expected'], 15)
        c = facts['comparisons']['board_fp16_vs_board_int8']
        self.assertEqual((c['pairs'], c['a_only'], c['b_only']), (15, 0, 5))
        self.assertEqual(c['mcnemar_p'], 0.0625)
        dep = [r for r in facts['deployment'] if r['source'].startswith('closed loop')]
        self.assertEqual([r['backend'] for r in dep], ['fp16', 'int8'])
        self.assertEqual(dep[0]['e2e_ms_p99_worst'], 50.0)
        dev_facts = tables.build(self.runs, self.out / 'dev', None, dev=True)
        self.assertEqual(dev_facts['groups']['board_int8']['counted'], 1)

    def test_the_colour_batch_is_told_apart_by_its_eyes(self):
        self.assertEqual(tables.eyes_of(''), 'yolo')
        self.assertEqual(tables.eyes_of('--eyes colour'), 'colour')
        self.assertEqual(tables.eyes_of("--eyes=colour --viewer"), 'colour')
        mac = write_batch(self.runs, 'batch_mac_cpu_stage1_layouts', 'mac', 'cpu', all_episodes(SEEDS[:2]))
        colour = write_batch(self.runs, 'batch_mac_colour_stage1_layouts', 'mac', 'cpu', all_episodes(SEEDS[:2]),
                             extra='--eyes colour')
        oracle = write_batch(self.runs, 'batch_mac_oracle', 'mac', 'cpu', ['1000:bus'], extra='--eyes oracle')
        write_episode(oracle / 'episodes' / '1000_bus', 1000, 'bus')
        for item in all_episodes(SEEDS[:2]):
            seed, target = item.split(':')
            write_episode(mac / 'episodes' / f'{seed}_{target}', int(seed), target)
            write_episode(colour / 'episodes' / f'{seed}_{target}', int(seed), target,
                          'wrong_toy' if target == 'henry' else 'delivered')
        facts = tables.build(self.runs, self.out, None, dev=False)
        self.assertEqual(sorted(facts['groups']), ['mac_colour', 'mac_cpu'])     # the oracle batch is nobody's
        g = facts['groups']['mac_colour']
        self.assertEqual((g['backend'], g['eyes'], g['delivered']['k'], g['wrong_toy']), ('cpu', 'colour', 4, 2))
        self.assertEqual(facts['groups']['mac_cpu']['delivered']['k'], 6)
        c = facts['comparisons']['mac_colour_vs_mac_cpu']
        self.assertEqual((c['pairs'], c['a_only'], c['b_only']), (6, 0, 2))
        det = [r for r in facts['detection'] if r['source'].startswith('closed loop')]
        self.assertEqual([r['detector'] for r in det], [tables.LABELS['mac_cpu'], tables.LABELS['mac_colour']])

    def test_a_layouts_path_typed_differently_still_matches(self):
        d = write_batch(self.runs, 'batch_board_int8_x', 'board', 'int8', ['1000:bus'],
                        layouts='./configs/control/stage1_layouts.json')
        write_episode(d / 'episodes' / '1000_bus', 1000, 'bus', board=True)
        self.assertIn('board_int8', tables.build(self.runs, self.out, None, dev=False)['groups'])

    def test_two_runs_write_the_same_tables(self):
        self.board_batch(fail={(1001, 'henry')})
        mac = write_batch(self.runs, 'batch_mac_cpu_stage1_layouts', 'mac', 'cpu', all_episodes())
        for item in all_episodes():
            seed, target = item.split(':')
            write_episode(mac / 'episodes' / f'{seed}_{target}', int(seed), target,
                          'drop_missed' if seed == '1002' else 'delivered')
        a, b = self.out / 'a', self.out / 'b'
        fa = tables.build(self.runs, a, None, dev=False)
        tables.build(self.runs, b, None, dev=False)
        for name in ('stage4_facts.json', 'episodes.csv', 'failures.csv', 't1_main.csv', 't5_deployment.csv'):
            self.assertEqual((a / name).read_bytes(), (b / name).read_bytes(), name)
        c = fa['comparisons']['board_int8_vs_mac_cpu']
        self.assertEqual((c['a_only'], c['b_only'], c['a_only_episodes']), (3, 1, ['1002_bus', '1002_henry', '1002_ketchup']))

    def test_failure_notes_are_merged(self):
        self.board_batch(fail={(1001, 'henry')})
        self.out.mkdir()
        (self.out / 'failure_notes.json').write_text(json.dumps({'board_int8/1001_henry': 'parked 3 cm right'}))
        tables.build(self.runs, self.out, None, dev=False)
        with (self.out / 'failures.csv').open() as f:
            self.assertEqual(next(csv.DictReader(f))['note'], 'parked 3 cm right')


class LastSegmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def onef(self, name: str, stall: bool = False, starts: int = 30, blind_held=None):
        d = self.runs / name
        d.mkdir()
        blind_held = blind_held if blind_held is not None else set(range(starts))

        def start(i, part, held):
            return {'part': part, 'i': i, 'toy': episode.TARGETS[i % 3], 'closes': 1 if held else 3, 'held': held,
                    'held_within_3': held, 'start_s': 15.0, 'fell': False,
                    'picks': [{'ran': True, 'grasped': held, 'park_err_m': [0.0, 0.01]}]}
        res = {'f_main': [start(i, 'f_main', i % 6 != 0) for i in range(starts)],
               'f_blind': [start(i, 'f_blind', i in blind_held) for i in range(starts)]}
        (d / 'summary.json').write_text(json.dumps({'meta': {'repo_commit': 'cc0931a'}, 'results': res}))
        with (d / 'truth.csv').open('w') as f:
            f.write('t,mono\n')
            mono = 100.0
            for k in range(500):                                # a stalled body: every 5th step waits 70 ms
                f.write(f'{k * 0.02:.3f},{mono:.4f}\n')
                mono += 0.07 if stall and k % 5 == 0 else 0.02
        return d

    def test_the_earliest_healthy_complete_run_after_the_freeze(self):
        self.onef('20261027T090000Z_s1f_s2')                    # before the freeze
        self.onef('20261028T060000Z_s1f_s2', stall=True)        # unhealthy
        self.onef('20261028T061000Z_s1f_s2', starts=10)         # not complete
        good = self.onef('20261028T062000Z_s1f_s2', blind_held=set(range(29)))
        self.onef('20261028T070000Z_s1f_s2')                    # later: not chosen
        seg = tables.last_segment(self.runs, None)
        self.assertEqual(Path(seg['run']).name, good.name)
        self.assertEqual([Path(t['run']).name for t in seg['tried']],
                         ['20261028T060000Z_s1f_s2', '20261028T062000Z_s1f_s2'])
        self.assertFalse(seg['tried'][0]['sim_health']['ok'])
        self.assertEqual(seg['inching']['held_within_3']['k'], 25)
        self.assertEqual(seg['blind']['held_within_3']['k'], 29)
        p = seg['paired_held_within_3']
        self.assertEqual((p['blind_only'], p['inching_only']), (5, 1))
        self.assertEqual(p['mcnemar_p'], tables.mcnemar(5, 1))


class LoadTests(unittest.TestCase):
    def test_samples_inside_a_groups_episodes_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            run = runs / '20261028T050000Z_board_load'
            (run / 'board').mkdir(parents=True)
            (run / 'meta.json').write_text(json.dumps({'offset_s': 10.0}))
            t0 = tables.utc_seconds('20261028T050100Z')
            with (run / 'board' / 'load.csv').open('w') as f:
                f.write('t_unix,cpu_all_pct,cpu4_pct,npu0_pct\n')
                for k in range(200):                           # board clock 10 s behind the Mac
                    f.write(f'{t0 - 10 + k:.3f},{10 + (k >= 100) * 40},{50},{7}\n')
            rows = {'board_int8': [{'started_utc': '20261028T050100Z', 'elapsed_s': 40.0}],
                    'board_cpu': [{'started_utc': '20261028T050240Z', 'elapsed_s': 40.0}],
                    'mac_cpu': [{'started_utc': '20261028T050100Z', 'elapsed_s': 40.0}]}
            load = tables.load_by_group(runs, rows)
            self.assertEqual(sorted(load), ['board_cpu', 'board_int8'])
            self.assertEqual(load['board_int8']['samples'], 56)
            self.assertEqual(load['board_int8']['cpu_all_pct'], 10.0)
            self.assertEqual(load['board_cpu']['cpu_all_pct'], 50.0)


class BoardLoadScriptTests(unittest.TestCase):
    def test_one_sample_row_and_summary(self):
        spec = importlib.util.spec_from_file_location('board_load', REPO / 'scripts' / 'stage4' / 'board_load.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not Path('/proc/stat').exists():
            self.skipTest('no /proc (not Linux)')
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(mod, 'npu', return_value=None):
            self.assertEqual(mod.main(['--out', tmp, '--period', '0.05', '--duration', '0.2']), 0)
            rows = list(csv.DictReader((Path(tmp) / 'load.csv').open()))
            self.assertGreaterEqual(len(rows), 2)
            self.assertIn('cpu0_pct', rows[0])
            self.assertTrue(rows[0]['cpu_all_pct'])
            self.assertEqual(json.loads((Path(tmp) / 'summary.json').read_text())['rows'], len(rows))


if __name__ == '__main__':
    unittest.main()
