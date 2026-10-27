"""Stage 3G batch runs without a robot: the episode list, batch.json on resume, which episode folders
count as results, the table, the freeze check (stage3/batch.py), and the episode options the runner uses."""
from __future__ import annotations

import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from microduck_toy.stage3 import batch, episode

REPO = Path(__file__).resolve().parents[1]


def write_episode(ep_dir: Path, seed=3000, target='bus', outcome='delivered', brain_end=None, board=False,
                  health_ok=True, log=True, meta=None, **extra) -> Path:
    """An episode folder as `episode 3e --out` leaves it (only the fields batch.py reads)."""
    brain_end = brain_end or ('dropped' if outcome in ('delivered', 'drop_missed') else outcome)
    label = f'e_{seed}_{target}'
    result = {'part': 'e', 'label': label, 'seed': seed, 'target': target, 'outcome': outcome,
              'brain_end': brain_end, 'delivered_strict': outcome == 'delivered', 'elapsed_s': 40.0,
              'closes': [{'result': 'grasped', 'toy': target}], 'grasped': 1, 'falls': [], 'tof_stops': [],
              'states_s': {'SPIN': 7.0}, 'disturbed': [],
              'latency_e2e_p99_ms': 45.0 if board else None}
    if board:
        result['robotd'] = {'ticks_expected': 2000, 'missed': 0, 'hz_min': 49.9}
    result.update(extra)
    doc = {'meta': {'target': 'board' if board else 'mac', **(meta or {})},
           'sim_health': {'ok': health_ok, 'stalled': 0.0 if health_ok else 0.2, 'odom_ratio': {'ratio': 0.99}},
           'results': [result],
           'trials': [{'seed': seed, 'target': target, 'brain': {'end': brain_end, 'error': 'boom'}}]}
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / 'summary.json').write_text(json.dumps(doc))
    if log:
        (ep_dir / 'episodes' / label).mkdir(parents=True)
        (ep_dir / 'episodes' / label / 'brain.jsonl').write_text('{}\n')
    return ep_dir


class PlanTests(unittest.TestCase):
    def test_seed_by_seed_like_3e(self):
        self.assertEqual(batch.plan('configs/control/stage3_layouts.json', '3003-3004', 'bus,henry,ketchup', False),
                         ['3003:bus', '3003:henry', '3003:ketchup', '3004:bus', '3004:henry', '3004:ketchup'])

    def test_every_seed_of_the_file_by_default(self):
        eps = batch.plan('configs/control/stage3_layouts.json', None, 'henry', False)
        self.assertEqual(eps, [f'{s}:henry' for s in range(3000, 3010)])

    def test_formal_seeds_need_formal(self):
        with self.assertRaises(SystemExit) as caught:
            batch.plan('configs/control/stage1_layouts.json', '1000-1001', 'bus', False)
        self.assertIn('formal', str(caught.exception))
        self.assertEqual(batch.plan('configs/control/stage1_layouts.json', '1019', 'bus', True), ['1019:bus'])

    def test_refuses_seeds_not_in_the_file_and_unknown_or_repeated_targets(self):
        for seeds, targets in (('2999-3000', 'bus'), ('3000', 'croc'), ('3000', 'bus,bus'), ('3000', ''),
                               ('3001-3000', 'bus'), ('30a0', 'bus')):
            with self.subTest(seeds=seeds, targets=targets), self.assertRaises(SystemExit):
                batch.plan('configs/control/stage3_layouts.json', seeds, targets, False)


class BeginTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / 'batch_x'
        self.settings = {'target': 'board', 'layouts': 'configs/control/stage3_layouts.json',
                         'episodes': ['3003:bus', '3003:henry'], 'backend': 'int8', 'formal': False,
                         'code_sha256': 'a' * 64, 'extra': ''}

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_call_writes_the_settings_and_the_layouts_checksum(self):
        doc = batch.begin(self.dir, self.settings, 'c0ffee', False)
        saved = json.loads((self.dir / 'batch.json').read_text())
        self.assertEqual(saved, doc)
        self.assertEqual(len(saved['settings']['layouts_sha256']), 64)
        self.assertEqual(saved['invocations'][0]['commit'], 'c0ffee')

    def test_a_resume_with_the_same_code_adds_an_invocation(self):
        batch.begin(self.dir, self.settings, 'c0ffee', True)
        doc = batch.begin(self.dir, self.settings, 'beef', False)      # committed in between: same bundle
        self.assertEqual([i['commit'] for i in doc['invocations']], ['c0ffee', 'beef'])

    def test_a_resume_with_other_code_or_settings_is_refused(self):
        batch.begin(self.dir, self.settings, None, None)
        for key, value in (('code_sha256', 'b' * 64), ('episodes', ['3003:bus']), ('backend', 'fp16')):
            with self.subTest(key=key), self.assertRaises(SystemExit) as caught:
                batch.begin(self.dir, {**self.settings, key: value})
            self.assertIn(key, str(caught.exception))
        self.assertEqual(len(json.loads((self.dir / 'batch.json').read_text())['invocations']), 1)


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / '3000_bus'

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_finished_episode_is_done_whatever_its_end(self):
        for outcome in ('delivered', 'grasp_fail', 'search_fail', 'drop_missed', 'wrong_toy', 'timeout'):
            with self.subTest(outcome=outcome):
                d = write_episode(Path(self.tmp.name) / outcome, outcome=outcome)
                self.assertEqual(batch.status(d), ('done', outcome))

    def test_attempts_without_a_result(self):
        self.dir.mkdir()
        self.assertEqual(batch.status(self.dir)[0], 'incomplete')
        cases = [({'meta': {'interrupted': True}}, 'interrupted'),
                 ({'meta': {'error': 'TimeoutError: x'}}, 'infra'),
                 ({'brain_end': 'error'}, 'infra'),
                 ({'health_ok': False}, 'infra'),
                 ({'log': False}, 'infra')]
        for k, (kw, want) in enumerate(cases):
            with self.subTest(kw=kw):
                d = write_episode(Path(self.tmp.name) / f'case{k}', **kw)
                self.assertEqual(batch.status(d)[0], want)
        d = write_episode(Path(self.tmp.name) / 'brain', brain_end='error')
        self.assertIn('boom', batch.status(d)[1])

    def test_a_scorer_error_is_not_a_result(self):
        d = write_episode(self.dir)
        doc = json.loads((d / 'summary.json').read_text())
        doc['score_error'] = 'KeyError: x'
        (d / 'summary.json').write_text(json.dumps(doc))
        self.assertEqual(batch.status(d)[0], 'infra')

    def test_a_board_episode_needs_its_latency_and_robotd_logs(self):
        self.assertEqual(batch.status(write_episode(self.dir, board=True))[0], 'done')
        d = write_episode(Path(self.tmp.name) / 'lat', board=True, latency_e2e_p99_ms=None)
        self.assertEqual(batch.status(d), ('infra', '缺板端检测时延'))
        d = write_episode(Path(self.tmp.name) / 'ticks', board=True, robotd=None)
        self.assertEqual(batch.status(d)[0], 'infra')


class TableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / 'batch_board'

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, board):
        batch.begin(self.dir, {'target': 'board' if board else 'mac', 'layouts': 'configs/control/stage3_layouts.json',
                               'episodes': ['3000:bus', '3000:henry', '3000:ketchup'],
                               'backend': 'int8' if board else 'cpu', 'formal': False, 'code_sha256': 'a' * 64,
                               'extra': ''})
        eps = self.dir / 'episodes'
        write_episode(eps / '3000_bus', board=board)
        write_episode(eps / '3000_henry', target='henry', outcome='grasp_fail', board=board)
        write_episode(eps / '3000_ketchup.infra1', target='ketchup', brain_end='error', board=board)
        (eps / '3000_ketchup.interrupted1').mkdir()

    def test_counts_results_and_lists_the_attempts_that_did_not_count(self):
        self.make(board=False)
        text = io.StringIO()
        out = batch.table(self.dir, out=text)
        self.assertEqual((out['planned'], out['done']), (3, 2))
        self.assertEqual(out['totals']['delivered'], 1)
        self.assertEqual(out['totals']['ends'], {'delivered': 1, 'grasp_fail': 1})
        self.assertEqual(out['per_target'], {'bus': {'done': 1, 'delivered': 1}, 'henry': {'done': 1, 'delivered': 0},
                                             'ketchup': {'done': 0, 'delivered': 0}})
        self.assertEqual([n['episode'] for n in out['not_done']], ['3000_ketchup'])
        self.assertEqual(sorted(f['folder'] for f in out['failed_attempts']),
                         ['3000_ketchup.infra1', '3000_ketchup.interrupted1'])
        self.assertIsNone(out['board'])
        self.assertEqual(json.loads((self.dir / 'summary.json').read_text())['done'], 2)
        self.assertIn('完成 2/3', text.getvalue())
        self.assertIn('3000_ketchup.infra1', text.getvalue())

    def test_board_batches_add_robotd_and_latency(self):
        self.make(board=True)
        out = batch.table(self.dir, out=io.StringIO())
        self.assertEqual(out['board']['robotd_ticks'], 4000)
        self.assertEqual(out['board']['robotd_missed'], 0)
        self.assertEqual(out['board']['e2e_p99_ms_worst_episode'], 45.0)

    def test_pad_counts_chinese_as_two_columns(self):
        self.assertEqual(batch.pad('结果', 6), '结果  ')
        self.assertEqual(batch.width('ab结果'), 6)


class FormalCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runs = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def trials(self, folder, *seeds):
        d = self.runs / folder
        d.mkdir(parents=True, exist_ok=True)
        (d / 'trials.jsonl').write_text(''.join(json.dumps({'seed': s}) + '\n' for s in seeds))

    def test_only_stage3_folders_and_batches_count(self):
        self.trials('20261027T054310Z_s33e_s2', 3000, 3001)
        self.trials('20261015T000000Z_s1a_s2', 1000)            # 1A loaded every formal layout, no brain
        r = batch.formal_check(self.runs)
        self.assertTrue(r['ok'])
        self.assertEqual((r['stage3_folders'], r['seeds_named']), (1, 2))
        b = self.runs / 'batch_mac_cpu_stage1_layouts_1000-1000'
        b.mkdir()
        (b / 'batch.json').write_text(json.dumps({'settings': {'episodes': ['1000:bus']}}))
        r = batch.formal_check(self.runs)
        self.assertFalse(r['ok'])
        self.assertEqual([h['seed'] for h in r['formal_hits']], [1000])

    def test_seeds_in_summaries_count_when_trials_jsonl_is_empty(self):
        d = self.runs / '20261027T101202Z_s33e_s2_board'
        d.mkdir()
        (d / 'trials.jsonl').write_text('')
        (d / 'summary.json').write_text(json.dumps({'trials': [{'seed': 1005}]}))
        self.assertFalse(batch.formal_check(self.runs)['ok'])


class EpisodeOptionTests(unittest.TestCase):
    def test_dev_episodes_read_another_layouts_file_but_keep_formal_seeds_out(self):
        with self.assertRaises(SystemExit):
            episode.dev_episodes('1000:bus', quick=False, path='configs/control/stage1_layouts.json')
        eps = episode.dev_episodes('1000:bus', quick=False, path='configs/control/stage1_layouts.json', formal=True)
        self.assertEqual(eps[0]['layout']['seed'], 1000)
        with self.assertRaises(SystemExit):
            episode.dev_episodes('3999:bus', quick=False)
        with self.assertRaises(SystemExit):
            episode.dev_episodes('3000:croc', quick=False)

    def test_the_runner_options(self):
        a = episode.parser().parse_args(['3e', '--out', '/x/3000_bus', '--layouts', 'f.json', '--formal'])
        self.assertEqual((a.out, a.layouts, a.formal), ('/x/3000_bus', 'f.json', True))
        a = episode.parser().parse_args(['3e'])
        self.assertEqual((a.out, a.layouts, a.formal), (None, None, False))

    def test_ctrl_c_stops_the_brain_on_the_board_but_not_pkill_itself(self):
        pattern = episode.BOARD_BRAIN_PATTERN
        self.assertTrue(re.search(pattern, '/home/linaro/venv/bin/python -u -m microduck_toy.stage3.brain --target bus'))
        with mock.patch.object(episode.subprocess, 'run') as run, mock.patch('builtins.print'):
            episode.stop_board_brain('RK3588')
        (args,), _ = run.call_args
        self.assertEqual(args[:2], ['ssh', 'RK3588'])
        self.assertIsNone(re.search(pattern, f'bash -c {args[2]}'))       # pkill's own shell line
        self.assertIn('pkill -INT -f', args[2])


if __name__ == '__main__':
    unittest.main()
