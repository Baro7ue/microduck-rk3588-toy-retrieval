"""Stage 4E's detector (stage3/eyes.py Colour): toys by Stage 2's colour threshold on YOLO's own input,
the tray by YOLO, and the --eyes colour option of the brain and the episode runner."""
from __future__ import annotations

import json
from pathlib import Path
import time
import unittest

import numpy as np

from microduck_toy.stage3 import brain, episode, eyes
from microduck_toy.vision import views

REPO = Path(__file__).resolve().parents[1]


class FakeLive:
    """LiveDetector's detect(): a tray and a (wrong) YOLO bus box, whatever the frame shows."""

    def __init__(self):
        self.calls, self.closed = 0, False

    def detect(self, frame, state, view, margin):
        self.calls += 1
        dets = [{'cls': 'tray', 'conf': 0.9, 'xyxy': [10.0, 500.0, 120.0, 560.0]},
                {'cls': 'bus', 'conf': 0.8, 'xyxy': [0.0, 0.0, 5.0, 5.0]}]
        return dets, views.letterbox_mapping(), {'e2e': 20.0, 'inference': 15.0}

    def close(self):
        self.closed = True


def frame_with(blocks) -> eyes.Frame:
    """An upright 640x360 frame (rows x columns) on a grey floor with coloured blocks, sent as UYVY."""
    img = np.full((views.FRAME_H, views.FRAME_W, 3), 128, np.uint8)
    for (x0, y0, x1, y1), rgb in blocks:
        img[y0:y1, x0:x1] = rgb
    now = time.perf_counter()
    return eyes.Frame(1, eyes.upright_to_uyvy(img), now, now, time.monotonic())


class ColourTests(unittest.TestCase):
    def setUp(self):
        s2 = json.loads((REPO / 'configs' / 'vision' / 'stage2.json').read_text())
        self.live = FakeLive()
        self.det = eyes.Colour(eyes.Yolo(self.live, 30), s2['baselines']['colour'], float(s2['detect']['conf']))

    def test_toys_by_colour_tray_by_yolo(self):
        yellow, green, red = (235, 190, 20), (40, 160, 60), (200, 20, 20)
        f = frame_with([((100, 300, 160, 340), yellow), ((200, 400, 260, 430), green), ((40, 200, 70, 220), red)])
        boxes, kind, ms = self.det.boxes(f, None, 'approach')
        by = {b['cls']: b for b in boxes}
        self.assertEqual(sorted(by), ['bus', 'henry', 'ketchup', 'tray'])
        self.assertEqual(kind, 'letterbox')
        for cls, want in (('bus', (100, 300, 160, 340)), ('henry', (200, 400, 260, 430)), ('ketchup', (40, 200, 70, 220))):
            for got, w in zip(by[cls]['xyxy'], want):
                self.assertAlmostEqual(got, w, delta=4.0, msg=cls)
        self.assertEqual(by['tray']['xyxy'], [10.0, 500.0, 120.0, 560.0])     # YOLO's tray, YOLO's bus dropped
        self.assertNotIn([0.0, 0.0, 5.0, 5.0], [b['xyxy'] for b in boxes])
        self.assertIn('colour', ms)
        self.assertEqual(ms['e2e'], 20.0)

    def test_the_baseline_has_no_croc_class(self):
        self.assertEqual(sorted(self.det.colour.names), ['bus', 'henry', 'ketchup', 'thomas'])

    def test_specks_below_yolos_conf_are_dropped(self):
        boxes, _, _ = self.det.boxes(frame_with([((100, 300, 104, 304), (235, 190, 20))]), None, 'approach')
        self.assertEqual([b['cls'] for b in boxes], ['tray'])           # 2x2 input px: score < 0.25

    def test_close_closes_yolo(self):
        self.det.close()
        self.assertTrue(self.live.closed)


class OptionTests(unittest.TestCase):
    def test_brain_and_episode_take_eyes_colour(self):
        a = brain.parser().parse_args(['--target', 'bus', '--start=0,0,0', '--socket', 's', '--out', 'o',
                                       '--eyes', 'colour', '--backend', 'cpu'])
        self.assertEqual(a.eyes, 'colour')
        self.assertEqual(episode.parser().parse_args(['3e', '--eyes', 'colour']).eyes, 'colour')


if __name__ == '__main__':
    unittest.main()
