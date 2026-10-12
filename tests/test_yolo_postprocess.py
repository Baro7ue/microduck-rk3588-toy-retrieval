# Moved 2026-10-11 from work/tests/test_detect_yolo11_image.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Offline geometry and COCO postprocessing checks; no model or network required."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from microduck_toy.vision import yolo_postprocess as detector


class YoloImageTests(unittest.TestCase):
    def test_letterbox_odd_border_and_bgr_to_rgb(self):
        image = np.empty((362, 640, 3), dtype=np.uint8)
        image[:] = (20, 40, 80)
        tensor, gain, padding = detector.letterbox(image)
        self.assertEqual(tensor.shape, (1, 3, 320, 320))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(gain, 0.5)
        self.assertEqual(padding, (0, 69))
        np.testing.assert_allclose(tensor[0, :, 68, 0], [114 / 255] * 3)
        np.testing.assert_allclose(tensor[0, :, 69, 0], [80 / 255, 40 / 255, 20 / 255])
        np.testing.assert_allclose(tensor[0, :, 249, 0], [80 / 255, 40 / 255, 20 / 255])
        np.testing.assert_allclose(tensor[0, :, 250, 0], [114 / 255] * 3)

    def test_inverse_coordinates_remove_actual_border_and_clip(self):
        boxes = np.array([[10, 89, 300, 239], [-10, 60, 330, 260]], dtype=np.float32)
        restored = detector.restore_boxes(boxes, (362, 640), 0.5, (0, 69))
        np.testing.assert_allclose(restored, [[20, 40, 600, 340], [0, 0, 640, 362]])
        np.testing.assert_allclose(boxes[0], [10, 89, 300, 239])

    def test_nms_suppresses_same_class_but_preserves_overlapping_ball(self):
        boxes = np.array([[0, 0, 100, 100], [0, 0, 100, 100],
                          [0, 0, 100, 100], [200, 200, 250, 250]], dtype=np.float32)
        keep = detector.class_aware_nms(boxes, np.array([0.9, 0.8, 0.7, 0.6]),
                                        np.array([0, 0, 32, 0]), 0.45)
        self.assertEqual(keep.tolist(), [0, 2, 3])

    def test_argmax_uses_all_80_classes_before_filtering_targets(self):
        raw = np.zeros((1, 84, 2100), dtype=np.float32)
        raw[0, :4, :3] = np.array([[80, 80, 80], [80, 80, 80],
                                  [40, 40, 40], [20, 20, 20]])
        raw[0, 4, 0] = 0.9
        raw[0, 4 + 32, 1] = 0.85
        raw[0, 4, 2] = 0.5
        raw[0, 4 + 1, 2] = 0.95
        boxes, scores, classes = detector.decode_predictions(raw, 0.25)
        np.testing.assert_allclose(boxes, [[60, 70, 100, 90], [60, 70, 100, 90]])
        np.testing.assert_allclose(scores, [0.9, 0.85])
        self.assertEqual(classes.tolist(), [0, 32])

    def test_model_validation_rejects_wrong_ball_name_and_dynamic_shape(self):
        names = {i: f'class{i}' for i in range(80)}
        names.update({0: 'person', 32: 'sports ball'})
        input_node = SimpleNamespace(name='images', shape=[1, 3, 320, 320], type='tensor(float)')
        output_node = SimpleNamespace(name='output0', shape=[1, 84, 2100], type='tensor(float)')
        metadata = {'names': repr(names), 'task': 'detect'}
        session = SimpleNamespace(get_inputs=lambda: [input_node], get_outputs=lambda: [output_node],
                                  get_modelmeta=lambda: SimpleNamespace(custom_metadata_map=metadata))
        self.assertEqual(detector.validate_model(session)['names'][32], 'sports ball')
        names[32] = 'ball'
        metadata['names'] = json.dumps(names)
        with self.assertRaisesRegex(ValueError, 'sports ball'):
            detector.validate_model(session)
        metadata['names'] = repr({**names, 32: 'sports ball'})
        input_node.shape = ['batch', 3, 320, 320]
        with self.assertRaisesRegex(ValueError, 'input'):
            detector.validate_model(session)

    def test_orange_class_is_available_only_when_requested_without_relabeling(self):
        # Always filtering to 0/32 loses the tracked ball when YOLO calls it orange.
        raw = np.zeros((1, 84, 2100), dtype=np.float32)
        raw[0, :4, :2] = [[160, 160], [160, 160], [40, 40], [40, 40]]
        raw[0, 4 + 49, 0] = .8
        raw[0, 4 + 32, 1] = .7
        self.assertEqual(detector.decode_predictions(raw, .25)[2].tolist(), [32])
        boxes, scores, classes = detector.decode_predictions(raw, .25, class_ids=(0, 32, 49))
        self.assertEqual(classes.tolist(), [49, 32])
        np.testing.assert_allclose(scores, [.8, .7])
        np.testing.assert_allclose(boxes, [[140, 140, 180, 180]] * 2)

    def test_missing_model_preserves_full_traceback(self):
        with tempfile.TemporaryDirectory(prefix='yolo-image-error-') as temporary:
            root = Path(temporary)
            with contextlib.redirect_stderr(io.StringIO()):
                code = detector.main(['--model', str(root / 'missing.onnx'),
                                      '--image', str(root / 'missing.jpg'),
                                      '--output-dir', str(root / 'output')])
            self.assertEqual(code, 1)
            error = (root / 'output/error.txt').read_text()
            self.assertIn('Traceback', error)
            self.assertIn('FileNotFoundError', error)
            self.assertIn('missing.onnx', error)


if __name__ == '__main__':
    unittest.main()
