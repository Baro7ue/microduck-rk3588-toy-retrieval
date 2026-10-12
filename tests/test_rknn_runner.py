# Moved 2026-10-11 from work/tests/test_probe_board_yolo11_npu_camera.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""NPU adapter contracts with a fake SDK; these tests do not execute the NPU."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
SOURCE = REPO / 'src/microduck_toy/vision/rknn_runner.py'
MODELS = REPO / 'models/yolo/yolo11n_coco'
probe = None
if SOURCE.exists():
    spec = importlib.util.spec_from_file_location('board_yolo_npu_camera', SOURCE)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

from test_yolo_camera_probe import camera


class FakeSDK:
    def __init__(self):
        self.release_count = 0
        self.inputs = []
        self.load_code = self.init_code = 0
        self.output = [np.zeros((1, 84, 2100), dtype=np.float32)]

    def load_rknn(self, path):
        self.loaded_path = path
        return self.load_code

    def init_runtime(self):
        return self.init_code

    def get_sdk_version(self):
        return ('fake-api', 'fake-driver')

    def inference(self, *, inputs, data_format, inputs_pass_through):
        self.inputs.append((inputs[0].copy(), data_format, inputs_pass_through))
        return self.output

    def release(self):
        self.release_count += 1


class NPUCameraTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(probe, 'the NPU camera adapter is missing')
        self.temporary = tempfile.TemporaryDirectory(prefix='camera-npu-contract-')
        self.addCleanup(self.temporary.cleanup)
        self.model = Path(self.temporary.name) / 'yolo11n_coco_fp16_320.rknn'
        shutil.copyfile(MODELS / self.model.name, self.model)
        self.conversion = json.loads((MODELS / 'conversion.json').read_text())
        self.save_conversion()
        self.sdk = FakeSDK()
        api = ModuleType('rknnlite.api')
        api.RKNNLite = lambda: self.sdk
        package = ModuleType('rknnlite')
        package.api = api
        self.modules = mock.patch.dict(sys.modules, {'rknnlite': package, 'rknnlite.api': api})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.metadata = mock.patch.object(probe, 'version', return_value='2.3.2')
        self.metadata.start()
        self.addCleanup(self.metadata.stop)

    def save_conversion(self):
        (self.model.parent / 'conversion.json').write_text(json.dumps(self.conversion))

    def test_real_uyvy_preprocessing_supplies_rgb_uint8_nhwc_without_double_scaling(self):
        # BGR/RGB reversal, NCHW, or dividing twice breaks the actual SDK feed contract.
        detector = probe.NPUCameraDetector(self.model)
        self.addCleanup(detector.close)
        self.assertEqual(len(self.sdk.inputs), 5)
        for pixels, layout, passthrough in self.sdk.inputs:
            self.assertEqual(pixels.shape, (1, 320, 320, 3))
            self.assertEqual(pixels.dtype, np.uint8)
            self.assertFalse(pixels.any())
            self.assertEqual((layout, passthrough), (['nhwc'], [0]))
        raw = self.sdk.output[0]
        raw[0, :4, :3] = [[160, 160, 160], [160, 160, 160],
                          [40, 40, 40], [20, 20, 20]]
        raw[0, 4, :2] = [0.9, 0.8]
        raw[0, 36, 2] = 0.7
        payload = bytes([90, 81, 240, 81]) * (640 * 360 // 2)
        result, annotated = detector.process(payload)
        pixels, layout, passthrough = self.sdk.inputs[-1]
        self.assertEqual(pixels.shape, (1, 320, 320, 3))
        self.assertEqual(pixels.dtype, np.uint8)
        self.assertTrue(pixels.flags.c_contiguous)
        np.testing.assert_array_equal(pixels[0, 160, 69], [114, 114, 114])
        np.testing.assert_array_equal(pixels[0, 160, 70], [238, 14, 14])
        self.assertEqual((layout, passthrough), (['nhwc'], [0]))
        self.assertEqual(annotated.shape, (640, 360, 3))
        self.assertEqual([item['class_name'] for item in result['detections']],
                         ['person', 'sports ball'])
        np.testing.assert_allclose(result['detections'][0]['xyxy'], [140, 300, 220, 340])
        self.assertGreater(result['inference_call_ms'], 0)
        self.assertGreaterEqual(result['complete_processing_ms'], result['inference_call_ms'])
        self.assertEqual(detector.runtime_info['runtime'], 'RKNNLite')
        self.assertEqual(detector.runtime_info['sdk_version'], "('fake-api', 'fake-driver')")
        detector.close()
        detector.close()
        self.assertEqual(self.sdk.release_count, 1)

    def test_mismatched_conversion_provenance_or_std255_is_rejected_before_runtime(self):
        # A different model or compiler normalization must not silently enter the camera run.
        valid = self.conversion.copy()
        for field, wrong in [('rknn_sha256', '0' * 64), ('onnx_sha256', '0' * 64),
                             ('target_platform', 'rk3568'), ('do_quantization', True),
                             ('mean_values', [[1, 1, 1]]), ('std_values', [[1, 1, 1]])]:
            with self.subTest(field=field):
                self.conversion = {**valid, field: wrong}
                self.save_conversion()
                with self.assertRaises(ValueError):
                    probe.NPUCameraDetector(self.model)
                self.assertFalse(hasattr(self.sdk, 'loaded_path'))
        self.conversion = valid
        self.save_conversion()
        with self.model.open('ab') as stream:
            stream.write(b'changed model')
        with self.assertRaises(ValueError):
            probe.NPUCameraDetector(self.model)

    def test_opt_in_orange_output_keeps_raw_class_and_coordinates(self):
        # Relabeling every orange as ball would hide the model's actual classification.
        detector = probe.NPUCameraDetector(self.model, include_orange=True)
        self.addCleanup(detector.close)
        raw = self.sdk.output[0]
        raw[0, :4, 0] = [160, 160, 40, 20]
        raw[0, 4 + 49, 0] = .8
        payload = bytes([128, 100, 128, 100]) * (640 * 360 // 2)
        result, _ = detector.process(payload)
        target, = result['detections']
        self.assertEqual((target['class_id'], target['class_name']), (49, 'orange'))
        np.testing.assert_allclose(target['xyxy'], [140, 300, 220, 340])

    def test_failed_loading_initialization_or_warmup_releases_runtime(self):
        # Every failure after RKNNLite allocation must release it, including malformed outputs.
        for load, init, output in [(-1, 0, None), (0, -2, None), (0, 0, None),
                                   (0, 0, [np.zeros((1, 84, 2100), dtype=np.float64)])]:
            with self.subTest(load=load, init=init, dtype=getattr(output, 'dtype', None)):
                self.sdk.release_count = 0
                self.sdk.load_code, self.sdk.init_code = load, init
                self.sdk.output = output
                with self.assertRaises((ValueError, RuntimeError)):
                    probe.NPUCameraDetector(self.model)
                self.assertEqual(self.sdk.release_count, 1)

    def test_malformed_runtime_output_is_not_published_as_detections(self):
        # Wrong shape, nonfinite output, or multiple output tensors must fail the frame.
        detector = probe.NPUCameraDetector(self.model)
        self.addCleanup(detector.close)
        payload = bytes([128, 100, 128, 100]) * (640 * 360 // 2)
        nonfinite = np.zeros((1, 84, 2100), dtype=np.float32)
        nonfinite[0, 0, 0] = np.nan
        for output in [[np.zeros((1, 84, 2099), dtype=np.float32)], [nonfinite],
                       [nonfinite, nonfinite]]:
            with self.subTest(shape=[item.shape for item in output]):
                self.sdk.output = output
                with self.assertRaises(ValueError):
                    detector.process(payload)

    def run_camera_fixture(self):
        directory = self.model.parent / 'camera-run'
        with camera() as (address, payload), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            status = probe.main(duration=0.25, address=address, model_path=self.model,
                                directory=directory)
        summary = json.loads((directory / 'summary.json').read_text())
        detections = [json.loads(line) for line in
                      (directory / 'detections.jsonl').read_text().splitlines()]
        frames = [json.loads(line) for line in
                  (directory / 'frames.jsonl').read_text().splitlines()]
        return status, summary, detections, frames, directory, payload

    def test_npu_camera_loop_records_runtime_and_inference_samples_then_releases(self):
        # Wrong backend dispatch or a CPU-only report path would fail despite valid frame inference.
        status, summary, detections, frames, directory, payload = self.run_camera_fixture()
        self.assertEqual(status, 0)
        self.assertEqual(summary['outcome'], 'completed')
        self.assertEqual(summary['runtime']['backend'], 'npu')
        self.assertEqual(summary['runtime']['runtime'], 'RKNNLite')
        self.assertEqual(summary['runtime']['version'], '2.3.2')
        self.assertNotIn('providers', summary)
        self.assertNotIn('intra_op_threads', summary)
        self.assertNotIn('session_run_ms', summary['timing_ms'])
        self.assertEqual(summary['processed_frame_count'], len(detections))
        self.assertGreaterEqual(len(detections), 2)
        self.assertEqual(summary['received_frame_count'], len(frames))
        self.assertEqual(len(self.sdk.inputs), 5 + len(detections))
        self.assertEqual(self.sdk.release_count, 1)
        recorded_frames = {frame['index']: frame for frame in frames}
        samples = []
        for detection in detections:
            samples.append(detection['inference_call_ms'])
            self.assertGreater(detection['inference_call_ms'], 0)
            self.assertNotIn('session_run_ms', detection)
            self.assertEqual(detection['payload_sha256'],
                             recorded_frames[detection['receiver_index']]['payload_sha256'])
        self.assertAlmostEqual(summary['timing_ms']['inference_call_ms']['p50'],
                               float(np.median(samples)))
        self.assertAlmostEqual(summary['timing_ms']['inference_call_ms']['p95'],
                               float(np.percentile(samples, 95)))
        self.assertEqual((directory / 'camera_first.uyvy').read_bytes(), payload)
        self.assertTrue((directory / 'first_annotated.jpg').exists())
        self.assertTrue((directory / 'last_annotated.jpg').exists())
        self.assertFalse((directory / 'error.txt').exists())

    def test_failed_frame_inference_keeps_partial_evidence_and_releases_runtime(self):
        # A frame exception must revoke success without losing prior results or leaking RKNNLite.
        inference = self.sdk.inference

        def fail_second_frame(**kwargs):
            if len(self.sdk.inputs) >= 6:
                raise RuntimeError('fixture NPU frame inference failed')
            return inference(**kwargs)

        with mock.patch.object(self.sdk, 'inference', side_effect=fail_second_frame):
            status, summary, detections, frames, directory, _ = self.run_camera_fixture()
        self.assertEqual(status, 1)
        self.assertEqual(summary['outcome'], 'failed')
        self.assertEqual(summary['processed_frame_count'], 1)
        self.assertEqual(len(detections), 1)
        self.assertGreaterEqual(len(frames), 2)
        self.assertEqual(self.sdk.release_count, 1)
        error = (directory / 'error.txt').read_text()
        self.assertIn('Traceback', error)
        self.assertIn('RuntimeError: fixture NPU frame inference failed', error)
        self.assertEqual(summary['error'], error)
        self.assertEqual(summary['timing_ms']['inference_call_ms']['count'], 1)
        self.assertTrue((directory / 'first_annotated.jpg').exists())
        self.assertTrue((directory / 'last_annotated.jpg').exists())


if __name__ == '__main__':
    unittest.main()
