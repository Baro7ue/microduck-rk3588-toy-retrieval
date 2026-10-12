"""Hand-derived camera-ray rotation cases, independent of detector or MuJoCo."""
import importlib.util
import math
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'ball_follow_camera.py'
camera = None
if SOURCE.exists():
    spec = importlib.util.spec_from_file_location('ball_follow_camera', SOURCE)
    camera = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camera)

IDENTITY = [1., 0., 0., 0.]
FORWARD = [.5, -.5, .5, -.5]  # CV right/down/forward map to trunk -y/-z/+x.
FOCAL = 434.55844122715706


def yaw_camera(angle):
    # Explicit world-z yaw times FORWARD; no production quaternion helper in expectations.
    c, s = math.cos(angle/2), math.sin(angle/2)
    return [.5*(c+s), -.5*(c+s), .5*(c-s), .5*(-c+s)]


class CameraGeometryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(camera, 'camera rotation geometry is not implemented')

    def test_world_camera_composes_parent_before_child(self):
        # Reversing the noncommuting rotations changes the y sign in the result.
        h = math.sqrt(.5)
        state = {'imu': {'quat': [h, 0, 0, h]},
                 'frames': {'camera': {'quat': [h, h, 0, 0]}}}
        for actual, expected in zip(camera.world_camera_quat(state), [.5, .5, .5, .5]):
            self.assertAlmostEqual(actual, expected)

    def test_absent_legacy_pose_is_unavailable_not_a_made_up_identity(self):
        for state in (None, {}, {'imu': {}}, {'imu': {'quat': IDENTITY}, 'frames': {}}):
            with self.subTest(state=state):
                self.assertIsNone(camera.world_camera_quat(state))

    def test_present_malformed_pose_is_rejected(self):
        for quat in (None, [0, 0, 0, 0], [.94, 0, 0, 0], [1.06, 0, 0, 0],
                     [True, 0, 0, 0], [float('nan'), 0, 0, 0], [1, 0, 0], 'identity'):
            with self.subTest(quat=quat), self.assertRaises(ValueError):
                camera.world_camera_quat({'imu': {'quat': quat},
                                          'frames': {'camera': {'quat': IDENTITY}}})

    def test_malformed_present_pose_container_is_not_treated_as_legacy_absence(self):
        for container in (42, [], 'wrong'):
            for state in ({'imu': container, 'frames': {'camera': {'quat': IDENTITY}}},
                          {'imu': {'quat': IDENTITY}, 'frames': {'camera': container}}):
                with self.subTest(state=state), self.assertRaises(ValueError):
                    camera.world_camera_quat(state)

    def test_near_unit_boundaries_are_normalized_and_sign_is_equivalent(self):
        for scale in (.95, 1.05, -1.05):
            q = camera.world_camera_quat({'imu': {'quat': [scale, 0, 0, 0]},
                                          'frames': {'camera': {'quat': FORWARD}}})
            predicted = camera.predict_box([160, 300, 200, 340], FORWARD, q)
            for actual, expected in zip(predicted, [160, 300, 200, 340]):
                self.assertAlmostEqual(actual, expected)

    def test_positive_and_negative_world_yaw_move_previous_forward_target_opposite_view_turn(self):
        # Looking left makes a fixed forward target appear to the right, and vice versa.
        for angle in (-.2, .2):
            with self.subTest(angle=angle):
                box = camera.predict_box([160, 300, 200, 340], FORWARD, yaw_camera(angle))
                self.assertAlmostEqual((box[0]+box[2])/2, 180+FOCAL*math.tan(angle))
                self.assertAlmostEqual((box[1]+box[3])/2, 320)
                self.assertAlmostEqual(box[2]-box[0], 40)
                self.assertAlmostEqual(box[3]-box[1], 40)

    def test_identity_rotation_preserves_off_center_box(self):
        self.assertEqual(camera.predict_box([12, 120, 72, 160], IDENTITY, IDENTITY),
                         [12, 120, 72, 160])

    def test_previous_camera_pose_is_used_in_relative_rotation(self):
        # Only the relative yaw matters for a previously centered target.
        box = camera.predict_box([160, 300, 200, 340], yaw_camera(.2), yaw_camera(.4))
        self.assertAlmostEqual((box[0]+box[2])/2, 180+FOCAL*math.tan(.2))

    def test_behind_or_tangent_camera_ray_is_rejected(self):
        for angle in (math.pi/2, math.pi):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                camera.predict_box([160, 300, 200, 340], FORWARD, yaw_camera(angle))

    def test_nonfinite_box_or_invalid_quaternion_cannot_make_a_projection(self):
        for box, previous, current in (([0, 0, float('inf'), 10], IDENTITY, IDENTITY),
                                       ([0, 0, 0, 10], IDENTITY, IDENTITY),
                                       ([0, 0, 10, 10], [0, 0, 0, 0], IDENTITY),
                                       ([0, 0, 10, 10], IDENTITY, [1, 0, float('nan'), 0])):
            with self.subTest(box=box, previous=previous, current=current), self.assertRaises(ValueError):
                camera.predict_box(box, previous, current)

    def test_overflowing_box_dimensions_cannot_make_a_nonfinite_projection_box(self):
        with self.assertRaises(ValueError):
            camera.predict_box([-1e308, 0, 1e308, 10], IDENTITY, IDENTITY)


if __name__ == '__main__':
    unittest.main()
