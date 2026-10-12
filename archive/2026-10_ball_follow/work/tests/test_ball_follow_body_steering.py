"""Body steering geometry and safety, independent of camera or robot devices."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ball_follow_camera as camera
from ball_follow_control import FollowControl

FOCAL = 434.55844122715706
FORWARD = [.5, -.5, .5, -.5]  # CV right/down/forward → trunk -y/-z/+x.
IDENTITY = [1., 0., 0., 0.]


def trunk_quat(yaw=0., pitch=0., roll=0.):
    c, s, d, t, a, b = (math.cos(yaw/2), math.sin(yaw/2), math.cos(pitch/2),
                        math.sin(pitch/2), math.cos(roll/2), math.sin(roll/2))
    return [c*d*a+s*t*b, c*d*b-s*t*a, c*t*a+s*d*b, s*d*a-c*t*b]


def pose_and_box(yaw=0., pitch=0., roll=0., target=(1., 0., -.15), width=40.):
    """Independent Euler matrix projects a fixed body ray into a moving head camera."""
    cy, sy, cp, sp, cr, sr = (math.cos(yaw), math.sin(yaw), math.cos(pitch),
                              math.sin(pitch), math.cos(roll), math.sin(roll))
    rotation = ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
                (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr),
                (-sp, cp*sr, cp*cr))
    u, v, w = [sum(rotation[row][column]*target[row] for row in range(3))
               for column in range(3)]
    # The neutral camera's transpose maps (u,v,w) to CV (-v,-w,u).
    cx, image_y = 180-FOCAL*v/u, 320-FOCAL*w/u
    c, s, d, t, a, b = (math.cos(yaw/2), math.sin(yaw/2), math.cos(pitch/2),
                        math.sin(pitch/2), math.cos(roll/2), math.sin(roll/2))
    qw, qx, qy, qz = c*d*a+s*t*b, c*d*b-s*t*a, c*t*a+s*d*b, s*d*a-c*t*b
    quat = [.5*(qw+qx-qy+qz), .5*(-qw+qx-qy-qz),
            .5*(qw+qx+qy-qz), .5*(-qw+qx+qy+qz)]
    return quat, [cx-width/2, image_y-width/2, cx+width/2, image_y+width/2]


def target(box=None, class_id=32, confidence=.5):
    return {'class_id': class_id, 'confidence': confidence,
            'xyxy': list(box or [160., 300., 200., 340.])}


class BodyGeometryTests(unittest.TestCase):
    def helper(self, box, quat, trunk=IDENTITY):
        self.assertTrue(hasattr(camera, 'body_target_error'), 'body geometry helper is missing')
        return camera.body_target_error(box, quat, trunk)

    def test_body_roll_pitch_does_not_mix_downward_target_into_left_right_bearing(self):
        fixed = (1., .06, -.3)
        for pitch, roll in ((.2, .3), (-.2, -.3), (.1, -.25)):
            with self.subTest(pitch=pitch, roll=roll):
                quat, box = pose_and_box(0, pitch, roll, target=fixed)
                error, bearing = self.helper(box, quat, trunk_quat(0, pitch, roll))
                self.assertAlmostEqual(error, -FOCAL*.06/180)
                self.assertAlmostEqual(bearing, math.atan2(.06, 1))

    def test_rotating_target_and_body_heading_together_keeps_horizontal_error(self):
        for yaw in (-.4, .4):
            fixed = (math.cos(yaw)-.06*math.sin(yaw),
                     math.sin(yaw)+.06*math.cos(yaw), -.15)
            quat, box = pose_and_box(yaw+.1, .2, -.15, target=fixed)
            error, bearing = self.helper(box, quat, trunk_quat(yaw, .2, -.15))
            self.assertAlmostEqual(error, -FOCAL*.06/180)
            self.assertAlmostEqual(bearing, math.atan2(.06, 1))

    def test_vertical_trunk_heading_and_invalid_trunk_quaternion_are_rejected(self):
        for trunk in ([math.sqrt(.5), 0, math.sqrt(.5), 0],
                      trunk_quat(pitch=math.pi/2-1e-8), [0, 0, 0, 0],
                      [1, float('nan'), 0, 0]):
            with self.subTest(trunk=trunk), self.assertRaises(ValueError):
                self.helper([160, 300, 200, 340], FORWARD, trunk)

    def test_near_vertical_target_ray_has_no_reliable_horizontal_bearing(self):
        quat, _ = pose_and_box(pitch=-math.pi/2+1e-8)
        # Its forward horizontal component is positive, above the along-only epsilon.
        with self.assertRaises(ValueError):
            self.helper([160, 300, 200, 340], quat)

    def test_same_body_target_is_invariant_to_head_yaw_pitch_and_roll(self):
        # A fixed off-axis target has one body bearing even as its image moves in both axes.
        fixed = (1., .06, -.15)
        for angles in ((0, 0, 0), (.2, .1, -.15), (-.2, -.1, .15), (.1, -.15, -.2)):
            with self.subTest(angles=angles):
                quat, box = pose_and_box(*angles, target=fixed)
                error, bearing = self.helper(box, quat)
                self.assertAlmostEqual(error, -FOCAL*.06/180)
                self.assertAlmostEqual(bearing, math.atan2(.06, 1))

    def test_forward_target_keeps_zero_error_across_opposite_head_yaw(self):
        for yaw in (-.2, .2):
            quat, box = pose_and_box(yaw)
            self.assertAlmostEqual(self.helper(box, quat)[0], 0)
            self.assertAlmostEqual(self.helper(box, quat)[1], 0)

    def test_neutral_image_right_is_body_right_and_left_is_body_left(self):
        for center, sign in ((220., -1), (140., 1)):
            error, bearing = self.helper([center-20, 300, center+20, 340], FORWARD)
            self.assertAlmostEqual(error, (center-180)/180)
            self.assertAlmostEqual(bearing, math.atan2(sign*40/FOCAL, 1))

    def test_near_unit_quaternion_and_opposite_sign_keep_same_result(self):
        expected = self.helper([200, 300, 240, 340], FORWARD)
        for scale in (.95, 1.05, -1.):
            actual = self.helper([200, 300, 240, 340], [scale*x for x in FORWARD])
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b)

    def test_nonfinite_malformed_box_and_quaternion_are_rejected(self):
        for box, quat in (([1, 2, 1, 3], FORWARD), ([0, 0, 2, float('nan')], FORWARD),
                          ([0, 0, 2, True], FORWARD), ([0, 0, 2], FORWARD),
                          ([160, 300, 200, 340], [0, 0, 0, 0]),
                          ([160, 300, 200, 340], [1, 0, float('inf'), 0]),
                          ([1e308, 0, 1.5e308, 40], FORWARD)):
            with self.subTest(box=box, quat=quat), self.assertRaises(ValueError):
                self.helper(box, quat)

    def test_target_behind_or_tangent_to_body_forward_axis_is_rejected(self):
        for yaw in (math.pi/2, math.pi):
            # Center ray is along camera axis; turn that axis behind the trunk.
            c, s = math.cos(yaw/2), math.sin(yaw/2)
            quat = [.5*(c+s), -.5*(c+s), .5*(c-s), .5*(-c+s)]
            with self.subTest(yaw=yaw), self.assertRaises(ValueError):
                self.helper([160, 300, 200, 340], quat)


class BodySteeringTests(unittest.TestCase):
    def follow(self, **kwargs):
        try:
            return FollowControl(100, stop_area=.06, orange_continuation=True,
                                 camera_motion_compensation=True, body_steering=True, **kwargs)
        except TypeError as error:
            self.fail(f'opt-in body steering is unavailable: {error}')

    def observe(self, follow, index, detections=None, body=IDENTITY, world=FORWARD, now=None):
        now = 100+index*.05 if now is None else now
        return follow.observe([target()] if detections is None else detections,
                              now, now, index, camera_quat=world,
                              trunk_quat=body)

    def prime(self, follow, box=None, body=IDENTITY, world=FORWARD):
        for index in range(1, 6):
            result = self.observe(follow, index, [target(box)], body, world)
            self.assertEqual(result['vx'], .1 if index == 5 else 0)
        return result

    def test_constructor_requires_explicit_boolean_and_camera_compensation(self):
        for flag in (1, 0, 'true', None, []):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                FollowControl(100, body_steering=flag)
        with self.assertRaises(ValueError):
            FollowControl(100, orange_continuation=True, body_steering=True)
        FollowControl(100, body_steering=False)

    def test_five_full_frames_are_required_and_metadata_separates_the_two_errors(self):
        follow = self.follow()
        quat, box = pose_and_box(.2)
        result = self.prime(follow, box, IDENTITY, quat)
        self.assertEqual(result.get('body_steering'), True)
        self.assertAlmostEqual(result['image_error'], (sum((box[0], box[2]))/2-180)/180)
        self.assertAlmostEqual(result['body_bearing_rad'], 0)
        self.assertAlmostEqual(result['e'], 0)
        self.assertEqual(result['vyaw'], 0)

    def test_fixed_target_motion_does_not_make_the_body_chase_the_head(self):
        follow = self.follow()
        q0, b0 = pose_and_box(-.2, target=(1, .04, -.1))
        self.prime(follow, b0, IDENTITY, q0)
        for index, angles in enumerate(((.2, .1, -.15), (-.1, -.1, .2)), 6):
            quat, box = pose_and_box(*angles, target=(1, .04, -.1))
            result = self.observe(follow, index, [target(box)], IDENTITY, quat)
            self.assertAlmostEqual(result['vyaw'], .6*FOCAL*.04/180)
            self.assertAlmostEqual(result['body_bearing_rad'], math.atan2(.04, 1))

    def test_true_left_right_turns_keep_original_gain_limit_and_deadband(self):
        for center, expected in ((140, .6*40/180), (220, -.6*40/180),
                                 (100, .2), (260, -.2), (185.4, 0)):
            with self.subTest(center=center):
                follow = self.follow()
                result = self.prime(follow, [center-20, 300, center+20, 340])
                self.assertAlmostEqual(result['vyaw'], expected)

    def test_disabled_mode_keeps_raw_steering_and_ignores_optional_body_pose(self):
        for explicit in (False,):
            follow = FollowControl(100, orange_continuation=True,
                                   camera_motion_compensation=True, body_steering=explicit)
            quat, box = pose_and_box(.2)
            for index in range(1, 6):
                now = 100+index*.05
                result = follow.observe([target(box)], now, now, index,
                                        camera_quat=quat, trunk_quat='ignored')
            self.assertEqual(result['vyaw'], -.2)
            self.assertNotIn('body_steering', result)
            self.assertNotIn('body_bearing_rad', result)
            self.assertNotIn('image_error', result)

    def test_low_confidence_size_jump_and_ambiguity_still_stop(self):
        cases = (([target(confidence=.249)], 'low_confidence'),
                 ([target([120, 260, 240, 380])], 'target_scale_jump'),
                 ([target(), target([170, 300, 210, 340], 49)], 'ambiguous_target'),
                 ([target(), target([220, 300, 260, 340])], 'multiple_balls'))
        for detections, reason in cases:
            with self.subTest(reason=reason):
                follow = self.follow()
                self.prime(follow)
                result = self.observe(follow, 6, detections)
                self.assertEqual(result['reason'], reason)
                self.assertEqual((result['vx'], result['vy'], result['vyaw']), (0, 0, 0))

    def test_large_clipped_box_holds_before_size_or_area_and_can_recover(self):
        follow = self.follow()
        self.prime(follow)
        result = self.observe(follow, 6, [target([0, 240, 160, 400])])
        self.assertEqual(result['mode'], 'holding')
        self.assertEqual(result['association'], 'clipped_hold')
        self.assertIsNone(result['image_error'])
        self.assertIsNone(result['body_bearing_rad'])
        self.assertIsNone(result['area'])
        self.assertEqual((result['vx'], result['vy'], result['vyaw']), (0, 0, 0))
        self.assertFalse(follow.reached)
        result = self.observe(follow, 7, [target(class_id=49)])
        self.assertEqual(result['mode'], 'following')
        self.assertEqual(result['vx'], .1)

    def test_startup_clipping_still_stops_immediately(self):
        result = self.observe(self.follow(), 1, [target([0, 300, 40, 340])])
        self.assertEqual(result['reason'], 'clipped_target')
        self.assertEqual(result['vx'], 0)

    def test_missing_body_pose_zero_holds_and_recovers_only_on_fresh_frame(self):
        follow = self.follow()
        self.prime(follow)
        result = self.observe(follow, 6, body=None)
        self.assertEqual(result['mode'], 'holding')
        self.assertEqual(result['association'], 'camera_pose_hold')
        self.assertIsNone(result['image_error'])
        self.assertIsNone(result['body_bearing_rad'])
        self.assertEqual((result['vx'], result['vy'], result['vyaw']), (0, 0, 0))
        duplicate = self.observe(follow, 6, now=100.4)
        self.assertEqual(duplicate['mode'], 'holding')
        recovered = self.observe(follow, 7, now=100.45)
        self.assertEqual(recovered['mode'], 'following')
        self.assertEqual(recovered['vx'], .1)

    def test_body_pose_missing_and_clipping_share_one_deadline(self):
        follow = self.follow()
        self.prime(follow)
        self.observe(follow, 6, body=None)
        self.observe(follow, 7, [target([0, 240, 160, 400])], now=100.4)
        self.observe(follow, 8, [], now=100.5)
        result = self.observe(follow, 9, now=100.6)
        self.assertEqual(result['reason'], 'camera_pose_unavailable')
        self.assertEqual(result['vx'], 0)

    def test_startup_missing_body_pose_does_not_count_as_ball_acquisition(self):
        follow = self.follow()
        self.observe(follow, 1, body=None)
        for index in range(2, 7):
            result = self.observe(follow, index)
            self.assertEqual(result['vx'], .1 if index == 6 else 0)

    def test_invalid_body_quaternion_is_a_controlled_latched_stop(self):
        for quat in ([0, 0, 0, 0], [True, 0, 0, 0], [1, 0, float('nan'), 0], 'bad'):
            with self.subTest(quat=quat):
                follow = self.follow()
                self.prime(follow)
                result = self.observe(follow, 6, body=quat)
                self.assertEqual(result['reason'], 'invalid_camera_pose')
                self.assertEqual(result['vx'], 0)
                self.assertEqual(self.observe(follow, 7)['vx'], 0)

    def test_nonforward_body_target_is_a_controlled_stop_before_acquiring_or_moving(self):
        for yaw in (math.pi/2, math.pi):
            follow = self.follow()
            result = self.observe(follow, 1, body=trunk_quat(yaw))
            self.assertEqual(result['reason'], 'invalid_target_direction')
            self.assertEqual(result['vx'], 0)
            self.assertIsNone(follow.baseline_area)

    def test_near_vertical_body_heading_is_controlled_invalid_direction(self):
        follow = self.follow()
        result = self.observe(follow, 1, body=trunk_quat(pitch=math.pi/2))
        self.assertEqual(result['reason'], 'invalid_target_direction')
        self.assertEqual(result['vx'], 0)

    def test_near_vertical_target_is_a_controlled_invalid_direction(self):
        follow = self.follow()
        quat, _ = pose_and_box(pitch=-math.pi/2+1e-8)
        result = self.observe(follow, 1, world=quat)
        self.assertEqual(result['reason'], 'invalid_target_direction')
        self.assertEqual(result['vx'], 0)

    def test_body_tilt_does_not_change_turn_command_for_a_fixed_world_target(self):
        follow = self.follow()
        camera_quat, box = pose_and_box(target=(1, .06, -.15))
        self.prime(follow, box, IDENTITY, camera_quat)
        for index, (pitch, roll) in enumerate(((.2, .3), (-.2, -.3)), 6):
            camera_quat, box = pose_and_box(0, pitch, roll, target=(1, .06, -.15))
            result = self.observe(follow, index, [target(box)],
                                  trunk_quat(0, pitch, roll), camera_quat)
            self.assertAlmostEqual(result['vyaw'], .6*FOCAL*.06/180)
            self.assertAlmostEqual(result['body_bearing_rad'], math.atan2(.06, 1))

    def test_raw_area_still_controls_fixed_stop_and_latches_before_motion(self):
        follow = self.follow()
        # 120x120 is exactly 6.25%; body steering must not transform box corners or area.
        result = self.observe(follow, 1, [target([120, 260, 240, 380])])
        self.assertEqual(result['reason'], 'target_area_reached')
        self.assertEqual(result['area'], .0625)
        self.assertEqual(result['vx'], 0)
        self.assertTrue(follow.reached)
        self.assertEqual(self.observe(follow, 2)['vx'], 0)


if __name__ == '__main__':
    unittest.main()
