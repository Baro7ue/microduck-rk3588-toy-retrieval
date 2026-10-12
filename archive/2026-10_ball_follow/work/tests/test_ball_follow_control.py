"""Behavioral tests for the bounded, fail-closed visual ball controller."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
SOURCE = WORK/'ball_follow_control.py'
control = None
if SOURCE.exists():
    spec = importlib.util.spec_from_file_location('ball_follow_control', SOURCE)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)


def ball(center=180, width=40, height=40, confidence=.5):
    return {'class_id': 32, 'class_name': 'sports ball', 'confidence': confidence,
            'xyxy': [center-width/2, 320-height/2, center+width/2, 320+height/2]}


def prime(instance, center=180):
    for i in range(1, 6):
        instance.observe([ball(center)], received_at=100+i*.05, now=100+i*.05, receiver_index=i)


class OrangeContinuationTests(unittest.TestCase):
    def follow(self):
        try:
            return control.FollowControl(100, stop_area=.1, orange_continuation=True)
        except TypeError as error:
            self.fail(f'optional orange continuation is unavailable: {error}')

    def test_enabled_continuation_still_requires_five_sports_ball_frames(self):
        # Enabling association must not bypass the five-frame readiness requirement.
        follow = self.follow()
        for index in range(1, 6):
            now = 100+index*.05
            result = follow.observe([ball()], now, now, index)
            self.assertEqual(result['vx'], .1 if index == 5 else 0)

    def test_nonboolean_continuation_flag_cannot_enable_extra_target_classes(self):
        # Truthy strings or integers must not silently opt into class-49 association.
        for value in (1, 0, 'true', 'false', None, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                control.FollowControl(100, orange_continuation=value)

    @staticmethod
    def orange(center=180, width=40, height=40, confidence=.5):
        target = ball(center, width, height, confidence)
        target.update(class_id=49, class_name='orange')
        return target

    def test_orange_alone_never_acquires_or_authorizes_motion(self):
        # Treating class 49 as an acquisition class would follow any nearby orange.
        for acquired_frames in (0, 4):
            with self.subTest(acquired_frames=acquired_frames):
                follow = self.follow()
                for index in range(1, acquired_frames+1):
                    now = 100+index*.05
                    follow.observe([ball()], now, now, index)
                result = follow.observe([self.orange(width=200, height=200)], 100.25, 100.25, 5)
                self.assertEqual(result['reason'], 'missing_ball')
                self.assertEqual(result['vx'], 0)
                self.assertFalse(follow.reached)
                self.assertIsNone(follow.baseline_area)

    def test_confirmed_ball_can_change_to_orange_and_back_without_relabeling(self):
        # Strict class-32 filtering would lose the same spatially continuous target.
        follow = self.follow()
        prime(follow)
        result = follow.observe([self.orange(center=190, width=44, height=44)], 100.3, 100.3, 6)
        self.assertEqual(result['vx'], .1)
        self.assertIsNone(result['reason'])
        self.assertEqual(result.get('source_class_id'), 49)
        self.assertEqual(result.get('association'), 'spatial_continuation')
        result = follow.observe([ball(center=195, width=48, height=48)], 100.35, 100.35, 7)
        self.assertEqual(result['vx'], .1)
        self.assertEqual(result.get('source_class_id'), 32)

    def test_measured_gait_camera_jump_continues_the_confirmed_ball(self):
        # Head motion can move a far ball by 40 pixels between consecutive fresh frames.
        follow = self.follow()
        first = ball()
        first['xyxy'] = [152.18, 151.41, 187.10, 189.22]
        for index in range(1, 6):
            now = 100+index*.05
            follow.observe([first], now, now, index)
        fixtures = [
            (6, 100.3, 32, [155.85, 174.52, 190.19, 211.96]),
            (7, 100.36, 49, [157.64, 214.98, 191.01, 251.16]),
            (9, 100.42, 32, [162.79364, 230.92993, 195.04047, 264.97284]),
            (10, 100.48, 32, [196.47226, 206.40950, 231.34543, 242.81071]),
        ]
        for index, now, class_id, box in fixtures:
            with self.subTest(index=index):
                target = {'class_id': class_id, 'confidence': .5, 'xyxy': box}
                result = follow.observe([target], now, now, index)
                self.assertEqual(result['vx'], .1)
                self.assertIsNone(result['reason'])
                self.assertEqual(result.get('source_class_id'), class_id)

    def test_nonoverlapping_center_jump_must_stay_within_target_scaled_window(self):
        # A fixed/global search window would associate remote oranges with the tracked ball.
        for box in ([160, 361, 200, 401], [221, 300, 261, 340]):
            with self.subTest(box=box):
                follow = self.follow()
                prime(follow)
                target = self.orange()
                target['xyxy'] = box
                result = follow.observe([target], 100.3, 100.3, 6)
                self.assertEqual(result['reason'], 'target_jump')
                self.assertEqual(result['vx'], 0)

    def test_measured_large_camera_jump_preserves_unique_sports_ball_tracking(self):
        # Pixel camera motion is independent of far-ball size; class 32 remains trusted.
        follow = self.follow()
        first = ball()
        first['xyxy'] = [166.54400634765625, 204.09927368164062,
                         201.79888916015625, 241.05807495117188]
        for index in range(1, 6):
            now = 100+index*.05
            follow.observe([first], now, now, index)
        target = ball()
        target['xyxy'] = [235.90670776367188, 186.9763641357422,
                          272.7321472167969, 226.8302459716797]
        result = follow.observe([target], 100.33, 100.33, 6)
        self.assertEqual(result['vx'], .1)
        self.assertIsNone(result['reason'])
        self.assertEqual(result.get('source_class_id'), 32)
        self.assertEqual(result.get('association'), 'sports_ball_detection')

    def test_multiple_sports_balls_stop_even_when_only_one_is_spatially_local(self):
        # Spatial filtering must not silently discard a second semantic ball candidate.
        follow = self.follow()
        prime(follow)
        result = follow.observe([ball(), ball(center=300)], 100.3, 100.3, 6)
        self.assertEqual(result['vx'], 0)
        self.assertEqual(result['reason'], 'multiple_balls')

    def test_local_orange_and_distinct_sports_ball_stop_as_ambiguous(self):
        # A semantic ball and a separate plausible continuation cannot be resolved safely.
        follow = self.follow()
        prime(follow)
        result = follow.observe([self.orange(), ball(center=300)], 100.3, 100.3, 6)
        self.assertEqual(result['vx'], 0)
        self.assertEqual(result['reason'], 'ambiguous_target')

    def test_large_centered_object_cannot_bypass_size_jump_guard(self):
        # A center gate scaled by the candidate size must not admit a much larger object.
        follow = self.follow()
        prime(follow)
        result = follow.observe([self.orange(width=200, height=200)], 100.3, 100.3, 6)
        self.assertEqual(result['reason'], 'target_scale_jump')
        self.assertEqual(result['vx'], 0)

    def test_remote_orange_latches_without_switching_target(self):
        # Class 49 is only a spatial continuation, never a freely selectable target.
        follow = self.follow()
        prime(follow)
        result = follow.observe([self.orange(center=300)], 100.3, 100.3, 6)
        self.assertEqual(result['reason'], 'target_jump')
        self.assertEqual(result['vx'], 0)
        follow.observe([self.orange()], 100.35, 100.35, 7)
        self.assertEqual(follow.command(100.36), (0, 0, 0))

    def test_remote_orange_does_not_make_one_local_match_ambiguous(self):
        # Counting every 32/49 box instead of spatial matches would stop on distant objects.
        follow = self.follow()
        prime(follow)
        result = follow.observe([self.orange(center=190), self.orange(center=300)], 100.3, 100.3, 6)
        self.assertEqual(result['vx'], .1)
        self.assertEqual(result.get('source_class_id'), 49)
        self.assertIsNone(result['reason'])

    def test_two_overlapping_class_candidates_latch_as_ambiguous(self):
        # Choosing the higher-confidence match hides ambiguous object identity.
        follow = self.follow()
        prime(follow)
        result = follow.observe([ball(center=175), self.orange(center=185)], 100.3, 100.3, 6)
        self.assertEqual(result['reason'], 'ambiguous_target')
        self.assertEqual(result['vx'], 0)

    def test_same_box_class_32_49_duplicate_pair_counts_as_one_confirmed_target(self):
        # A class-boundary duplicate must not manufacture two distinct tracked objects.
        follow = self.follow()
        prime(follow)
        result = follow.observe([self.orange(center=181, confidence=.8), ball(confidence=.5)],
                                100.3, 100.3, 6)
        self.assertEqual(result['vx'], .1)
        self.assertIsNone(result['reason'])
        self.assertEqual(result.get('source_class_id'), 32)
        self.assertEqual(result.get('association'), 'sports_ball_detection')

    def test_area_ratio_rejects_large_or_small_same_center_jumps(self):
        # An overlapping box alone must not associate a size jump to the confirmed ball.
        for size in (20, 80):
            with self.subTest(size=size):
                follow = self.follow()
                prime(follow)
                result = follow.observe([self.orange(width=size, height=size)], 100.3, 100.3, 6)
                self.assertEqual(result['reason'], 'target_scale_jump')
                self.assertEqual(result['vx'], 0)

    def test_low_confidence_continuation_still_stops_immediately(self):
        # A weak class switch remains invalid even when an absence hold is available.
        follow = self.follow()
        prime(follow)
        result = follow.observe([self.orange(confidence=.249)], 100.3, 100.3, 6)
        self.assertEqual(result['reason'], 'low_confidence')
        self.assertEqual(result['vx'], 0)

    def test_clipped_target_holds_zero_and_recovers_from_last_full_box(self):
        # A partial box must neither command motion nor replace the association reference.
        follow = self.follow()
        prime(follow, center=40)
        result = follow.observe([self.orange(center=21)], 100.3, 100.3, 6)
        self.assertEqual((result['vx'], result['vy'], result['vyaw']), (0, 0, 0))
        self.assertIsNone(result['reason'])
        self.assertEqual(result['mode'], 'holding')
        self.assertEqual(result.get('association'), 'clipped_hold')
        self.assertEqual(result.get('source_class_id'), 49)
        self.assertIsNone(result['area'])
        # Center 90 is within 60px of the last FULL center40, but not clipped center21.
        result = follow.observe([self.orange(center=90)], 100.4, 100.4, 7)
        self.assertEqual(result['vx'], .1)
        self.assertIsNone(result['reason'])

    def test_partial_box_size_never_reaches_threshold_or_replaces_full_size_reference(self):
        # A huge partial box cannot be used for distance or size-jump decisions.
        for class_id in (32, 49):
            with self.subTest(class_id=class_id):
                follow = self.follow()
                prime(follow)
                partial = {'class_id': class_id, 'confidence': .5,
                           'xyxy': [0, 250, 300, 390]}
                result = follow.observe([partial], 100.3, 100.3, 6)
                self.assertEqual(result['vx'], 0)
                self.assertIsNone(result['reason'])
                self.assertIsNone(result['area'])
                self.assertEqual(result.get('source_class_id'), class_id)
                self.assertEqual(result.get('association'), 'clipped_hold')
                self.assertFalse(follow.reached)
                result = follow.observe([self.orange()], 100.4, 100.4, 7)
                self.assertEqual(result['vx'], .1)
                self.assertIsNone(result['reason'])

    def test_complete_box_margin_boundary_stays_usable(self):
        # The two-pixel complete margin is inclusive; a box inside it must remain usable.
        follow = self.follow()
        prime(follow, center=40)
        result = follow.observe([self.orange(center=22)], 100.3, 100.3, 6)
        self.assertEqual(result['vx'], .1)
        self.assertIsNone(result['reason'])

    def test_startup_clipping_still_latches_before_any_acquisition_or_threshold(self):
        # Absence recovery is available only after five full sports-ball observations.
        for startup_count in (0, 4):
            with self.subTest(startup_count=startup_count):
                follow = self.follow()
                for index in range(1, startup_count+1):
                    now = 100+index*.05
                    follow.observe([ball()], now, now, index)
                partial = {'class_id': 32, 'confidence': .5, 'xyxy': [0, 0, 360, 640]}
                result = follow.observe([partial], 100.25, 100.25, 5)
                self.assertEqual(result['vx'], 0)
                self.assertEqual(result['reason'], 'clipped_target')
                self.assertFalse(follow.reached)
                self.assertIsNone(follow.baseline_area)

    def test_clipped_absence_deadline_latches_and_never_resumes(self):
        # Continued partial frames cannot extend the first clipping deadline.
        follow = self.follow()
        prime(follow, center=40)
        follow.observe([self.orange(center=21)], 100.3, 100.3, 6)
        follow.observe([self.orange(center=21)], 100.5, 100.5, 7)
        self.assertEqual(follow.command(100.599), (0, 0, 0))
        self.assertIsNone(follow.reason)
        self.assertEqual(follow.command(100.6), (0, 0, 0))
        self.assertEqual(follow.reason, 'clipped_target')
        result = follow.observe([ball(center=40)], 100.61, 100.61, 8)
        self.assertEqual(result['vx'], 0)
        self.assertEqual(result['reason'], 'clipped_target')

    def test_duplicate_full_frame_cannot_clear_clipped_hold_or_extend_deadline(self):
        # A full detection on a repeated receiver index is not fresh recovery.
        follow = self.follow()
        prime(follow, center=40)
        follow.observe([self.orange(center=21)], 100.3, 100.3, 6)
        result = follow.observe([ball(center=40)], 100.5, 100.5, 6)
        self.assertEqual(result['vx'], 0)
        self.assertIsNone(result['reason'])
        self.assertEqual(follow.last_frame_at, 100.3)
        self.assertEqual(follow.last_frame_index, 6)
        follow.command(100.6)
        self.assertEqual(follow.reason, 'clipped_target')

    def test_mixed_empty_and_clipped_frames_share_first_absence_deadline_and_reason(self):
        # Switching absence type must not refresh time or hide the initiating reason.
        for first_clipped, reason in ((True, 'clipped_target'), (False, 'missing_ball')):
            with self.subTest(first_clipped=first_clipped):
                follow = self.follow()
                prime(follow, center=40)
                clipped = [self.orange(center=21)]
                first, second = (clipped, []) if first_clipped else ([], clipped)
                follow.observe(first, 100.3, 100.3, 6)
                result = follow.observe(second, 100.5, 100.5, 7)
                self.assertEqual(result['vx'], 0)
                self.assertIsNone(result['reason'])
                follow.command(100.6)
                self.assertEqual(follow.reason, reason)

    def test_absence_hold_uses_shorter_configured_feedback_age_budget(self):
        # The hold must never outlive a stricter configured feedback freshness limit.
        follow = control.FollowControl(100, max_age=.15, stop_area=.1,
                                       orange_continuation=True)
        prime(follow)
        follow.observe([], 100.3, 100.3, 6)
        self.assertEqual(follow.command(100.449), (0, 0, 0))
        self.assertIsNone(follow.reason)
        follow.command(100.45)
        self.assertEqual(follow.reason, 'missing_ball')

    def test_empty_frame_immediately_holds_zero_and_fresh_match_resumes_within_grace(self):
        # Retaining the old forward command while waiting would continue blind motion.
        follow = self.follow()
        prime(follow)
        result = follow.observe([], 100.3, 100.3, 6)
        self.assertEqual((result['vx'], result['vy'], result['vyaw']), (0, 0, 0))
        self.assertEqual(result['mode'], 'holding')
        self.assertEqual(result.get('association'), 'missing_hold')
        self.assertIsNone(result.get('source_class_id'))
        self.assertIsNone(result['reason'])
        self.assertEqual(follow.command(100.399), (0, 0, 0))
        result = follow.observe([self.orange(center=190)], 100.4, 100.4, 7)
        self.assertEqual(result['vx'], .1)
        self.assertIsNone(result['reason'])
        self.assertEqual(result.get('source_class_id'), 49)

    def test_missing_grace_expires_at_feedback_age_boundary_and_latches(self):
        # Extending the grace on every empty frame would keep an untracked target alive.
        follow = self.follow()
        prime(follow)
        follow.observe([], 100.3, 100.3, 6)
        follow.observe([], 100.4, 100.4, 7)
        self.assertEqual(follow.command(100.599), (0, 0, 0))
        self.assertIsNone(follow.reason)
        self.assertEqual(follow.command(100.6), (0, 0, 0))
        self.assertEqual(follow.reason, 'missing_ball')
        follow.observe([self.orange()], 100.61, 100.61, 8)
        self.assertEqual(follow.command(100.62), (0, 0, 0))

    def test_duplicate_match_does_not_clear_missing_hold_or_extend_deadline(self):
        # A repeated receiver index must not count as fresh reacquisition.
        follow = self.follow()
        prime(follow)
        follow.observe([], 100.3, 100.3, 6)
        result = follow.observe([self.orange()], 100.4, 100.4, 6)
        self.assertEqual(result['vx'], 0)
        self.assertIsNone(result['reason'])
        self.assertEqual(follow.last_frame_at, 100.3)
        self.assertEqual(follow.last_frame_index, 6)
        follow.command(100.6)
        self.assertEqual(follow.reason, 'missing_ball')

    def test_fresh_match_at_grace_deadline_cannot_conceal_timeout(self):
        # Checking a new candidate first would let late feedback erase the expired hold.
        follow = self.follow()
        prime(follow)
        follow.observe([], 100.3, 100.3, 6)
        result = follow.observe([self.orange()], 100.6, 100.6, 7)
        self.assertEqual(result['vx'], 0)
        self.assertEqual(result['reason'], 'missing_ball')

    def test_spatial_continuation_stops_at_fixed_area_before_issuing_motion(self):
        # Accepting a valid class switch must still latch the visual stop threshold.
        follow = control.FollowControl(100, stop_area=.012, orange_continuation=True)
        prime(follow)
        for index, size in ((6, 50), (7, 60)):
            now = 100+index*.05
            result = follow.observe([self.orange(width=size)], now, now, index)
            self.assertEqual(result['vx'], .1)
        result = follow.observe([self.orange(width=69.12)], 100.4, 100.4, 8)
        self.assertEqual(result['reason'], 'target_area_reached')
        self.assertEqual(result['vx'], 0)
        self.assertEqual(result.get('source_class_id'), 49)
        self.assertTrue(follow.reached)
        follow.observe([ball()], 100.45, 100.45, 9)
        self.assertEqual(follow.command(100.46), (0, 0, 0))

    def test_future_and_stale_frames_still_stop_during_missing_hold(self):
        # Grace only tolerates fresh absent detections, never invalid feedback time.
        for received, reason in ((100.41, 'future_frame'), (100, 'stale_detection')):
            with self.subTest(reason=reason):
                follow = self.follow()
                prime(follow)
                follow.observe([], 100.3, 100.3, 6)
                result = follow.observe([self.orange()], received, 100.4, 7)
                self.assertEqual(result['vx'], 0)
                self.assertEqual(result['reason'], reason)


class BallFollowTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(control, 'the fail-closed ball controller has not been implemented')

    def test_waits_for_five_fresh_frames_and_keeps_median_only_as_diagnostic(self):
        # Startup still needs five frames, but its median must not change the stop threshold.
        follow = control.FollowControl(100)
        self.assertEqual(follow.target_area, .012)
        for i, size in enumerate([20, 40, 50, 45, 30], 1):
            result = follow.observe([ball(width=size, height=size)], 100+i*.05, 100+i*.05, i)
            self.assertEqual(result['stop_area'], .012)
            if i < 5:
                self.assertEqual((result['vx'], result['vy'], result['vyaw']), (0, 0, 0))
                self.assertIsNone(follow.baseline_area)
                self.assertIsNone(follow.moving_started_at)
        self.assertEqual(follow.command(100.26), (.1, 0, 0))
        self.assertAlmostEqual(follow.baseline_area, 1/144)
        self.assertEqual(follow.moving_started_at, 100.25)
        self.assertEqual(follow.last_frame_at, 100.25)
        self.assertEqual(follow.last_frame_index, 5)
        self.assertAlmostEqual(result['start_area'], 1/144)
        self.assertEqual(result['stop_area'], .012)
        self.assertIsNone(result['reason'])

    def test_left_and_right_visual_error_produce_correct_turn_sign(self):
        # Reversing camera error or yaw sign steers away from the target.
        for center, want_error, want_yaw in [(144, -.2, .12), (216, .2, -.12)]:
            with self.subTest(center=center):
                follow = control.FollowControl(100)
                prime(follow)
                result = follow.observe([ball(center)], 100.3, 100.3, 6)
                self.assertAlmostEqual(result['e'], want_error)
                self.assertAlmostEqual(result['vyaw'], want_yaw)
                self.assertEqual((result['vx'], result['vy']), (.1, 0))

    def test_yaw_saturates_and_deadband_includes_its_boundary(self):
        # Removing the clamp or mishandling the inclusive deadband creates unwanted turns.
        for center, want_yaw in [(20, .2), (340, -.2), (174.6, 0), (185.4, 0)]:
            with self.subTest(center=center):
                follow = control.FollowControl(100)
                prime(follow)
                result = follow.observe([ball(center)], 100.3, 100.3, 6)
                self.assertEqual(result['vyaw'], want_yaw)

    def test_fixed_default_threshold_stops_and_latches_after_target_shrinks(self):
        # The default threshold must be 1.2% of the image, regardless of startup area.
        follow = control.FollowControl(100)
        prime(follow)
        below = follow.observe([ball(width=69.119)], 100.3, 100.3, 6)
        self.assertEqual((below['vx'], below['vy'], below['vyaw']), (.1, 0, 0))
        result = follow.observe([ball(width=69.12)], 100.35, 100.35, 7)
        self.assertEqual(result['area'], .012)
        self.assertEqual((result['vx'], result['vy'], result['vyaw']), (0, 0, 0))
        self.assertEqual(result['reason'], 'target_area_reached')
        self.assertTrue(follow.reached)
        follow.observe([ball(width=10, height=10)], 100.4, 100.4, 8)
        self.assertEqual(follow.command(100.41), (0, 0, 0))
        self.assertEqual(follow.reason, 'target_area_reached')
        self.assertAlmostEqual(follow.baseline_area, 1/144)

    def test_different_initial_sizes_reach_the_same_fixed_area_threshold(self):
        # Initial-area multiples would stop at different image sizes for these two starts.
        for initial_size, baseline in [(20, 1/576), (50, 25/2304)]:
            with self.subTest(initial_size=initial_size):
                follow = control.FollowControl(100, speed=.3, stop_area=.012)
                self.assertEqual(follow.target_area, .012)
                for index in range(1, 6):
                    now = 100+index*.05
                    follow.observe([ball(width=initial_size, height=initial_size)], now, now, index)
                self.assertAlmostEqual(follow.baseline_area, baseline)
                below = follow.observe([ball(width=69.119)], 100.3, 100.3, 6)
                self.assertEqual((below['vx'], below['vy'], below['vyaw']), (.3, 0, 0))
                self.assertIsNone(below['reason'])
                at = follow.observe([ball(width=69.12)], 100.35, 100.35, 7)
                self.assertEqual(at['area'], .012)
                self.assertEqual(at['stop_area'], .012)
                self.assertEqual((at['vx'], at['vy'], at['vyaw']), (0, 0, 0))
                self.assertEqual(at['reason'], 'target_area_reached')
                self.assertTrue(follow.reached)
                follow.observe([ball(width=10, height=10)], 100.4, 100.4, 8)
                self.assertEqual(follow.command(100.41), (0, 0, 0))
                self.assertAlmostEqual(follow.baseline_area, baseline)
                self.assertEqual(follow.target_area, .012)

    def test_first_valid_frame_at_or_above_threshold_stops_without_startup_motion(self):
        # Waiting for the fifth frame could command motion toward an already-near ball.
        for width in (69.12, 69.121, 80):
            with self.subTest(width=width):
                follow = control.FollowControl(100, stop_area=.012)
                self.assertEqual(follow.command(100.01), (0, 0, 0))
                result = follow.observe([ball(width=width)], 100.05, 100.05, 1)
                self.assertEqual((result['vx'], result['vy'], result['vyaw']), (0, 0, 0))
                self.assertEqual(result['reason'], 'target_area_reached')
                self.assertTrue(follow.reached)
                self.assertIsNone(follow.baseline_area)
                self.assertIsNone(follow.moving_started_at)
                prime(follow)
                self.assertEqual(follow.command(100.26), (0, 0, 0))
                self.assertEqual(follow.reason, 'target_area_reached')

    def test_custom_fixed_threshold_controls_stop_instead_of_the_default(self):
        # Ignoring stop_area or using the startup median would stop at the wrong boundary.
        follow = control.FollowControl(100, stop_area=.02)
        prime(follow)
        below = follow.observe([ball(width=36, height=120)], 100.3, 100.3, 6)
        self.assertEqual((below['vx'], below['vy'], below['vyaw']), (.1, 0, 0))
        result = follow.observe([ball(width=36, height=128)], 100.35, 100.35, 7)
        self.assertEqual(result['area'], .02)
        self.assertEqual(result['stop_area'], .02)
        self.assertEqual(result['reason'], 'target_area_reached')
        self.assertEqual((result['vx'], result['vy'], result['vyaw']), (0, 0, 0))

    def test_invalid_fixed_threshold_is_rejected(self):
        # An invalid normalized image area must never authorize a movement trial.
        for threshold in [0, -.012, 1.000001, float('nan'), float('inf'), True, '.012', None]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                control.FollowControl(100, stop_area=threshold)

    def test_full_image_threshold_is_valid_and_stops_at_its_boundary(self):
        follow = control.FollowControl(100, stop_area=1)
        target = ball()
        target['xyxy'] = [0, 0, 360, 640]
        result = follow.observe([target], 100.05, 100.05, 1)
        self.assertEqual(result['area'], 1)
        self.assertEqual(result['reason'], 'target_area_reached')
        self.assertEqual(follow.command(100.06), (0, 0, 0))

    def test_missing_and_multiple_balls_latch_even_during_startup(self):
        # Treating no ball as startup waiting allows a later frame to bypass the approved stop rule.
        for detections in [[], [ball(), ball(220)]]:
            with self.subTest(detections=detections):
                follow = control.FollowControl(100)
                follow.observe(detections, 100.01, 100.01, 1)
                self.assertIsNotNone(follow.reason)
                prime(follow)
                self.assertEqual(follow.command(100.26), (0, 0, 0))

    def test_low_confidence_and_malformed_boxes_fail_closed(self):
        # NaN comparisons otherwise silently pass; invalid geometry must never generate movement.
        malformed = [ball(confidence=.249), ball(confidence=float('nan')), ball(confidence=1.1),
                     {'class_id':32,'confidence':.5,'xyxy':[float('nan'),300,200,340]},
                     {'class_id':32,'confidence':.5,'xyxy':[160,300,float('inf'),340]},
                     {'class_id':32,'confidence':.5,'xyxy':[-.01,300,200,340]},
                     {'class_id':32,'confidence':.5,'xyxy':[160,300,360.01,340]},
                     {'class_id':32,'confidence':.5,'xyxy':[160,-.01,200,340]},
                     {'class_id':32,'confidence':.5,'xyxy':[160,300,200,640.01]},
                     {'class_id':32,'confidence':.5,'xyxy':[200,300,200,340]},
                     {'class_id':32,'confidence':.5}, {'class_id':True,'confidence':.5,'xyxy':[160,300,200,340]}]
        for bad in malformed:
            with self.subTest(bad=bad):
                follow = control.FollowControl(100)
                prime(follow)
                follow.observe([bad], 100.3, 100.3, 6)
                self.assertIsNotNone(follow.reason)
                self.assertEqual(follow.command(100.31), (0, 0, 0))

    def test_confidence_threshold_inclusive_and_other_classes_do_not_count_as_balls(self):
        # Filtering on >0.25 or counting a valid person as a second ball breaks the contract.
        follow = control.FollowControl(100)
        person = {'class_id':0,'confidence':.9,'xyxy':[10,10,50,100]}
        for i in range(1, 6):
            follow.observe([person, ball(confidence=.25)], 100+i*.05, 100+i*.05, i)
        self.assertEqual(follow.command(100.26), (.1, 0, 0))

    def test_duplicate_or_reversed_index_cannot_extend_feedback_lifetime(self):
        # Accepting a repeat as fresh keeps stale control alive indefinitely.
        for index in [5, 4]:
            with self.subTest(index=index):
                follow = control.FollowControl(100)
                prime(follow)
                follow.observe([ball(216)], 100.45, 100.45, index)
                self.assertEqual(follow.last_frame_at, 100.25)
                self.assertEqual(follow.last_frame_index, 5)
                self.assertEqual(follow.command(100.55), (.1, 0, 0))
                self.assertEqual(follow.command(100.550001), (0, 0, 0))
                self.assertIsNotNone(follow.reason)

    def test_duplicate_frames_do_not_complete_five_frame_startup(self):
        # Five observations of one receiver index are not five consecutive valid frames.
        follow = control.FollowControl(100)
        for i in range(1, 6):
            follow.observe([ball()], 100+i*.02, 100+i*.02, 1)
        self.assertIsNone(follow.baseline_area)
        self.assertEqual(follow.command(100.11), (0, 0, 0))

    def test_stale_inference_and_long_gap_stop_before_accepting_new_frame(self):
        # Refreshing the timestamp before checking age conceals slow inference or a feed gap.
        follow = control.FollowControl(100)
        follow.observe([ball()], 100, 100.300001, 1)
        self.assertIsNotNone(follow.reason)
        self.assertIsNone(follow.last_frame_at)
        follow = control.FollowControl(100)
        prime(follow)
        follow.observe([ball()], 100.6, 100.6, 6)
        self.assertEqual(follow.command(100.6), (0, 0, 0))
        self.assertEqual(follow.last_frame_at, 100.25)

    def test_future_or_nonfinite_times_stop_without_issuing_motion(self):
        # A future timestamp creates a negative age and defeats the freshness gate.
        for received, now in [(100.31,100.3),(float('nan'),100.3),(100.3,float('inf'))]:
            with self.subTest(received=received, now=now):
                follow = control.FollowControl(100)
                prime(follow)
                result = follow.observe([ball()], received, now, 6)
                self.assertEqual((result['vx'],result['vy'],result['vyaw']), (0,0,0))
                self.assertIsNotNone(follow.reason)

    def test_startup_waits_five_seconds_only_before_first_frame(self):
        # Applying the 0.3-second feed timeout before connection prevents valid startup.
        follow = control.FollowControl(100)
        self.assertEqual(follow.command(104.99), (0,0,0))
        self.assertIsNone(follow.reason)
        self.assertEqual(follow.command(105), (0,0,0))
        self.assertIsNotNone(follow.reason)
        follow = control.FollowControl(100)
        follow.observe([ball()], 104.8, 104.8, 1)
        self.assertEqual(follow.command(105.1), (0,0,0))
        self.assertIsNone(follow.reason)
        follow.command(105.100001)
        self.assertIsNotNone(follow.reason)

    def test_absolute_deadline_is_not_extended_by_fresh_frames(self):
        # Using moving_started_at instead of started_at extends the approved duration.
        follow = control.FollowControl(100, duration=.4)
        prime(follow)
        follow.observe([ball()], 100.39, 100.39, 6)
        self.assertEqual(follow.command(100.399), (.1,0,0))
        self.assertEqual(follow.command(100.4), (0,0,0))
        self.assertIsNotNone(follow.reason)

    def test_stop_preserves_first_reason_and_never_restarts(self):
        # Overwriting the first failure hides why the controller stopped.
        follow = control.FollowControl(100)
        prime(follow)
        follow.stop('manual_stop')
        follow.stop('later_stop')
        self.assertEqual(follow.reason, 'manual_stop')
        self.assertFalse(follow.reached)
        self.assertEqual(follow.command(100.3), (0,0,0))

    def test_invalid_parameter_limits_are_rejected(self):
        # Nonfinite or oversized limits invalidate the bounded experiment.
        for kwargs in [{'started_at':float('nan')},{'duration':0},{'duration':-1},
                       {'duration':30.0001},{'duration':float('inf')},{'duration':True},
                       {'max_age':0},{'max_age':.300001},{'max_age':float('nan')}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                control.FollowControl(**{'started_at':100, **kwargs})

    def test_selected_speed_changes_forward_command_only_after_five_frames(self):
        # Ignoring the selected speed or scaling yaw with it breaks the controlled comparison.
        for speed in [.1, .3]:
            with self.subTest(speed=speed):
                follow = control.FollowControl(100, speed=speed)
                for index in range(1, 6):
                    result = follow.observe([ball(216)], 100+index*.05, 100+index*.05, index)
                    if index < 5:
                        self.assertEqual((result['vx'], result['vy'], result['vyaw']), (0, 0, 0))
                self.assertEqual(follow.command(100.26), (speed, 0, -.12))
                self.assertEqual(result['vx'], speed)
                self.assertEqual(result['vy'], 0)
                self.assertAlmostEqual(result['e'], .2)
                self.assertAlmostEqual(result['vyaw'], -.12)
                self.assertAlmostEqual(result['start_area'], 1/144)
                self.assertEqual(result['stop_area'], .012)

    def test_speed_point_three_keeps_area_and_missing_ball_stop_latched(self):
        # A faster comparison must not relax the fixed area stop or restart after target loss.
        for targets, reason, reached in [([ball(width=80)], 'target_area_reached', True),
                                         ([], 'missing_ball', False)]:
            with self.subTest(reason=reason):
                follow = control.FollowControl(100, speed=.3)
                prime(follow)
                self.assertEqual(follow.command(100.26), (.3, 0, 0))
                result = follow.observe(targets, 100.3, 100.3, 6)
                self.assertEqual((result['vx'], result['vy'], result['vyaw']), (0, 0, 0))
                self.assertEqual(result['reason'], reason)
                self.assertEqual(follow.reached, reached)
                follow.observe([ball(width=10, height=10)], 100.35, 100.35, 7)
                self.assertEqual(follow.command(100.36), (0, 0, 0))
                self.assertEqual(follow.reason, reason)

    def test_speed_point_three_keeps_feedback_age_and_absolute_duration_limits(self):
        # Selecting .3 m/s must not extend stale feedback or the fixed trial deadline.
        follow = control.FollowControl(100, speed=.3)
        prime(follow)
        self.assertEqual(follow.command(100.55), (.3, 0, 0))
        self.assertEqual(follow.command(100.550001), (0, 0, 0))
        self.assertEqual(follow.reason, 'frame_timeout')
        follow.observe([ball()], 100.56, 100.56, 6)
        self.assertEqual(follow.command(100.57), (0, 0, 0))
        follow = control.FollowControl(100, duration=.4, speed=.3)
        prime(follow)
        follow.observe([ball()], 100.39, 100.39, 6)
        self.assertEqual(follow.command(100.399), (.3, 0, 0))
        self.assertEqual(follow.command(100.4), (0, 0, 0))
        self.assertEqual(follow.reason, 'duration_limit')

    def test_speed_outside_two_approved_values_is_rejected(self):
        # Rejecting invalid speeds prevents arbitrary/unbounded trials through the Python API.
        for speed in [0, -.1, .2, .300001, .4, float('nan'), float('inf'), True, '0.3', None]:
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                control.FollowControl(100, speed=speed)


class CameraMotionContinuationTests(unittest.TestCase):
    identity = [1., 0., 0., 0.]
    angle = -math.atan(100/434.55844122715706)
    shifted = [math.cos(angle/2), 0., math.sin(angle/2), 0.]

    def follow(self, **kwargs):
        try:
            return control.FollowControl(100, **{'orange_continuation': True,
                'camera_motion_compensation': True, 'stop_area': .1, **kwargs})
        except TypeError as error:
            self.fail(f'opt-in camera motion compensation is unavailable: {error}')

    def prime(self, instance, center=180, width=40):
        for index in range(1, 6):
            now = 100+index*.05
            instance.observe([ball(center, width, width)], now, now, index,
                             camera_quat=self.identity)

    def test_camera_option_is_boolean_and_requires_explicit_orange_mode(self):
        for value in (1, 0, 'true', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.follow(camera_motion_compensation=value)
        with self.assertRaises(ValueError):
            self.follow(orange_continuation=False)

    def test_camera_compensated_orange_jump_changes_association_only(self):
        follow = self.follow()
        self.prime(follow)
        result = follow.observe([OrangeContinuationTests.orange(center=280)],
            100.3, 100.3, 6, camera_quat=self.shifted)
        self.assertIsNone(result['reason'])
        self.assertEqual(result['vx'], .1)
        self.assertAlmostEqual(result['vyaw'], -.2)  # Existing raw-pixel steering saturation.
        self.assertAlmostEqual(result['area'], 1600/(360*640))
        self.assertEqual(result['association'], 'camera_motion_continuation')
        self.assertTrue(result.get('camera_motion_compensation'))
        self.assertAlmostEqual(result['predicted_center'][0], 280)
        self.assertAlmostEqual(result['predicted_center'][1], 320)

    def test_raw_match_remains_accepted_even_when_predicted_center_is_elsewhere(self):
        follow = self.follow()
        self.prime(follow)
        result = follow.observe([OrangeContinuationTests.orange()], 100.3, 100.3, 6,
                                camera_quat=self.shifted)
        self.assertEqual(result['vx'], .1)
        self.assertEqual(result['association'], 'spatial_continuation')
        self.assertEqual(result['vyaw'], 0)

    def test_raw_and_compensated_matches_form_one_ambiguity_union(self):
        # Selecting only the compensated candidate would silently switch away from a raw match.
        for use_sports_ball in (False, True):
            with self.subTest(use_sports_ball=use_sports_ball):
                follow = self.follow()
                self.prime(follow)
                targets = [OrangeContinuationTests.orange(),
                           ball(280) if use_sports_ball else OrangeContinuationTests.orange(280)]
                result = follow.observe(targets, 100.3, 100.3, 6, camera_quat=self.shifted)
                self.assertEqual(result['reason'], 'ambiguous_target')
                self.assertEqual(result['vx'], 0)

    def test_predicted_match_does_not_relax_confidence_scale_or_full_box_checks(self):
        for target, reason, mode in ((OrangeContinuationTests.orange(280, confidence=.24),
                                      'low_confidence', 'stopped'),
                                     (OrangeContinuationTests.orange(280, width=80, height=80),
                                      'target_scale_jump', 'stopped'),
                                     (OrangeContinuationTests.orange(340, width=40, height=40),
                                      None, 'holding')):
            with self.subTest(reason=reason, mode=mode):
                follow = self.follow()
                self.prime(follow)
                result = follow.observe([target], 100.3, 100.3, 6, camera_quat=self.shifted)
                self.assertEqual(result['reason'], reason)
                self.assertEqual(result['mode'], mode)
                self.assertEqual(result['vx'], 0)
                if mode == 'holding':
                    self.assertIsNone(result['area'])

    def test_remote_orange_outside_both_references_still_latches(self):
        follow = self.follow()
        self.prime(follow)
        target = OrangeContinuationTests.orange(60)
        result = follow.observe([target], 100.3, 100.3, 6, camera_quat=self.shifted)
        self.assertEqual(result['reason'], 'target_jump')
        self.assertEqual(result['vx'], 0)

    def test_none_pose_holds_before_acquisition_and_cannot_count_a_frame(self):
        follow = self.follow()
        result = follow.observe([ball()], 100.05, 100.05, 1)
        self.assertEqual(result['mode'], 'holding')
        self.assertEqual(result['vx'], 0)
        self.assertIsNone(result['area'])
        self.assertIsNone(follow.baseline_area)
        for index in range(2, 6):
            now = 100+index*.05
            result = follow.observe([ball()], now, now, index, camera_quat=self.identity)
            self.assertEqual(result['vx'], 0)
        result = follow.observe([ball()], 100.3, 100.3, 6, camera_quat=self.identity)
        self.assertEqual(result['vx'], .1)

    def test_missing_pose_recovers_using_last_full_pose_and_never_extends_deadline(self):
        follow = self.follow()
        self.prime(follow)
        result = follow.observe([ball(280)], 100.3, 100.3, 6)
        self.assertEqual(result['vx'], 0)
        self.assertEqual(result['association'], 'camera_pose_hold')
        result = follow.observe([OrangeContinuationTests.orange(280)], 100.4, 100.4, 7,
                                camera_quat=self.shifted)
        self.assertEqual(result['vx'], .1)
        self.assertEqual(result['association'], 'camera_motion_continuation')
        follow.observe([ball()], 100.45, 100.45, 8)
        follow.observe([ball()], 100.65, 100.65, 9)
        self.assertEqual(follow.command(100.75), (0, 0, 0))
        self.assertEqual(follow.reason, 'camera_pose_unavailable')
        result = follow.observe([ball()], 100.76, 100.76, 10, camera_quat=self.identity)
        self.assertEqual(result['vx'], 0)
        self.assertEqual(result['reason'], 'camera_pose_unavailable')

    def test_duplicate_pose_recovery_cannot_clear_hold_or_refresh_frame(self):
        follow = self.follow()
        self.prime(follow)
        follow.observe([ball()], 100.3, 100.3, 6)
        result = follow.observe([ball()], 100.5, 100.5, 6, camera_quat=self.identity)
        self.assertEqual(result['mode'], 'holding')
        self.assertEqual(follow.last_frame_at, 100.3)
        follow.command(100.6)
        self.assertEqual(follow.reason, 'camera_pose_unavailable')

    def test_pose_missing_and_empty_share_first_absence_deadline(self):
        for missing_pose_first, reason in ((True, 'camera_pose_unavailable'),
                                           (False, 'missing_ball')):
            with self.subTest(missing_pose_first=missing_pose_first):
                follow = self.follow()
                self.prime(follow)
                if missing_pose_first:
                    follow.observe([ball()], 100.3, 100.3, 6)
                    follow.observe([], 100.5, 100.5, 7, camera_quat=self.identity)
                else:
                    follow.observe([], 100.3, 100.3, 6, camera_quat=self.identity)
                    follow.observe([ball()], 100.5, 100.5, 7)
                follow.command(100.6)
                self.assertEqual(follow.reason, reason)

    def test_clipped_frame_does_not_replace_full_reference_pose(self):
        follow = self.follow()
        self.prime(follow, center=40)
        # This raw spatial match clips; its shifted quaternion cannot replace the old pose.
        result = follow.observe([OrangeContinuationTests.orange(center=21)], 100.3, 100.3, 6,
                                camera_quat=self.shifted)
        self.assertEqual(result['association'], 'clipped_hold')
        result = follow.observe([OrangeContinuationTests.orange(40)], 100.4, 100.4, 7,
                                camera_quat=self.identity)
        self.assertEqual(result['vx'], .1)
        self.assertAlmostEqual(result['predicted_center'][0], 40)

    def test_invalid_pose_and_behind_camera_projection_latch_zero(self):
        for quat, reason in (([0, 0, 0, 0], 'invalid_camera_pose'),
                             ([1, 0, float('nan'), 0], 'invalid_camera_pose'),
                             ([1.06, 0, 0, 0], 'invalid_camera_pose'),
                             ([0, 0, 1, 0], 'invalid_camera_projection')):
            with self.subTest(reason=reason, quat=quat):
                follow = self.follow()
                self.prime(follow)
                result = follow.observe([OrangeContinuationTests.orange()], 100.3, 100.3, 6,
                                        camera_quat=quat)
                self.assertEqual(result['reason'], reason)
                self.assertEqual(result['vx'], 0)
                self.assertEqual(follow.command(100.35), (0, 0, 0))

    def test_fixed_area_stop_uses_raw_box_after_pose_compensated_continuation(self):
        follow = self.follow(stop_area=.06)
        self.prime(follow, width=100)
        target = OrangeContinuationTests.orange(280, width=120, height=120)
        result = follow.observe([target], 100.3, 100.3, 6, camera_quat=self.shifted)
        self.assertEqual(result['reason'], 'target_area_reached')
        self.assertAlmostEqual(result['area'], .0625)
        self.assertEqual(result['vx'], 0)
        self.assertTrue(follow.reached)

    def test_default_mode_does_not_require_or_validate_camera_pose(self):
        follow = control.FollowControl(100)
        for index in range(1, 6):
            now = 100+index*.05
            try:
                result = follow.observe([ball()], now, now, index, camera_quat=[float('nan')]*4)
            except TypeError as error:
                self.fail(f'optional camera pose API is unavailable: {error}')
        self.assertEqual(result['vx'], .1)
        self.assertNotIn('camera_motion_compensation', result)
        self.assertNotIn('predicted_center', result)


if __name__ == '__main__':
    unittest.main()
