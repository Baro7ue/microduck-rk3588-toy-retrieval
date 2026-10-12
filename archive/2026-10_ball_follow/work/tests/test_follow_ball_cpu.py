"""Drive must stop even while detection is blocked or body updates freeze."""
import importlib.util
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RobotFixture:
    def __init__(self):
        self.began = time.monotonic()
        self.frozen = False
        self.error = None

    def snapshot(self):
        return ({'policy': 'walk', 'safety': {'fallen': False, 'limp': False},
                 'move': {'applied': [.1, 0, 0]}},
                self.began if self.frozen else time.monotonic())

    def move(self, *_command):
        pass


class CameraFixture:
    error = None

    def raise_if_failed(self):
        if self.error:
            raise RuntimeError(self.error)


def load_feature(case):
    case.assertIsNotNone(importlib.util.find_spec('follow_ball_cpu'),
                         'visual follow drive is not implemented')
    from follow_ball_cpu import GuardedDrive
    from ball_follow_control import FollowControl
    now = time.monotonic()
    logic = FollowControl(now, duration=2)
    for i in range(1, 6):
        logic.observe([{'class_id': 32, 'confidence': .8,
                        'xyxy': [160, 300, 200, 340]}], now, now, i)
    return GuardedDrive, logic


class DriveTests(unittest.TestCase):
    def test_cancel_between_snapshot_and_send_cannot_send_a_tail_nonzero(self):
        drive_class, logic = load_feature(self)

        class CancellingRobot(RobotFixture):
            def snapshot(self):
                state = super().snapshot()
                drive.cancel.set()  # stop() has set cancellation, before taking the lock.
                return state

        drive = drive_class(CancellingRobot(), logic, CameraFixture())
        drive.start()
        self.assertTrue(drive.done.wait(.3))
        drive.stop()
        self.assertTrue(drive.records)
        self.assertTrue(all(row['vx'] == 0 for row in drive.records))

    def test_slow_send_does_not_accumulate_between_control_periods(self):
        drive_class, logic = load_feature(self)

        class SlowRobot(RobotFixture):
            def move(self, *_command):
                time.sleep(.02)

        drive = drive_class(SlowRobot(), logic, CameraFixture())
        drive.start()
        self.assertTrue(drive.done.wait(.8))
        drive.stop()
        stamps = [r['sent_monotonic_s'] for r in drive.records if r['vx'] > 0]
        self.assertGreaterEqual(len(stamps), 3)
        self.assertLess((stamps[-1]-stamps[0])/(len(stamps)-1), .065)

    def test_no_new_detection_stops_without_waiting_for_inference(self):
        drive_class, logic = load_feature(self)
        drive = drive_class(RobotFixture(), logic, CameraFixture())
        drive.start()
        # Emulates an inference that never publishes another result.
        self.assertTrue(drive.done.wait(.7))
        drive.stop()
        self.assertIsNotNone(logic.reason)
        self.assertFalse(logic.reached)
        self.assertGreater(len([r for r in drive.records if r['vx'] > 0]), 0)
        self.assertEqual([drive.records[-1][k] for k in ('vx', 'vy', 'vyaw')], [0, 0, 0])
        self.assertLessEqual(drive.records[-1]['sent_monotonic_s'] - logic.started_at, .5)

    def test_body_freeze_stops_despite_fresh_camera_results(self):
        drive_class, logic = load_feature(self)
        robot = RobotFixture()
        robot.frozen = True
        drive = drive_class(robot, logic, CameraFixture())
        drive.start()
        index = 6
        until = time.monotonic() + .7
        while not drive.done.is_set() and time.monotonic() < until:
            now = time.monotonic()
            drive.observe([{'class_id': 32, 'confidence': .8,
                            'xyxy': [160, 300, 200, 340]}], now, now, index)
            index += 1
            time.sleep(.02)
        self.assertTrue(drive.done.is_set())
        drive.stop()
        self.assertIsNotNone(drive.error)
        self.assertEqual(drive.records[-1]['vx'], 0)

    def test_camera_disconnect_cannot_refresh_nonzero_intent(self):
        drive_class, logic = load_feature(self)
        camera = CameraFixture()
        camera.error = 'camera EOF'
        drive = drive_class(RobotFixture(), logic, camera)
        drive.start()
        self.assertTrue(drive.done.wait(.3))
        drive.stop()
        self.assertTrue(drive.error)
        self.assertTrue(all(row['vx'] == 0 for row in drive.records))

    def test_latched_stop_cannot_be_overwritten_by_later_detections(self):
        drive_class, logic = load_feature(self)
        drive = drive_class(RobotFixture(), logic, CameraFixture())
        drive.start()
        drive.stop('manual_stop')
        cutoff = time.monotonic()
        drive.observe([{'class_id': 32, 'confidence': .8,
                        'xyxy': [160, 300, 200, 340]}], cutoff, cutoff, 6)
        self.assertEqual(logic.command(cutoff), (0, 0, 0))
        self.assertFalse(drive.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
