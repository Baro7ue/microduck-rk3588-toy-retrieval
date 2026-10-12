"""Causal camera-pose selection from real temporary UNIX state notifications."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ball_follow_rpc as rpc
from test_ball_follow_rpc import STATE, UnixPeer, wait_for


def posed_state(sensor_at, imu=None, camera=None):
    state = copy.deepcopy(STATE)
    state['t_ns'] = round(sensor_at*1e9)
    state['imu'] = {'quat': [1., 0., 0., 0.] if imu is None else imu}
    state['frames'] = {'camera': {'quat': [1., 0., 0., 0.] if camera is None else camera}}
    return state


class CameraPoseHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pose-history-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clock = SimpleNamespace(now=100.)
        patcher = mock.patch.object(rpc, 'time', SimpleNamespace(monotonic=lambda: self.clock.now))
        patcher.start()
        self.addCleanup(patcher.stop)

    def link(self, sensor_at=100., received_at=100., state=None):
        self.clock.now = received_at
        peer = UnixPeer(self.root/'robot.sock', state=posed_state(sensor_at) if state is None else state)
        self.addCleanup(peer.close)
        link = rpc.RobotLink(self.root/'robot.sock', self.root/'evidence', io_timeout=.05)
        self.addCleanup(link.close)
        link.start()
        link.wait_ready()
        return link, peer

    def pose(self, link, frame_at, **kwargs):
        self.assertTrue(callable(getattr(link, 'camera_pose_at', None)),
                        'causal camera-pose history has not been implemented')
        return link.camera_pose_at(frame_at, **kwargs)

    def send(self, link, peer, sensor_at, received_at, **kwargs):
        self.clock.now = received_at
        peer.send_state(posed_state(sensor_at, **kwargs))
        wait_for(lambda: link.snapshot()[0]['t_ns'] == round(sensor_at*1e9))

    def test_latest_causal_sample_composes_world_orientation_and_preserves_latest_state(self):
        # Selecting the newest snapshot instead of the frame's causal history fails here.
        link, peer = self.link()
        half = 2**-.5
        self.send(link, peer, 100.04, 100.05, imu=[half, 0., 0., half])
        self.send(link, peer, 100.08, 100.09)
        pose = self.pose(link, 100.055)
        self.assertEqual(pose['sensor_t_ns'], 100040000000)
        self.assertEqual(pose['received_monotonic_s'], 100.05)
        self.assertAlmostEqual(pose['sensor_to_frame_s'], .015)
        self.assertAlmostEqual(pose['receipt_to_frame_s'], .005)
        for actual, expected in zip(pose['quat'], [half, 0., 0., half]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(link.snapshot()[0]['t_ns'], 100080000000)

    def test_trunk_orientation_uses_same_causal_sample_and_is_normalized(self):
        link, peer = self.link()
        half = 2**-.5
        # Three states are causal for this frame; a fourth is newer than the frame.
        # Distinct trunk and camera rotations expose either latest-state lookup or
        # returning the composed world quaternion in place of trunk-to-world.
        self.send(link, peer, 100.02, 100.03, camera=[half, 0., half, 0.])
        selected_camera = [1.02*half, 1.02*half, 0., 0.]
        self.send(link, peer, 100.04, 100.05,
                  imu=[1.01*half, 0., 0., 1.01*half], camera=selected_camera)
        self.send(link, peer, 100.08, 100.09, camera=[half, 0., 0., half])
        pose = self.pose(link, 100.075)
        self.assertEqual(set(pose), {'quat', 'trunk_quat', 'sensor_t_ns',
                                    'received_monotonic_s', 'sensor_to_frame_s',
                                    'receipt_to_frame_s'})
        self.assertEqual(pose['sensor_t_ns'], 100040000000)
        self.assertEqual(pose['received_monotonic_s'], 100.05)
        self.assertAlmostEqual(pose['sensor_to_frame_s'], .035)
        self.assertAlmostEqual(pose['receipt_to_frame_s'], .025)
        for actual, expected in zip(pose['trunk_quat'], [half, 0., 0., half]):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(pose['quat'], [.5, .5, .5, .5]):
            self.assertAlmostEqual(actual, expected)
        latest, _ = link.snapshot()
        self.assertEqual(latest['t_ns'], 100080000000)
        self.assertEqual(latest['imu']['quat'], [1., 0., 0., 0.])
        self.assertEqual(latest['frames']['camera']['quat'], [half, 0., 0., half])
        self.assertIsNone(link.error)

    def test_future_sensor_sample_is_unavailable_even_if_received(self):
        # Receipt alone cannot authorize a pose from after the image receipt.
        link, _ = self.link(sensor_at=100.03)
        self.assertIsNone(self.pose(link, 100.02))

    def test_future_local_receipt_is_unavailable_even_if_sensor_is_past(self):
        # Retrospective sensor time cannot make a not-yet-received state causal.
        link, _ = self.link(sensor_at=100., received_at=100.05)
        self.assertIsNone(self.pose(link, 100.04))

    def test_sensor_skew_bound_is_independent_of_receipt_freshness(self):
        link, _ = self.link(sensor_at=99.9)
        self.assertIsNone(self.pose(link, 100.01))
        self.assertIsNotNone(self.pose(link, 100.01, max_skew=.12))

    def test_receipt_skew_bound_is_independent_of_sensor_freshness(self):
        link, _ = self.link(sensor_at=100., received_at=99.9)
        self.assertIsNone(self.pose(link, 100.01))
        self.assertIsNotNone(self.pose(link, 100.01, max_skew=.12))

    def test_exact_skew_boundary_is_accepted_and_older_sample_is_rejected(self):
        link, _ = self.link()
        self.assertIsNotNone(self.pose(link, 100.1))
        self.assertIsNone(self.pose(link, 100.100001))

    def test_duplicate_tick_does_not_refresh_pose_or_history_receipt(self):
        # Replacing a cached sample on duplicate ticks hides sensor staleness.
        link, peer = self.link()
        half = 2**-.5
        self.clock.now = 100.04
        peer.send_state(posed_state(100., imu=[half, 0., 0., half],
                                   camera=[half, 0., 0., half]))
        wait_for(lambda: abs(link.snapshot()[0]['frames']['camera']['quat'][3]) > .7)
        pose = self.pose(link, 100.05)
        self.assertEqual(list(pose['quat']), [1., 0., 0., 0.])
        self.assertEqual(list(pose['trunk_quat']), [1., 0., 0., 0.])
        self.assertEqual(pose['received_monotonic_s'], 100.)
        self.assertEqual(link.snapshot()[1], 100.)

    def test_history_is_bounded_and_does_not_retain_evicted_poses(self):
        # An unbounded history would still return the very old frame's pose.
        link, peer = self.link()
        for index in range(70):
            self.send(link, peer, 101+index*.01, 101+index*.01)
        self.assertIsNone(self.pose(link, 101.02))
        self.assertEqual(self.pose(link, 101.69)['sensor_t_ns'], 101690000000)

    def test_missing_legacy_pose_is_unavailable_without_transport_failure(self):
        state = copy.deepcopy(STATE)
        state['t_ns'] = 100000000000
        link, _ = self.link(state=state)
        self.assertIsNone(self.pose(link, 100.01))
        self.assertIsNone(link.error)

    def test_malformed_selected_quaternion_raises_without_poisoning_transport(self):
        # Invalid geometry is a controller fault, not a broken state socket.
        link, peer = self.link()
        self.send(link, peer, 100.02, 100.02, camera=[0., 0., 0., 0.])
        with self.assertRaises(ValueError):
            self.pose(link, 100.03)
        self.assertIsNone(link.error)
        self.assertEqual(link.snapshot()[0]['t_ns'], 100020000000)

    def test_nonfinite_quaternion_remains_a_geometry_error_and_is_logged(self):
        # Serializing the receipt envelope must not convert pose validation into EOF.
        link, peer = self.link()
        self.send(link, peer, 100.02, 100.02, camera=[float('nan'), 0., 0., 0.])
        with self.assertRaises(ValueError):
            self.pose(link, 100.03)
        self.assertIsNone(link.error)
        packets = [json.loads(line) for line in (self.root/'evidence/states.jsonl').read_text().splitlines()]
        latest = [packet for packet in packets if packet.get('method') == 'robot.state'][-1]
        self.assertEqual(latest['received_monotonic_s'], 100.02)

    def test_nonfinite_or_invalid_time_queries_are_unavailable(self):
        link, _ = self.link()
        for frame, skew in [(float('nan'), .1), (float('inf'), .1), (True, .1),
                            (100.01, float('nan')), (100.01, -1), (100.01, True)]:
            with self.subTest(frame=frame, skew=skew):
                self.assertIsNone(self.pose(link, frame, max_skew=skew))

    def test_state_log_adds_local_receipt_only_to_envelope(self):
        # Without receipt in the log, later alignment cannot distinguish late delivery.
        link, _ = self.link()
        packets = [json.loads(line) for line in (self.root/'evidence/states.jsonl').read_text().splitlines()]
        state = next(packet for packet in packets if packet.get('method') == 'robot.state')
        self.assertEqual(state.get('received_monotonic_s'), 100.)
        self.assertEqual(state['params'], posed_state(100.))
        self.assertNotIn('received_monotonic_s', link.snapshot()[0])
        models = next(packet for packet in packets if packet.get('id') == 1)
        self.assertNotIn('received_monotonic_s', models)


if __name__ == '__main__':
    unittest.main()
