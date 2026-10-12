# Moved 2026-10-11 from work/tests/test_ball_follow_rpc.py (original kept unchanged in archive/2026-10_ball_follow/work/); only paths and imports changed.
"""Real temporary UNIX peers exercise the follower's transport boundaries."""
import copy
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    from microduck_toy.rpc.robotd_client import RobotLink
except ModuleNotFoundError as error:
    if error.name != 'microduck_toy.rpc.robotd_client':
        raise
    RobotLink = None


STATE = {'t_ns': 100, 'policy': 'stand', 'head': [0.0] * 4,
         'move': {'requested': [0.0] * 3, 'applied': [0.0] * 3},
         'safety': {'fallen': False, 'gravity': [0.0, 0.0, -1.0]}}
MODELS = {'accepted': True, 'walk': 'walk_17500.onnx', 'stand': 'velstand.onnx'}


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError('fixture condition did not arrive')


class UnixPeer:
    """A local protocol peer; never opens an IP socket or contacts robotd."""
    def __init__(self, path, state=None, models=None, request_handler=None):
        self.state = copy.deepcopy(STATE if state is None else state)
        self.models = copy.deepcopy(MODELS if models is None else models)
        self.request_handler = request_handler
        self.clients, self.moves, self.requests = [], [], []
        self.subscription = None
        self.subscription_request = None
        self.stopped = threading.Event()
        self.write_lock = threading.Lock()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(path))
        self.listener.listen()
        self.listener.settimeout(0.05)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stopped.is_set():
            try:
                client, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self.clients.append(client)
            threading.Thread(target=self.handle, args=(client,), daemon=True).start()

    @staticmethod
    def packet(message):
        return (json.dumps(message) + '\n').encode()

    def send(self, message):
        with self.write_lock:
            self.subscription.sendall(self.packet(message))

    def send_state(self, state):
        self.send({'jsonrpc': '2.0', 'method': 'robot.state', 'params': state})

    def handle(self, client):
        try:
            with client.makefile('rb') as reader:
                while line := reader.readline():
                    message = json.loads(line)
                    if message['method'] == 'robot.subscribe':
                        self.subscription = client
                        self.subscription_request = message
                        client.sendall(self.packet({'jsonrpc': '2.0', 'id': message['id'],
                                                    'result': self.models}))
                        if self.models.get('accepted'):
                            self.send_state(self.state)
                    elif 'id' not in message:
                        self.moves.append(message)
                    else:
                        self.requests.append((client, message))
                        if self.request_handler:
                            self.request_handler(client, message)
                        else:
                            client.sendall(self.packet({'jsonrpc': '2.0', 'id': message['id'],
                                                        'result': {'accepted': True}}))
        except (OSError, ValueError):
            pass  # Peer closure is an intentional fixture condition.

    def close_peer(self, client):
        try:
            client.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        client.close()

    def close(self):
        self.stopped.set()
        self.listener.close()
        for client in self.clients:
            self.close_peer(client)
        self.thread.join(0.5)


class RobotLinkTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(RobotLink, 'RobotLink behavior has not been implemented')
        self.temp = tempfile.TemporaryDirectory(prefix='ball-rpc-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def link(self, **peer_options):
        peer = UnixPeer(self.root / 'robot.sock', **peer_options)
        self.addCleanup(peer.close)
        link = RobotLink(self.root / 'robot.sock', self.root / 'evidence', io_timeout=0.05)
        self.addCleanup(link.close)
        link.start()
        return link, peer

    def test_ready_records_subscription_and_uses_persistent_notifications(self):
        # Broken notification framing or reconnecting for each move must fail here.
        link, peer = self.link()
        self.assertEqual(link.wait_ready()['policy'], 'stand')
        self.assertEqual(link.models['walk'], 'walk_17500.onnx')
        self.assertEqual(peer.subscription_request['params'], {'hz': 20})
        link.move(0.1, 0.0, -0.02)
        link.move(0.0, 0.0, 0.0)
        wait_for(lambda: len(peer.moves) == 2)
        self.assertEqual(len(peer.clients), 2)
        self.assertNotIn('id', peer.moves[0])
        self.assertEqual(peer.moves[0]['params'], {'vx': 0.1, 'vy': 0.0, 'vyaw': -0.02})
        lines = (self.root / 'evidence/states.jsonl').read_text().splitlines()
        self.assertTrue(any(json.loads(line).get('method') == 'robot.state' for line in lines))

    def test_request_skips_notification_and_wrong_id_before_matching_reply(self):
        # Taking the first line as a reply would return the wrong result.
        def reply(client, request):
            packets = [{'jsonrpc': '2.0', 'method': 'robot.state', 'params': STATE},
                       {'jsonrpc': '2.0', 'id': request['id'] + 50, 'result': {'wrong': True}},
                       {'jsonrpc': '2.0', 'id': request['id'], 'result': {'accepted': True}}]
            client.sendall(b''.join(UnixPeer.packet(packet) for packet in packets))
        link, _ = self.link(request_handler=reply)
        self.assertEqual(link.request('robot.stop', {}), {'accepted': True})

    def test_notifications_cannot_extend_request_deadline(self):
        # Resetting the deadline after each notification would wait indefinitely.
        def reply(client, request):
            until = time.monotonic() + 0.8
            while time.monotonic() < until:
                client.sendall(UnixPeer.packet({'jsonrpc': '2.0', 'method': 'robot.state',
                                               'params': STATE}))
                time.sleep(0.01)
        link, _ = self.link(request_handler=reply)
        began = time.monotonic()
        with self.assertRaises(TimeoutError):
            link.request('robot.stop', {})
        self.assertLess(time.monotonic() - began, 0.6)
        self.assertIn('Traceback', link.error)

    def test_matching_rpc_error_is_raised_and_retained(self):
        def reply(client, request):
            client.sendall(UnixPeer.packet({'jsonrpc': '2.0', 'id': request['id'],
                                           'error': {'code': -1, 'message': 'fixture refusal'}}))
        link, _ = self.link(request_handler=reply)
        with self.assertRaisesRegex(RuntimeError, 'fixture refusal'):
            link.request('robot.stop', {})
        self.assertIn('fixture refusal', link.error)

    def test_repeated_sensor_times_do_not_keep_state_fresh(self):
        # Refreshing on every receipt violates the advancing-sensor freshness contract.
        link, peer = self.link()
        link.wait_ready()
        original = link.snapshot()[1]
        for tick in [100, 100, 100]:
            state = copy.deepcopy(STATE)
            state['t_ns'] = tick
            peer.send_state(state)
        wait_for(lambda: len((self.root / 'evidence/states.jsonl').read_text().splitlines()) >= 5)
        self.assertEqual(link.snapshot()[1], original)
        state['t_ns'] = 101
        peer.send_state(state)
        wait_for(lambda: link.snapshot()[1] > original)
        self.assertEqual(link.snapshot()[0]['t_ns'], 101)

    def test_backward_sensor_time_is_a_visible_failure(self):
        # Silently accepting a restarted or out-of-order sensor clock must fail here.
        link, peer = self.link()
        link.wait_ready()
        original = link.snapshot()[1]
        state = copy.deepcopy(STATE)
        state['t_ns'] = 99
        peer.send_state(state)
        wait_for(lambda: link.error is not None)
        self.assertEqual(link.snapshot()[1], original)
        self.assertIn('backward', link.error)

    def test_initial_stand_conditions_are_required(self):
        # Each unsafe first state must be rejected before a caller can start motion.
        changes = [('policy', 'held'), ('fallen', True), ('gravity', -0.89),
                   ('requested', [0.01, 0.0, 0.0]), ('applied', [0.0, 0.0, 0.01]),
                   ('requested', [])]
        for index, (field, value) in enumerate(changes):
            with self.subTest(field=field):
                state = copy.deepcopy(STATE)
                if field == 'policy':
                    state[field] = value
                elif field == 'fallen':
                    state['safety'][field] = value
                elif field == 'gravity':
                    state['safety'][field][2] = value
                else:
                    state['move'][field] = value
                peer = UnixPeer(self.root / f'bad-{index}.sock', state=state)
                self.addCleanup(peer.close)
                link = RobotLink(self.root / f'bad-{index}.sock', self.root / f'bad-{index}')
                self.addCleanup(link.close)
                link.start()
                with self.assertRaises(RuntimeError):
                    link.wait_ready()

    def test_ready_does_not_return_stale_cached_state(self):
        # A cached good stand cannot authorize motion after its local freshness expires.
        link, _ = self.link()
        link.wait_ready()
        time.sleep(0.31)
        with self.assertRaises(TimeoutError):
            link.wait_ready(timeout=0.05)

    def test_wrong_models_and_subscription_refusal_are_visible(self):
        for index, models in enumerate([{'accepted': False, 'reason': 'subscription refused'},
                                        {**MODELS, 'walk': 'other.onnx'}]):
            peer = UnixPeer(self.root / f'model-{index}.sock', models=models)
            self.addCleanup(peer.close)
            link = RobotLink(self.root / f'model-{index}.sock', self.root / f'model-{index}')
            self.addCleanup(link.close)
            link.start()
            with self.assertRaises(RuntimeError):
                link.wait_ready()
            self.assertIn('Traceback', link.error)

    def test_state_eof_and_malformed_json_become_visible(self):
        link, peer = self.link()
        link.wait_ready()
        peer.subscription.sendall(b'{broken-json}\n')
        wait_for(lambda: link.error is not None)
        self.assertIn('JSONDecodeError', link.error)
        link.close()
        other_peer = UnixPeer(self.root / 'eof.sock')
        self.addCleanup(other_peer.close)
        other = RobotLink(self.root / 'eof.sock', self.root / 'eof')
        self.addCleanup(other.close)
        other.start()
        other.wait_ready()
        other_peer.close_peer(other_peer.subscription)
        wait_for(lambda: other.error is not None)
        self.assertIn('EOF', other.error)

    def test_oversized_state_message_is_rejected(self):
        link, peer = self.link()
        link.wait_ready()
        try:
            peer.subscription.sendall(b'x' * (1024 * 1024 + 1))
        except OSError:
            pass  # A correctly bounded reader may close before the last byte is sent.
        wait_for(lambda: link.error is not None)
        self.assertIn('1 MiB', link.error)

    def test_move_write_failure_is_visible_and_close_moves_preserves_state_stream(self):
        link, peer = self.link()
        link.wait_ready()
        link.move(0.0, 0.0, 0.0)
        wait_for(lambda: len(peer.moves) == 1)
        peer.close_peer(peer.clients[1])
        with self.assertRaises(OSError):
            link.move(0.1, 0.0, 0.0)
        self.assertIn('Traceback', link.error)
        # A failed send must close the stream so partially written JSON is never retried.
        with self.assertRaisesRegex(OSError, 'closed'):
            link.move(0.1, 0.0, 0.0)
        link.close_moves()
        state = copy.deepcopy(STATE)
        state['t_ns'] = 101
        peer.send_state(state)
        wait_for(lambda: link.snapshot()[0]['t_ns'] == 101)

    def test_stop_moves_follows_old_move_with_zero_and_stop_on_same_connection(self):
        # A stop on another connection can race an older in-flight move.
        def reply(client, request):
            client.sendall(UnixPeer.packet({'jsonrpc': '2.0', 'method': 'robot.state',
                                           'params': STATE}) +
                           UnixPeer.packet({'jsonrpc': '2.0', 'id': request['id'] + 10,
                                            'result': {'wrong': True}}) +
                           UnixPeer.packet({'jsonrpc': '2.0', 'id': request['id'],
                                            'result': {'accepted': True}}))
        link, peer = self.link(request_handler=reply)
        link.wait_ready()
        self.assertTrue(callable(getattr(link, 'stop_moves', None)), 'ordered stop is not implemented')
        link.move(0.1, 0.0, -0.02)
        self.assertEqual(link.stop_moves(), {'accepted': True})
        self.assertEqual(len(peer.clients), 2)
        self.assertEqual([message['params'] for message in peer.moves],
                         [{'vx': 0.1, 'vy': 0.0, 'vyaw': -0.02},
                          {'vx': 0.0, 'vy': 0.0, 'vyaw': 0.0}])
        self.assertIs(peer.requests[0][0], peer.clients[1])
        self.assertEqual(peer.requests[0][1]['method'], 'robot.stop')
        with self.assertRaisesRegex(OSError, 'closed'):
            link.move(0.1, 0.0, 0.0)

    def test_stop_moves_closes_connection_after_matching_error(self):
        def reply(client, request):
            client.sendall(UnixPeer.packet({'jsonrpc': '2.0', 'id': request['id'],
                'error': {'code': -1, 'message': 'stop fixture refusal'}}))
        link, _ = self.link(request_handler=reply)
        link.wait_ready()
        self.assertTrue(callable(getattr(link, 'stop_moves', None)), 'ordered stop is not implemented')
        with self.assertRaisesRegex(RuntimeError, 'stop fixture refusal'):
            link.stop_moves()
        self.assertIn('stop fixture refusal', link.error)
        with self.assertRaisesRegex(OSError, 'closed'):
            link.move(0.1, 0.0, 0.0)

    def test_stop_moves_has_total_deadline_even_with_notifications(self):
        def reply(client, request):
            until = time.monotonic() + 0.8
            while time.monotonic() < until:
                client.sendall(UnixPeer.packet({'jsonrpc': '2.0', 'method': 'robot.state',
                                               'params': STATE}))
                time.sleep(0.01)
        link, _ = self.link(request_handler=reply)
        link.wait_ready()
        self.assertTrue(callable(getattr(link, 'stop_moves', None)), 'ordered stop is not implemented')
        began = time.monotonic()
        with self.assertRaises(TimeoutError):
            link.stop_moves()
        self.assertLess(time.monotonic() - began, 0.6)
        self.assertIn('Traceback', link.error)
        with self.assertRaisesRegex(OSError, 'closed'):
            link.move(0.1, 0.0, 0.0)

    def test_close_wakes_state_reader_and_preserves_log_open_error(self):
        link, _ = self.link()
        link.wait_ready()
        began = time.monotonic()
        link.close()
        self.assertLess(time.monotonic() - began, 0.6)
        self.assertIsNone(link.error)
        bad_dir = self.root / 'unwritable-log'
        (bad_dir / 'states.jsonl').mkdir(parents=True)
        broken = RobotLink(self.root / 'unused.sock', bad_dir)
        with self.assertRaises(OSError):
            broken.start()
        self.assertIn('Traceback', broken.error)
        broken.close()


if __name__ == '__main__':
    unittest.main()
