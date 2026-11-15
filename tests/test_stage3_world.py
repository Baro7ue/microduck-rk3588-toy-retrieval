"""Stage 3 field picture without a simulator: the start pose and odometry, the sighting table,
routes around toys and the tray, the B segment start, the box planes and the dev layouts."""
from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

from microduck_toy.scene import toys
from microduck_toy.stage3 import world
from microduck_toy.stage3.world import Field, FieldPose, Sightings

REPO = Path(__file__).resolve().parents[1]
STAGE1 = json.loads((REPO / 'configs/control/stage1.json').read_text())
STAGE3 = json.loads((REPO / 'configs/control/stage3.json').read_text())
FIELD = Field.from_config(STAGE1, STAGE3)


def state_at(x, y, yaw):
    return {'odom': {'position': [x, y, 0.12], 'yaw': yaw}}


class FieldPoseTests(unittest.TestCase):
    def test_start_pose_is_where_odometry_starts(self):
        pose = FieldPose((-0.5, 0.2, math.pi / 2), (3.0, -1.0, 0.4))
        x, y, yaw = pose.of(state_at(3.0, -1.0, 0.4))
        self.assertAlmostEqual(x, -0.5)
        self.assertAlmostEqual(y, 0.2)
        self.assertAlmostEqual(yaw, math.pi / 2)

    def test_odometry_change_is_turned_into_the_field(self):
        # robotd's odometry heading 0.4 = field heading pi/2: 1 m along odometry heading -> +y
        pose = FieldPose((-0.5, 0.2, math.pi / 2), (3.0, -1.0, 0.4))
        moved = state_at(3.0 + math.cos(0.4), -1.0 + math.sin(0.4), 0.4 + 0.3)
        x, y, yaw = pose.of(moved)
        self.assertAlmostEqual(x, -0.5, places=9)
        self.assertAlmostEqual(y, 1.2, places=9)
        self.assertAlmostEqual(yaw, math.pi / 2 + 0.3, places=9)

    def test_body_and_field_points_invert(self):
        pose = FieldPose((0.1, -0.3, 2.5), (0.0, 0.0, 0.0))
        st = state_at(0.2, 0.1, -0.7)
        xy = pose.point(st, (0.4, -0.15))
        back = pose.body(st, xy)
        self.assertAlmostEqual(back[0], 0.4, places=9)
        self.assertAlmostEqual(back[1], -0.15, places=9)

    def test_bearing_left_is_positive(self):
        self.assertAlmostEqual(world.bearing_to((0, 0, 0), (1, 1)), math.pi / 4)
        self.assertAlmostEqual(abs(world.bearing_to((0, 0, math.pi), (1, 0))), math.pi)
        self.assertAlmostEqual(world.bearing_to((0, 0, math.pi / 2), (1, 0)), -math.pi / 2)
        self.assertAlmostEqual(world.wrap(3 * math.pi), math.pi)


class SightingTests(unittest.TestCase):
    def test_same_class_within_cluster_merges(self):
        seen = Sightings(cluster_m=0.15)
        a = seen.add('bus', (1.0, 0.0), 1.0, 1.0)
        b = seen.add('bus', (1.1, 0.0), 1.0, 0.8)
        self.assertIs(a, b)
        self.assertAlmostEqual(a.x, 1.05)
        self.assertEqual((a.frames, a.nearest_m), (2, 0.8))

    def test_weights_pull_the_mean(self):
        seen = Sightings(cluster_m=0.15)
        s = seen.add('bus', (1.0, 0.0), 0.25, 1.0)     # a spin frame
        seen.add('bus', (1.1, 0.0), 1.0, 1.0)           # a standing look
        self.assertAlmostEqual(s.x, 1.08)

    def test_far_or_other_class_is_a_new_spot(self):
        seen = Sightings(cluster_m=0.15)
        seen.add('bus', (1.0, 0.0), 1.0, 1.0)
        seen.add('bus', (1.3, 0.0), 1.0, 1.0)
        seen.add('croc', (1.0, 0.0), 1.0, 1.0)
        self.assertEqual(len(seen.spots), 3)
        self.assertEqual(len(seen.of_class('bus')), 2)

    def test_dropped_spot_takes_no_more_hits(self):
        seen = Sightings(cluster_m=0.15)
        old = seen.add('bus', (1.0, 0.0), 1.0, 1.0)
        old.status = 'dropped'
        new = seen.add('bus', (1.0, 0.0), 1.0, 1.0)
        self.assertIsNot(old, new)
        self.assertEqual(seen.of_class('bus'), [new])

    def test_obstacles_are_everything_but_the_target(self):
        seen = Sightings(cluster_m=0.15)
        t = seen.add('bus', (1.0, 0.0), 1.0, 1.0)
        d = seen.add('bus', (0.0, 0.5), 1.0, 1.0)
        d.status = 'distractor'
        seen.add('croc', (0.5, 0.5), 1.0, 1.0)
        obs = seen.obstacles('bus', skip=t)
        self.assertEqual(sorted(obs), [(0.0, 0.5), (0.5, 0.5)])
        self.assertNotIn(t.xy, seen.obstacles('bus', skip=t))


class RouteTests(unittest.TestCase):
    def test_free_line_is_one_leg(self):
        self.assertEqual(world.plan_route((-0.5, 0.0), (0.0, 0.5), [], FIELD), [(0.0, 0.5)])

    def test_goes_around_a_toy_with_clearance(self):
        start, goal, toy = (-0.6, 0.0), (0.2, 0.0), (-0.2, 0.0)
        route = world.plan_route(start, goal, [toy], FIELD)
        self.assertGreater(len(route), 1)
        self.assertEqual(route[-1], goal)
        p = start
        for q in route:
            self.assertGreaterEqual(world.seg_point_dist(p, q, toy), FIELD.clear_m - 1e-9)
            p = q
        self.assertLess(world.route_length(start, route), 1.0)

    def test_goes_around_the_tray(self):
        cx, cy = FIELD.tray_centre
        start, goal = (cx, cy - 0.5), (cx, cy + 0.5)
        route = world.plan_route(start, goal, [], FIELD)
        box = FIELD.tray_box()
        p = start
        for q in route:
            self.assertFalse(world.seg_hits_box(p, q, box), (p, q))
            p = q

    def drop_spot_and_far_goal(self):
        """A drop spot (0.19 m from the centre on x-, in the tray box) and a goal whose straight line
        from there crosses the bare tray (board run 20261112T025144Z: the bus kicked out)."""
        cx, cy = FIELD.tray_centre
        lim = FIELD.half_m - FIELD.keep_in_m
        start, goal = (cx - 0.19, cy), (cx + 0.05, min(lim, cy + 0.6))
        self.assertTrue(world.seg_hits_box(start, goal, FIELD.tray_box(-FIELD.tray_inflate_m)))
        return start, goal

    def test_without_tray_exit_a_start_in_the_tray_box_ignores_the_tray(self):
        start, goal = self.drop_spot_and_far_goal()       # the old rule, kept for Stage 3 (tray_exit_m 0)
        self.assertEqual(world.plan_route(start, goal, [], FIELD), [goal])

    def test_with_tray_exit_a_start_in_the_tray_box_steps_out_then_goes_round(self):
        import dataclasses
        fld = dataclasses.replace(FIELD, tray_exit_m=0.05)
        start, goal = self.drop_spot_and_far_goal()
        route = world.plan_route(start, goal, [], fld)
        cx, cy = fld.tray_centre
        box = fld.tray_box()
        self.assertFalse(world.in_box(route[0], box))
        self.assertGreater(math.dist(route[0], (cx, cy)), math.dist(start, (cx, cy)))  # straight away
        self.assertLess(abs(route[0][1] - start[1]), 1e-9)                          # along x- here
        p = route[0]
        for q in route[1:]:
            self.assertFalse(world.seg_hits_box(p, q, box), (p, q))
            p = q
        self.assertEqual(route[-1], goal)

    def test_tray_exit_near_the_wall_goes_out_sideways(self):
        import dataclasses
        fld = dataclasses.replace(FIELD, tray_exit_m=0.05)
        cx, cy = fld.tray_centre
        lim = fld.half_m - fld.keep_in_m
        start = (min(lim, cx + 0.12), cy + 0.02)          # behind the tray: straight out is past the wall
        out = world.tray_exit(start, fld)
        self.assertTrue(fld.inside(out), out)
        self.assertFalse(world.in_box(out, fld.tray_box()), out)

    def test_room_field_has_tray_exit(self):
        from microduck_toy.stage3 import brain as brain_lib
        configs = brain_lib.load_configs(override='configs/demo/room.json')
        fld = Field.from_config(configs['stage1'], configs['stage3'])
        self.assertEqual(fld.tray_exit_m, 0.05)
        self.assertEqual(FIELD.tray_exit_m, 0.0)

    def test_start_outside_the_keep_in_square_steps_back_in(self):
        lim = FIELD.half_m - FIELD.keep_in_m
        route = world.plan_route((-0.85, -0.03), (0.2, 0.0), [], FIELD)
        self.assertEqual(route[0], (-lim, -0.03))
        self.assertEqual(route[-1], (0.2, 0.0))

    def test_obstacle_on_the_goal_is_ignored(self):
        self.assertEqual(world.plan_route((-0.5, 0.0), (0.0, 0.0), [(0.05, 0.0)], FIELD), [(0.0, 0.0)])

    def test_stays_inside_the_walls(self):
        lim = FIELD.half_m - FIELD.keep_in_m
        route = world.plan_route((-lim, -0.3), (-lim, 0.3), [(-lim, 0.0)], FIELD)
        for q in route:
            self.assertTrue(FIELD.inside(q), q)

    def test_tof_circle_obstacle(self):
        route = world.plan_route((-0.6, 0.0), (0.2, 0.0), [((-0.2, 0.0), 0.2)], FIELD)
        p = (-0.6, 0.0)
        for q in route:
            self.assertGreaterEqual(world.seg_point_dist(p, q, (-0.2, 0.0)), 0.2 - 1e-9)
            p = q

    def test_liang_barsky(self):
        box = (0.0, 0.0, 1.0, 1.0)
        self.assertTrue(world.seg_hits_box((-1, 0.5), (2, 0.5), box))
        self.assertFalse(world.seg_hits_box((-1, 1.5), (2, 1.5), box))
        self.assertFalse(world.seg_hits_box((-1, 0.0), (2, 0.0), box))     # along an edge: not inside


class ApproachPointTests(unittest.TestCase):
    def test_on_the_side_the_robot_comes_from(self):
        p = world.approach_point((-0.8, 0.0), (0.0, 0.0), 0.6, [], FIELD, 0.12)
        self.assertAlmostEqual(p[0], -0.6)
        self.assertAlmostEqual(p[1], 0.0)

    def test_toy_in_the_corridor_moves_the_start(self):
        p = world.approach_point((-0.8, 0.0), (0.0, 0.0), 0.6, [(-0.3, 0.0)], FIELD, 0.12)
        self.assertAlmostEqual(math.dist(p, (0.0, 0.0)), 0.6)
        self.assertGreaterEqual(world.seg_point_dist(p, (0.0, 0.0), (-0.3, 0.0)), 0.12)
        self.assertAlmostEqual(abs(math.degrees(math.atan2(p[1], p[0]))), 150.0, places=6)

    def test_never_outside_or_across_the_tray(self):
        cx, cy = FIELD.tray_centre
        toy = (cx - 0.35, cy)                       # tray_clear_m from the tray centre
        p = world.approach_point((cx + 0.1, cy + 0.6), toy, 0.6, [], FIELD, 0.12)
        self.assertTrue(FIELD.inside(p))
        self.assertFalse(world.seg_hits_box(p, toy, FIELD.tray_box(-FIELD.tray_inflate_m)))


class ConfigTests(unittest.TestCase):
    def test_plane_z_is_half_the_toy_height(self):
        cat = toys.catalogue()
        for name in ('bus', 'henry', 'ketchup', 'croc', 'thomas'):
            self.assertAlmostEqual(STAGE3['plane_z_m'][name], cat[toys.resolve(name)].rest_z, places=4,
                                   msg=name)
        self.assertEqual(STAGE3['plane_z_m']['tray'], 0.0)     # the near rim meets the floor (3P)

    def test_dev_layouts_are_not_formal(self):
        doc = json.loads((REPO / 'configs/control/stage3_layouts.json').read_text())
        seeds = [l['seed'] for l in doc['layouts']]
        base = STAGE3['episode']['dev_seed_base']
        self.assertEqual(seeds, list(range(base, base + STAGE3['episode']['dev_seeds'])))
        formal = range(STAGE1['layout']['formal_seed_base'],
                       STAGE1['layout']['formal_seed_base'] + STAGE1['layout']['formal_seeds'])
        self.assertFalse(set(seeds) & set(formal))

    def test_dev_layouts_follow_the_stage1_rules(self):
        doc = json.loads((REPO / 'configs/control/stage3_layouts.json').read_text())
        rules, half = STAGE1['layout'], STAGE1['arena']['half_size_m']
        tray = STAGE1['arena']['tray']['centre_m']
        for lay in doc['layouts']:
            spots = [tuple(v[:2]) for v in lay['slots'].values()]
            for p in spots:
                self.assertLessEqual(max(abs(p[0]), abs(p[1])), half - rules['wall_clear_m'] + 1e-9)
                self.assertGreaterEqual(math.dist(p, tray), rules['tray_clear_m'] - 1e-9)
                self.assertGreaterEqual(math.dist(p, lay['robot'][:2]), rules['start_clear_m'] - 1e-9)
            for i, p in enumerate(spots):
                for q in spots[i + 1:]:
                    self.assertGreaterEqual(math.dist(p, q), rules['toy_gap_m'] - 1e-9)

    def test_tof_stop_leaves_room_to_slide(self):
        tof = STAGE3['tof']
        # stopping from 0.3 m/s slides 3.9-4.9 cm; the toes are 3.4-3.7 cm ahead of the trunk
        self.assertGreater(tof['stop_range_m'] - 0.049 - 0.037, 0.08)

    def test_tof_sees_a_wall_for_more_than_two_frames(self):
        """A wall counts once its slant range is >= self_blind_m and it is <= stop_range_m from the
        trunk; the sensor sits up to 8.5 cm ahead of the trunk (3P), and at 0.224 m/s and 15 Hz the
        robot closes 1.5 cm per frame."""
        tof = STAGE3['tof']
        window = tof['stop_range_m'] - 0.0846 - tof['self_blind_m']
        self.assertGreaterEqual(window, (tof['frames'] + 1) * 0.224 / tof['hz'])

if __name__ == '__main__':
    unittest.main()
