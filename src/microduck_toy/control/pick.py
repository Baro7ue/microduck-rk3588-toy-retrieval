"""The pick: open the beak, run robotd's GroundPick, close the beak at a chosen phase, confirm.

    result = ground_pick(robot, phi_close=0.33, period_s=4.0, held_threshold_deg=-1.5)

robotd never moves the mouth during GroundPick (robotd/src/main.rs:3073-3079): the 14 policy
actions skip the mouth joint, so the beak opens and shuts only on robot.mouth. This function does
what a real pick does: mouth open, `robot.do ground_pick`, count from the first tick whose policy
is ground_pick, at phase phi_close (phase = elapsed / period, robotd's 4 s walking-mode cycle) send
mouth closed, wait for the hand-back (robotd returns control at phase 0.7), stand `confirm_s`, then
read the mouth joint from robot.state: a toy in the beak stops it short of closed (-5 deg).

Times come from robot.state 't' (robotd's clock), so the phase does not depend on how fast this
process is woken. Runs unchanged against robotd on the board.
"""
from __future__ import annotations

import math
import time

MOUTH_INDEX = 9          # duck-control/src/model.rs:31


def mouth_angle(state: dict) -> float:
    return float(state['joints'][MOUTH_INDEX])


def ground_pick(robot, phi_close: float, period_s: float = 4.0, held_threshold_deg: float = -1.5,
                confirm_s: float = 1.0, open_lead_s: float = 0.3, start_timeout_s: float = 1.5,
                handback_timeout_s: float = 6.0, mark=None) -> dict:
    """Run one pick. `robot` is control.robot.Robot; `mark(label)` (optional) stamps truth logs.
    Returns the timings, whether the pick ran, and whether the mouth reads as holding."""
    stamp = mark or (lambda label: time.monotonic())
    out = {'phi_close': phi_close, 'ran': False}
    robot.mouth(1.0)
    time.sleep(open_lead_s)
    out['trigger'] = stamp('pick_trigger')
    out['reply'] = robot.request('robot.do', {'skill': 'ground_pick'})
    try:
        first = robot.wait(lambda s: s['policy'] == 'ground_pick', start_timeout_s,
                           'ground_pick start')
    except TimeoutError as error:
        robot.mouth(0.0)
        out['error'] = str(error)
        return out
    out['ran'] = True
    t0 = float(first['t'])
    out['start_t'] = t0
    close_at = t0 + phi_close * period_s
    state = robot.wait(lambda s: float(s['t']) >= close_at or s['policy'] != 'ground_pick',
                       period_s + 1.0, 'close phase')
    robot.mouth(0.0)
    out['close'] = stamp('pick_close')
    out['close_phase'] = (float(state['t']) - t0) / period_s
    try:
        back = robot.wait(lambda s: s['policy'] != 'ground_pick', handback_timeout_s,
                          'ground_pick hand-back')
        out['handback_t'] = float(back['t'])
        out['handback_phase'] = (float(back['t']) - t0) / period_s
        out['handback'] = stamp('pick_handback')
    except TimeoutError as error:
        out['error'] = str(error)
        return out
    time.sleep(confirm_s)
    state = robot.snapshot()
    angle = mouth_angle(state)
    out['mouth_deg'] = math.degrees(angle)
    out['held'] = out['mouth_deg'] > held_threshold_deg
    out['fallen'] = bool(state['safety']['fallen'])
    out['confirm'] = stamp('pick_confirm')
    return out
