"""The Stage 5 room demo's bowl (arena tray.shape 'bowl') as plain numbers, no MuJoCo: the sim body
builds it from these (scene/arena.py), the brain ranges it (stage3/tray.py measure_tray, also on the
board) and the demo judges with it (stage5/room.py in_tray).

    p = profile(tray)        # tray = the arena's tray config: size_m, base_size_m, rim_height_m, ...

A cone from radius base_r on the floor to radius lip_r at height h (the lip's outer edge), its wall
t thick, tilted alpha from the floor; the lip's inner edge is at radius lip_inner_r, height lip_top_z
(the bowl's highest point).
"""
from __future__ import annotations

import math


def profile(tray: dict) -> dict:
    lip_r, base_r = float(tray['size_m']) / 2, float(tray['base_size_m']) / 2
    h, t = float(tray['rim_height_m']), float(tray['rim_thickness_m'])
    if not 0 < base_r < lip_r or h <= 0:
        raise ValueError(f'bowl: need 0 < base_size_m < size_m and rim_height_m > 0 ({tray})')
    alpha = math.atan2(h, lip_r - base_r)
    return {'lip_r': lip_r, 'base_r': base_r, 'h': h, 't': t, 'alpha': alpha,
            'lip_inner_r': lip_r - t * math.sin(alpha), 'lip_top_z': h + t * math.cos(alpha)}
