"""Decode board-received simulator frames on Mac using the established converter."""
from pathlib import Path
from probe_runtime_visual_load import save_rotated_png

root = Path('/Users/ethan/Documents/RK3588 Microduck/task/evidence/G4')
directory = sorted(root.glob('camera_transport_*'))[-1]
for suffix in ('first', 'last'):
    raw = directory / f'camera_{suffix}.uyvy'
    destination = directory / f'camera_{suffix}_rotated.png'
    save_rotated_png(raw.read_bytes(), destination)
    print(destination)
