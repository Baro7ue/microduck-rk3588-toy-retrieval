"""Run the original Mac viewer with optional IMU-relative heading feedback."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

from heading_hold import HeadingConfig, make_heading_policy


ROOT = Path('/Users/ethan/Documents/RK3588 Microduck')
REPO = ROOT / 'task/microduck_r1'
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--heading-kp', type=float, default=1.0)
parser.add_argument('--heading-ki', type=float, default=0.0)
args, remaining = parser.parse_known_args()
if '--raw-accelerometer' in remaining:
    parser.error('Heading hold requires projected gravity; raw acceleration is unsupported')
try:
    config = HeadingConfig(kp=args.heading_kp, ki=args.heading_ki)
except ValueError as exc:
    parser.error(str(exc))
os.chdir(REPO)
spec = importlib.util.spec_from_file_location('upstream_replay', REPO / 'scripts/infer_policy.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
replay.PolicyInference = make_heading_policy(replay.PolicyInference, config)
sys.argv = [str(REPO / 'scripts/infer_policy.py'), *remaining]
replay.main()
