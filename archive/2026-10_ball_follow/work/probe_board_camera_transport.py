"""Receive only the Mac simulator's camera on the board; no body or robot RPC."""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback

from probe_runtime_visual_load import FrameWorker, interrupt, wait_until

ADDRESS = ('192.168.50.1', 7901)
ROOT = Path.home() / 'microduck-board' / 'run' / 'G4'


def main(duration: float = 30.0, address: tuple[str, int] = ADDRESS,
         root: Path = ROOT) -> int:
    started = dt.datetime.now(dt.timezone.utc)
    directory = Path(root) / started.strftime('camera_transport_%Y%m%d_%H%M%S_%fZ')
    directory.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'started_wall_utc': started.isoformat(),
              'duration_requested_s': duration, 'evidence_directory': str(directory),
              'scope': 'camera TCP receive only; no body connection or robot RPC'}
    worker = FrameWorker(address, directory)
    errors = []
    code = 0
    print(f'只读接收相机 {address[0]}:{address[1]}，持续 {duration:g} 秒。', flush=True)
    previous_sigterm = signal.signal(signal.SIGTERM, interrupt)
    try:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('duration must be finite and positive')
        worker.start()
        wait_until(time.monotonic() + duration, worker)
    except KeyboardInterrupt:
        code = 130
        errors.append(traceback.format_exc())
    except Exception:
        code = 1
        errors.append(traceback.format_exc())
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        try:
            if worker.ident is not None:
                worker.stop()
            worker.raise_if_failed()
            if code == 0 and not worker.records:
                raise RuntimeError('no complete camera frames received')
        except Exception:
            code = 130 if code == 130 else 1
            errors.append(traceback.format_exc())
        camera = worker.summary([])
        camera['png_generated'] = False
        camera['png_rotation_note'] = '90 degrees clockwise is a future conversion; no PNG was generated'
        report['camera'] = camera
        report['outcome'] = {0: 'completed', 130: 'interrupted'}.get(code, 'error')
        report['finished_wall_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        if errors:
            report['error'] = '\n'.join(errors)
            (directory / 'error.txt').write_text(report['error'], encoding='utf-8')
            print(report['error'], flush=True)
        (directory / 'summary.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'结果：{report["outcome"]}；frame_count={camera["frame_count"]}；'
              f'transport_fps={camera["transport_fps"]}；'
              f'interarrival_fps={camera["interarrival_fps"]}', flush=True)
        print(f'证据目录：{directory}', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
