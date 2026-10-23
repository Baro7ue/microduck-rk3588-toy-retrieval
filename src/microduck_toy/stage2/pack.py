"""Plan 2E: the evaluation pack the cloud hands to the Mac (parity) and the board (NPU runs).

    python -m microduck_toy.stage2.pack --data runs/<UTC>_s2a_dataset \
        --export runs/<UTC>_s2d_export --fp16 runs/<UTC>_s2d_rknn_fp16 --int8 runs/<UTC>_s2d_rknn_int8 \
        --cloud runs/<UTC>_s2e_cloud --out runs/<UTC>_s2e_pack

PACK/
  pack.json        what is inside, with sha256: models, splits, frame counts, parity ids
  stage2.json      the config the pack was made with (classes, thresholds, datagen)
  models/          split ONNX + export.json, the two RKNN files + their conversion.json
  data/            images/<split>/*.png and meta/<split>.jsonl for test, test_far, test_walk
  cloud/           the cloud's predictions on those splits (stage2.predict format):
                   fp32_<split>.jsonl (CPU ONNX), fp16sim_ / int8sim_<split>.jsonl (toolkit simulator)
  parity_ids.txt   the first 200 test frames (seeds 300000-300199): the Mac renders these again
  cpp/             camera payloads of --cpp-n test frames for the C++ benchmark (cpp_bench.make;
                   needs MuJoCo, so the pack is made in the cloud)

--cloud is a folder with fp32_<split>.jsonl; the simulator files come from the RKNN folders'
pred_<split>.jsonl. Every split must be present everywhere.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

from microduck_toy.paths import repo_root

SPLITS = ('test', 'test_far', 'test_walk')
PARITY_N = 200


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(data: Path, export: Path, fp16: Path, int8: Path, cloud: Path, out: Path,
          cpp_n: int = 100) -> dict:
    out.mkdir(parents=True)
    models = out / 'models'
    models.mkdir()
    split_onnx = next(export.glob('*_split_320.onnx'))
    files = {'split_onnx': split_onnx, 'export_json': export / 'export.json'}
    for tag, folder in (('fp16', fp16), ('int8', int8)):
        files[f'rknn_{tag}'] = next(folder.glob('*.rknn'))
        files[f'conversion_{tag}'] = folder / 'conversion.json'
    for key, src in files.items():
        name = src.name if not key.startswith('conversion') else f'{key}.json'
        shutil.copy2(src, models / name)
        files[key] = models / name
    (out / 'cloud').mkdir()
    counts = {}
    for split in SPLITS:
        meta = data / 'meta' / f'{split}.jsonl'
        ids = [json.loads(x)['id'] for x in meta.read_text().splitlines()]
        (out / 'data' / 'meta').mkdir(parents=True, exist_ok=True)
        shutil.copy2(meta, out / 'data' / 'meta' / meta.name)
        shutil.copytree(data / 'images' / split, out / 'data' / 'images' / split)
        for tag, src in (('fp32', cloud / f'fp32_{split}.jsonl'), ('fp16sim', fp16 / f'pred_{split}.jsonl'),
                         ('int8sim', int8 / f'pred_{split}.jsonl')):
            got = {json.loads(x)['id'] for x in src.read_text().splitlines()}
            if got != set(ids):
                raise SystemExit(f'{src} does not cover {split} ({len(got)} vs {len(ids)} frames)')
            shutil.copy2(src, out / 'cloud' / f'{tag}_{split}.jsonl')
        counts[split] = len(ids)
    test_ids = [json.loads(x)['id'] for x in (data / 'meta' / 'test.jsonl').read_text().splitlines()]
    (out / 'parity_ids.txt').write_text('\n'.join(test_ids[:PARITY_N]) + '\n')
    shutil.copy2(repo_root() / 'configs' / 'vision' / 'stage2.json', out / 'stage2.json')
    cpp = None
    if cpp_n:
        from microduck_toy.stage2 import cpp_bench
        cpp = cpp_bench.make(data, cpp_n, out / 'cpp')
    dataset = json.loads((data / 'dataset.json').read_text())
    info = {'created_utc': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'dataset': str(data), 'dataset_config_sha256': dataset.get('config_sha256'),
            'frames': counts, 'parity': {'split': 'test', 'n': PARITY_N}, 'cpp': cpp,
            'models': {k: {'file': f'models/{v.name}', 'sha256': sha256(v)} for k, v in files.items()},
            'source_pt_sha256': json.loads(files['export_json'].read_text()).get('weights_sha256')}
    (out / 'pack.json').write_text(json.dumps(info, indent=1) + '\n')
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for name in ('data', 'export', 'fp16', 'int8', 'cloud', 'out'):
        ap.add_argument(f'--{name}', type=Path, required=True)
    ap.add_argument('--cpp-n', type=int, default=100, help='C++ benchmark frames (0: none)')
    args = ap.parse_args(argv)
    info = build(args.data, args.export, args.fp16, args.int8, args.cloud, args.out, args.cpp_n)
    print(json.dumps({'frames': info['frames'], 'models': {k: v['sha256'][:12]
                                                          for k, v in info['models'].items()}}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
