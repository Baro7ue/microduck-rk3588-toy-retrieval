"""Write the plan 2C Colab training notebook for one dataset zip.

    python scripts/vision/stage2/make_colab_notebook.py ZIP_NAME ZIP_SHA256 \
        scripts/vision/stage2/train_colab.ipynb

The notebook mounts Google Drive, installs the Ultralytics version the Mac exported with, checks the
zip and the COCO weights by sha256, trains yolo11n (imgsz 320, <= 60 epochs, batch 32, seed 0,
patience 15), runs Ultralytics' own val/test and copies weights, curves and env.json back to Drive.
Markdown cells are in Chinese for the person running it.
"""
import json, sys
cells = []
def md(s): cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.strip('\n').splitlines(True)})
def code(s): cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip('\n').splitlines(True)})

md('''
# MicroDuck 阶段 2C：在 Colab 上微调 YOLO11n（6 类玩具）

计划：`claude-docs/stage2_plan_20261023/` 第 6 节。从 COCO 预训练的 `yolo11n.pt` 开始，输入 320，最多 60 轮，batch 32，seed 0，连续 15 轮不涨就停。

**运行前：**
1. 菜单「代码执行程序 → 更改运行时类型」选 **T4 GPU**。
2. 把数据集 zip 传到 Google Drive 的 `MyDrive/microduck_stage2/` 里（文件名见下面第 1 格的 `DATA_ZIP`）。
3. 「代码执行程序 → 全部运行」。第 2 格会弹出 Drive 授权，点允许。

**跑完后：** Drive 上 `MyDrive/microduck_stage2/outputs/<RUN_NAME>/` 里有 `best.pt`、`results.csv`、曲线图和 `env.json`。整个文件夹下载到 Mac 仓库的 `runs/<UTC>_s2c_train/`。预计半小时左右，T4 每小时约 2 个计算单元（以 Colab 页面为准）。
''')

code('''
# 1. Settings (change only if you renamed the zip or the Drive folder)
DRIVE_DIR = '/content/drive/MyDrive/microduck_stage2'
DATA_ZIP = '@ZIP_NAME@'
DATA_ZIP_SHA256 = '@ZIP_SHA256@'
RUN_NAME = 'yolo11n_toys_320_s0'
ULTRALYTICS = '8.3.200'          # same as the Mac export (models/yolo/yolo11n_coco/export.json)
COCO_PT_SHA256 = '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'
''')

code('''
# 2. Google Drive
from google.colab import drive
drive.mount('/content/drive')
import os
assert os.path.exists(f'{DRIVE_DIR}/{DATA_ZIP}'), f'put {DATA_ZIP} into {DRIVE_DIR} first'
''')

code('''
# 3. GPU and packages
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
!pip install -q ultralytics=={ULTRALYTICS}
import torch, ultralytics
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0))
print('ultralytics', ultralytics.__version__)
assert ultralytics.__version__ == ULTRALYTICS
''')

code('''
# 4. Dataset: check the zip, unpack to the local disk (faster than reading from Drive)
import hashlib, pathlib, shutil, subprocess

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

zip_sha = sha256(f'{DRIVE_DIR}/{DATA_ZIP}')
print('zip sha256', zip_sha)
assert zip_sha == DATA_ZIP_SHA256, 'zip differs from the one sent; upload it again'
shutil.rmtree('/content/data', ignore_errors=True)
pathlib.Path('/content/data').mkdir(parents=True)
subprocess.run(['unzip', '-q', f'{DRIVE_DIR}/{DATA_ZIP}', '-d', '/content/data'], check=True)
root = next(p.parent for p in pathlib.Path('/content/data').rglob('data.yaml'))
# Ultralytics resolves a relative "path:" against its own datasets folder, so make it absolute.
text = (root / 'data.yaml').read_text().splitlines()
text = [f'path: {root}' if line.startswith('path:') else line for line in text]
(root / 'data.yaml').write_text('\\n'.join(text) + '\\n')
print((root / 'data.yaml').read_text())
for split in ('train', 'val', 'test'):
    print(split, len(list((root / 'images' / split).glob('*.png'))), 'images')
DATA_YAML = str(root / 'data.yaml')
''')

code('''
# 5. COCO weights (Ultralytics downloads yolo11n.pt from its GitHub release v8.3.0)
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
pt_sha = sha256('yolo11n.pt')
print('yolo11n.pt sha256', pt_sha)
assert pt_sha == COCO_PT_SHA256, 'not the same COCO weights as the Mac'
''')

code('''
# 6. Train (about 30 min on a T4). Augmentation: Ultralytics defaults; hue jitter stays at the
# small default 0.015 because colour separates Henry from Thomas.
import time
t0 = time.time()
model.train(data=DATA_YAML, imgsz=320, epochs=60, batch=32, seed=0, deterministic=True,
            patience=15, cache='ram', workers=2, project='/content/runs', name=RUN_NAME,
            exist_ok=True, hsv_h=0.015, fliplr=0.5, plots=True)
train_min = (time.time() - t0) / 60
print(f'train {train_min:.1f} min')
''')

code('''
# 7. Ultralytics' own val / test numbers (for reference; 2E uses our evaluator on the test set)
run = pathlib.Path('/content/runs') / RUN_NAME
best = YOLO(str(run / 'weights' / 'best.pt'))
val = best.val(data=DATA_YAML, imgsz=320, batch=32, split='val', plots=True,
               project='/content/runs', name=RUN_NAME + '_val', exist_ok=True)
test = best.val(data=DATA_YAML, imgsz=320, batch=32, split='test', plots=True,
                project='/content/runs', name=RUN_NAME + '_test', exist_ok=True)
def per_class(m):
    return {m.names[int(c)]: round(float(ap), 4) for c, ap in zip(m.ap_class_index, m.box.ap50)}
summary = {'val': {'map50': round(float(val.box.map50), 4), 'map50_95': round(float(val.box.map), 4),
                   'ap50': per_class(val)},
           'test': {'map50': round(float(test.box.map50), 4), 'map50_95': round(float(test.box.map), 4),
                    'ap50': per_class(test)}}
print(summary)
''')

code('''
# 8. Save everything to Drive
import json, platform
out = pathlib.Path(DRIVE_DIR) / 'outputs' / RUN_NAME
shutil.rmtree(out, ignore_errors=True)
shutil.copytree(run, out)
for extra in (RUN_NAME + '_val', RUN_NAME + '_test'):
    shutil.copytree(pathlib.Path('/content/runs') / extra, out / extra)
env = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
       'ultralytics': ultralytics.__version__, 'python': platform.python_version(),
       'data_zip': DATA_ZIP, 'data_zip_sha256': zip_sha, 'coco_pt_sha256': pt_sha,
       'best_pt_sha256': sha256(run / 'weights' / 'best.pt'), 'train_min': round(train_min, 1),
       'ultralytics_metrics': summary,
       'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
(out / 'env.json').write_text(json.dumps(env, indent=1) + '\\n')
print(json.dumps(env, indent=1))
print('saved to', out)
''')

nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
      'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 0}
zip_name, zip_sha, out = sys.argv[1:4]
text = json.dumps(nb, indent=1, ensure_ascii=False).replace('@ZIP_NAME@', zip_name).replace('@ZIP_SHA256@', zip_sha)
open(out, 'w').write(text + '\n')
print(out)
