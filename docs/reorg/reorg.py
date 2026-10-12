#!/usr/bin/env python3
"""Move the "RK3588 Microduck" folder into the repository layout, with a log and an undo.

Usage (from any directory; the folder root is two levels above this file):
  python3 docs/reorg/reorg.py plan            dry run: every action and whether it can run
  python3 docs/reorg/reorg.py apply P1        run one phase (P1, P2, P3, P4, in order)
  python3 docs/reorg/reorg.py undo P2         reverse one phase, newest action first
  python3 docs/reorg/reorg.py check           every file in the P1 snapshot is unchanged at its new path
  python3 docs/reorg/reorg.py manifest        write MANIFEST.tsv in each local-only directory
  python3 docs/reorg/reorg.py status          which phases are applied

Nothing is ever overwritten. Each action is appended to docs/reorg/move_log.tsv as soon
as it has happened, so a phase that stops halfway can be fixed and re-run, or undone.
Only caches (.DS_Store, __pycache__) are deleted, in P1; that is the one step undo cannot reverse.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

PHASES = ('P1', 'P2', 'P3', 'P4')
ARCH = 'archive/2026-10_ball_follow'
TOOL_DIR = 'docs/reorg'
STAGED = TOOL_DIR + '/staged'
CACHE_FILES = ('.DS_Store',)
CACHE_DIRS = ('__pycache__',)
SKIP_WALK = ('.git',)
UPSTREAM_RL = 'third_party/microduck_rl/src/mjlab_microduck/robot/microduck'
YOLO = 'models/yolo/yolo11n_coco'

GUIDES = {
    'G3_10分钟持续运行与资源采样_20261007.md': '2026-10-07_g3_10min_run_resource_sampling.md',
    'G3_CPU策略推理耗时测量_20261007.md': '2026-10-07_g3_cpu_policy_inference_timing.md',
    'G3_反馈采样修复与重新验收_20261007.md': '2026-10-07_g3_feedback_sampling_fix_reacceptance.md',
    'G3_断联恢复与60秒行走_20261007.md': '2026-10-07_g3_disconnect_recovery_60s_walk.md',
    'G4_YOLO11n板端CPU单图检测_20261008.md': '2026-10-08_g4_yolo11n_board_cpu_single_image.md',
    'G4_一格距离停车_20261009.md': '2026-10-09_g4_one_cell_distance_stop.md',
    'G4_三维球相机正样本_20261008.md': '2026-10-08_g4_3d_ball_camera_positives.md',
    'G4_固定面积停车_20261009.md': '2026-10-09_g4_fixed_area_stop.md',
    'G4_地面反射对照与板端复核_20261008.md': '2026-10-08_g4_floor_reflection_control_board_recheck.md',
    'G4_板端接收仿真相机画面_20261007.md': '2026-10-07_g4_board_receives_sim_camera.md',
    'G4_相机连续CPU检测_20261008.md': '2026-10-08_g4_camera_continuous_cpu_detection.md',
    'G4_站立策略与相机检测联合检查_20261008.md': '2026-10-08_g4_stand_policy_with_camera_detection.md',
    'G4_统一启动与录屏_20261008.md': '2026-10-08_g4_unified_launch_and_recording.md',
    'G4_视觉接近与停止_20261008.md': '2026-10-08_g4_visual_approach_and_stop.md',
    'G4_转向参考修正与验收_20261010.md': '2026-10-10_g4_steering_reference_fix.md',
    'G4_速度对照_20261008.md': '2026-10-08_g4_speed_comparison.md',
}
OUTPUTS = {
    'MicroDuck左偏绕圈诊断_20261004.md': 'docs/analyses/2026-10-04_left_drift_circling_diagnosis.md',
    'MicroDuck航向保持_5分钟对照_20261004.png': 'docs/analyses/figures/2026-10-04_heading_hold_5min_comparison.png',
    'MicroDuck航向保持对照_20261004.png': 'docs/analyses/figures/2026-10-04_heading_hold_comparison.png',
    'MicroDuck航向保持对照_20261004.svg': 'docs/analyses/figures/2026-10-04_heading_hold_comparison.svg',
    'MicroDuck航向保持说明_20261004.md': 'docs/analyses/2026-10-04_heading_hold_notes.md',
    'MicroDuck训练平台期分析_20261004.ipynb': 'docs/analyses/2026-10-04_training_plateau_analysis.ipynb',
    'MicroDuck训练平台期分析_20261004.json': 'docs/analyses/2026-10-04_training_plateau_analysis.json',
    'MicroDuck训练平台期分析_20261004.md': 'docs/analyses/2026-10-04_training_plateau_analysis.md',
    'MicroDuck训练曲线分析_20261004.png': 'docs/analyses/figures/2026-10-04_training_curves.png',
    'MicroDuck训练曲线分析_20261004.svg': 'docs/analyses/figures/2026-10-04_training_curves.svg',
    'WandB训练时长核对_20261004.ipynb': 'docs/analyses/2026-10-04_wandb_training_duration_check.ipynb',
    'WandB训练时长核对_20261004.json': 'docs/analyses/2026-10-04_wandb_training_duration_check.json',
    '任务卡1_Mac串口启动日志_逐步教程.md': 'docs/guides/2026-10-03_task_card1_mac_serial_boot_log.md',
    '任务卡1_Mac串口启动日志_逐步教程.pdf': 'references/ai_notes/gpt_outputs/任务卡1_Mac串口启动日志_逐步教程.pdf',
}
VENDOR = ('01、程序源码', '02、开发板原理图', '03、核心板资料', '06、参考资料',
          '07、硬件资料', '08、RK官方文档', '10、用户手册')
RESEARCH = 'research_notes/MicroDuck 寻球 导航方案'

Rule = Tuple[str, str, Optional[str], Optional[str]]
RULES: List[Rule] = [
    # P1 is built in: clean caches, snapshot, install docs/reorg/staged into place.
    *[('P2', 'move', f'task/evidence/{g}', f'evidence/{g}')
      for g in ('G1', 'G2', 'G3', 'G4', 'walk_drift_20261004')],
    ('P2', 'move_rest', 'task/evidence', 'evidence/training_20261003'),
    ('P2', 'rmdir', 'task/evidence', None),
    ('P2', 'move', 'task1/evidence', 'evidence/task1_serial_boot'),
    ('P2', 'rmdir', 'task1', None),
    *[('P2', 'move', f'output/guides/{old}', f'docs/guides/{new}') for old, new in GUIDES.items()],
    ('P2', 'move', 'output/pdf/MicroDuck_下一阶段行动指南_20261004.md',
     'docs/analyses/2026-10-04_next_stage_action_guide.md'),
    ('P2', 'move_rest', 'output/pdf', 'references/ai_notes/gpt_outputs'),
    *[('P2', 'rmdir', d, None) for d in ('output/guides', 'output/pdf', 'output')],
    *[('P2', 'move', f'outputs/{old}', new) for old, new in OUTPUTS.items()],
    ('P2', 'rmdir', 'outputs', None),
    ('P2', 'move', 'reports/MicroDuck 寻球 导航方案.md', 'docs/plans/2026-10-10_ball_search_navigation_plan.md'),
    ('P2', 'move', 'reports/MicroDuck_玩具回收总体方案_Claude评估版_2026-10-11.md',
     'docs/plans/2026-10-11_toy_retrieval_plan_for_review.md'),
    ('P2', 'rmdir', 'reports', None),
    *[('P2', 'move', f'{RESEARCH}/{name}', f'references/research_ball_navigation/{name}')
      for name in ('bhuman_tree.json', 'bit-bots_tree.json')],
    ('P2', 'move', RESEARCH, 'docs/research/ball_navigation'),
    ('P2', 'rmdir', 'research_notes', None),
    ('P2', 'move_rest', 'learning', 'scripts/learning'),
    ('P2', 'rmdir', 'learning', None),
    *[('P2', 'move', f'参考资料/{name}', f'references/vendor_alientek/{name}') for name in VENDOR],
    ('P2', 'move_rest', '参考资料', 'references/ai_notes'),
    ('P2', 'rmdir', '参考资料', None),
    ('P2', 'move', 'tmp/pdfs', f'{ARCH}/pdf_build/tmp_pdfs'),
    ('P2', 'rmdir', 'tmp', None),
    ('P3', 'move', 'task/microduck', 'third_party/microduck'),
    ('P3', 'move', 'task/microduck_r1', 'third_party/microduck_rl'),
    ('P3', 'move', 'third_party/microduck_rl/.venv', 'third_party/microduck_rl/.venv.pre_reorg'),
    ('P3', 'move', 'task/policies', 'models/policies'),
    ('P3', 'move', 'task/checkpoints/qktrrkg5', 'models/checkpoints/walk_qktrrkg5'),
    ('P3', 'rmdir', 'task/checkpoints', None),
    ('P3', 'rmdir', 'task', None),
    ('P3', 'symlink', '../../../../' + UPSTREAM_RL + '/assets', 'sim/scenes/legacy_ball/ball_1m_yellow/assets'),
    *[('P4', 'copy', f'work/vision-offline/{src}', f'{YOLO}/{dst}') for src, dst in (
        ('yolo11n-cpu/models/yolo11n.pt', 'yolo11n_coco.pt'),
        ('yolo11n-cpu/models/yolo11n.onnx', 'yolo11n_coco_fp32_320.onnx'),
        ('yolo11n-cpu/models/export.json', 'export.json'),
        ('yolo11n-cpu-npu/models/yolo11n-rk3588-fp.rknn', 'yolo11n_coco_fp16_320.rknn'),
        ('yolo11n-cpu-npu/models/conversion.json', 'conversion.json'),
    )],
    *[('P4', 'move', f'work/{name}', f'offline/{name}')
      for name in ('g3-offline', 'npu-offline', 'vision-export', 'flash_tools')],
    ('P4', 'move', 'work/vision-offline', f'{ARCH}/vision-offline'),
    ('P4', 'move', 'work/task1_pdf_render', f'{ARCH}/pdf_build/task1_pdf_render'),
    ('P4', 'move', 'work', f'{ARCH}/work'),
]
# Upstream git repositories: moved as a whole, never walked, cleaned or hashed.
OPAQUE = ('task/microduck', 'task/microduck_r1', 'third_party/microduck', 'third_party/microduck_rl')
TOP_LEVEL = {
    '.git', '.gitignore', '.gitattributes', '.gitmodules', 'README.md', 'PROVENANCE.md',
    'pyproject.toml', 'env.sh', 'src', 'scripts', 'configs', 'sim', 'tests', 'docs', 'results',
    'report', 'third_party', 'models', 'data', 'runs', 'evidence', 'offline', 'references', 'archive',
}
LOCAL_ONLY = ('evidence', 'offline', 'references', 'models', f'{ARCH}/vision-offline',
              f'{ARCH}/pdf_build', 'data', 'runs')
BUSY_PATTERN = 'mjpython|robotd|body_server|tofd'


class ReorgError(RuntimeError):
    pass


class Row(NamedTuple):
    phase: str
    op: str
    src: str
    dst: Optional[str]
    status: str


class Record(NamedTuple):
    id: int
    ts: str
    phase: str
    op: str
    src: str
    dst: str
    ref: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def link_kind(path: Path) -> str:
    """Snapshot kind of a symlink: L if it resolves, B if it was already dangling."""
    return 'L' if os.path.exists(path) else 'B'


def is_cache(name: str, is_dir: bool) -> bool:
    return name in CACHE_DIRS if is_dir else name in CACHE_FILES


def under(path: str, prefixes: Iterable[str]) -> bool:
    return any(path == p or path.startswith(p + '/') for p in prefixes)


def parent(path: str) -> str:
    return path.rsplit('/', 1)[0] if '/' in path else ''


MANIFEST_HEADER = 'path\tbytes\tsha256'


def is_tool_manifest(path: Path) -> bool:
    """A MANIFEST.tsv written by `manifest` (regenerable, so undo may remove it)."""
    if path.name != 'MANIFEST.tsv' or path.is_symlink() or not path.is_file():
        return False
    with open(path, encoding='utf-8', errors='replace') as handle:
        return handle.readline().rstrip('\n') == MANIFEST_HEADER


def map_through(path: str, moves: Iterable[Tuple[str, str]]) -> str:
    """Where `path` ends up after the given (src, dst) moves, applied in order."""
    for src, dst in moves:
        if path == src or path.startswith(src + '/'):
            path = dst + path[len(src):]
    return path


class RealFS:
    """Acts on disk. Paths are relative to root, with '/' separators."""

    virtual = False

    def __init__(self, root: Path, opaque: Tuple[str, ...]):
        self.root = root
        self.opaque = opaque

    def p(self, rel: str) -> Path:
        return self.root / rel

    def lexists(self, rel: str) -> Optional[bool]:
        return os.path.lexists(self.p(rel))

    def isdir(self, rel: str) -> bool:
        path = self.p(rel)
        return path.is_dir() and not path.is_symlink()

    def children(self, rel: str) -> List[str]:
        return sorted(os.listdir(self.p(rel))) if self.isdir(rel) else []

    def files_under(self, rel: str) -> List[str]:
        out = []
        for base, dirs, files in os.walk(self.p(rel)):
            for name in sorted(dirs):
                if (Path(base) / name).is_symlink():
                    files.append(name)
            dirs[:] = sorted(d for d in dirs if not (Path(base) / d).is_symlink())
            for name in sorted(files):
                out.append((Path(base) / name).relative_to(self.p(rel)).as_posix())
        return out

    def rename(self, src: str, dst: str) -> None:
        os.rename(self.p(src), self.p(dst))

    def copy(self, src: str, dst: str) -> None:
        shutil.copy2(self.p(src), self.p(dst))

    def mkdir(self, rel: str) -> None:
        os.mkdir(self.p(rel))

    def rmdir(self, rel: str) -> None:
        os.rmdir(self.p(rel))

    def unlink(self, rel: str) -> None:
        os.unlink(self.p(rel))

    def symlink(self, target: str, link: str) -> None:
        os.symlink(target, self.p(link))

    def remove_tree(self, rel: str) -> None:
        shutil.rmtree(self.p(rel))

    def same_file(self, a: str, b: str) -> bool:
        return sha256(self.p(a)) == sha256(self.p(b))


class VirtualFS:
    """In-memory copy of the folder listing, so `plan` can simulate every phase."""

    virtual = True

    def __init__(self, root: Path, opaque: Tuple[str, ...]):
        self.opaque_set = set()
        self.kind: Dict[str, str] = {}
        for base, dirs, files in os.walk(root):
            rel_base = Path(base).relative_to(root).as_posix()
            rel_base = '' if rel_base == '.' else rel_base
            keep = []
            for name in sorted(dirs):
                rel = f'{rel_base}/{name}' if rel_base else name
                if name in SKIP_WALK and not rel_base:
                    continue
                if (Path(base) / name).is_symlink():
                    self.kind[rel] = 'link'
                elif rel in opaque:
                    self.kind[rel] = 'opaque'
                else:
                    self.kind[rel] = 'dir'
                    keep.append(name)
            dirs[:] = keep
            for name in files:
                rel = f'{rel_base}/{name}' if rel_base else name
                self.kind[rel] = 'link' if (Path(base) / name).is_symlink() else 'file'

    def _inside_opaque(self, rel: str) -> bool:
        probe = parent(rel)
        while probe:
            if self.kind.get(probe) == 'opaque':
                return True
            probe = parent(probe)
        return False

    def lexists(self, rel: str) -> Optional[bool]:
        if rel in self.kind:
            return True
        return None if self._inside_opaque(rel) else False

    def isdir(self, rel: str) -> bool:
        return self.kind.get(rel) in ('dir', 'opaque')

    def children(self, rel: str) -> List[str]:
        prefix = rel + '/' if rel else ''
        return sorted(k[len(prefix):] for k in self.kind if k.startswith(prefix) and '/' not in k[len(prefix):])

    def files_under(self, rel: str) -> List[str]:
        prefix = rel + '/'
        return sorted(k[len(prefix):] for k, v in self.kind.items()
                      if k.startswith(prefix) and v in ('file', 'link'))

    def rename(self, src: str, dst: str) -> None:
        moved = {k: v for k, v in self.kind.items() if k == src or k.startswith(src + '/')}
        for k in moved:
            del self.kind[k]
        for k, v in moved.items():
            self.kind[dst + k[len(src):]] = v

    def copy(self, src: str, dst: str) -> None:
        self.kind[dst] = 'file'

    def mkdir(self, rel: str) -> None:
        self.kind[rel] = 'dir'

    def rmdir(self, rel: str) -> None:
        del self.kind[rel]

    def unlink(self, rel: str) -> None:
        del self.kind[rel]

    def symlink(self, target: str, link: str) -> None:
        self.kind[link] = 'link'

    def remove_tree(self, rel: str) -> None:
        for k in [k for k in self.kind if k == rel or k.startswith(rel + '/')]:
            del self.kind[k]

    def same_file(self, a: str, b: str) -> bool:
        return True


class Reorg:
    def __init__(self, root: Path, rules: List[Rule] = RULES, opaque: Iterable[str] = OPAQUE,
                 top_level: Iterable[str] = TOP_LEVEL):
        self.root = Path(root).resolve()
        self.rules = list(rules)
        self.opaque = tuple(opaque)
        self.top_level = set(top_level)
        self.dir = self.root / TOOL_DIR
        self.log_path = self.dir / 'move_log.tsv'
        self.plan_path = self.dir / 'move_plan.tsv'
        self.snapshot_path = self.dir / 'snapshot_before.tsv'

    # ----- log -----
    def records(self) -> List[Record]:
        if not self.log_path.exists():
            return []
        out = []
        for line in self.log_path.read_text(encoding='utf-8').splitlines()[1:]:
            fields = line.split('\t')
            out.append(Record(int(fields[0]), *fields[1:7]))
        return out

    def effective(self) -> List[Record]:
        records = self.records()
        undone = {int(r.ref) for r in records if r.op == 'undo'}
        return [r for r in records if r.op != 'undo' and r.id not in undone]

    def _log(self, phase: str, op: str, src: str = '', dst: str = '', ref: str = '') -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        new = not self.log_path.exists()
        next_id = len(self.records()) + 1
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        with open(self.log_path, 'a', encoding='utf-8') as handle:
            if new:
                handle.write('id\tts\tphase\top\tsrc\tdst\tref\n')
            handle.write(f'{next_id}\t{stamp}\t{phase}\t{op}\t{src}\t{dst}\t{ref}\n')
            handle.flush()
            os.fsync(handle.fileno())

    def phase_state(self, phase: str) -> str:
        ops = {r.op for r in self.effective() if r.phase == phase}
        if 'end' in ops:
            return 'applied'
        return 'partial' if 'begin' in ops else 'not applied'

    # ----- actions shared by apply and plan -----
    def _run_phase(self, phase: str, fs, rows: Optional[List[Row]]) -> None:
        """Run (or, on a VirtualFS, simulate) one phase. Errors raise on disk, become rows in a plan."""
        done = {(r.op, r.src, r.dst) for r in self.effective() if r.phase == phase}

        def emit(op: str, src: str, dst: Optional[str], status: str) -> None:
            if rows is not None:
                rows.append(Row(phase, op, src, dst, status))
            if status.startswith('error') and not fs.virtual:
                raise ReorgError(f'{phase} {op} {src} -> {dst}: {status}')

        def ensure_parents(rel: str) -> None:
            missing = []
            probe = parent(rel)
            while probe and fs.lexists(probe) is False:
                missing.append(probe)
                probe = parent(probe)
            for path in reversed(missing):
                fs.mkdir(path)
                if not fs.virtual:
                    self._log(phase, 'mkdir', path)

        def move(src: str, dst: str) -> None:
            if ('move', src, dst) in done:
                emit('move', src, dst, 'done')
                return
            src_ok, dst_ok = fs.lexists(src), fs.lexists(dst)
            if src_ok is None or dst_ok is None:
                emit('move', src, dst, 'ok (inside an upstream repository, checked when applied)')
                return
            if not src_ok:
                emit('move', src, dst, 'error: source missing')
                return
            if dst_ok:
                emit('move', src, dst, 'error: destination already exists')
                return
            ensure_parents(dst)
            fs.rename(src, dst)
            if not fs.virtual:
                self._log(phase, 'move', src, dst)
            emit('move', src, dst, 'ok')

        def rmdir(rel: str, op: str = 'rmdir') -> None:
            if (op, rel, '') in done or fs.lexists(rel) is False:
                emit(op, rel, None, 'done' if (op, rel, '') in done else 'ok (already absent)')
                return
            left = [c for c in fs.children(rel) if c not in CACHE_FILES]
            if left:
                emit(op, rel, None, 'error: not empty: ' + ', '.join(left[:5]))
                return
            for cache in fs.children(rel):
                fs.unlink(f'{rel}/{cache}')
                if not fs.virtual:
                    self._log(phase, 'delete', f'{rel}/{cache}')
            fs.rmdir(rel)
            if not fs.virtual:
                self._log(phase, op, rel)
            emit(op, rel, None, 'ok')

        if phase == 'P1':
            removed = self._clean(fs, phase)
            emit('clean', '.DS_Store and __pycache__', None, f'ok ({removed} removed)')
            if not fs.virtual and ('snapshot', '', '') not in done:
                count = self._snapshot()
                self._log(phase, 'snapshot')
                emit('snapshot', str(self.snapshot_path.relative_to(self.root)), None, f'ok ({count} files)')
            staged = STAGED
            for rel in (fs.files_under(staged) if fs.isdir(staged) else []):
                move(f'{staged}/{rel}', rel)
            # Remove the now-empty staged tree, deepest directory first.
            if fs.isdir(staged):
                dirs = []
                stack = [staged]
                while stack:
                    current = stack.pop()
                    dirs.append(current)
                    stack.extend(f'{current}/{c}' for c in fs.children(current) if fs.isdir(f'{current}/{c}'))
                for rel in sorted(dirs, key=lambda d: d.count('/'), reverse=True):
                    rmdir(rel)

        for rule_phase, op, src, dst in self.rules:
            if rule_phase != phase:
                continue
            if op == 'move':
                move(src, dst)
            elif op == 'move_rest':
                for child in (fs.children(src) if fs.isdir(src) else []):
                    if child not in CACHE_FILES:
                        move(f'{src}/{child}', f'{dst}/{child}')
            elif op == 'rmdir':
                rmdir(src)
            elif op == 'copy':
                if ('copy', src, dst) in done:
                    emit('copy', src, dst, 'done')
                elif not fs.lexists(src):
                    emit('copy', src, dst, 'error: source missing')
                elif fs.lexists(dst):
                    emit('copy', src, dst, 'error: destination already exists')
                else:
                    ensure_parents(dst)
                    fs.copy(src, dst)
                    if not fs.virtual:
                        self._log(phase, 'copy', src, dst)
                    emit('copy', src, dst, 'ok')
            elif op == 'symlink':
                if ('symlink', src, dst) in done:
                    emit('symlink', dst, src, 'done')
                elif fs.lexists(dst):
                    emit('symlink', dst, src, 'error: link path already exists')
                else:
                    ensure_parents(dst)
                    fs.symlink(src, dst)
                    if not fs.virtual:
                        self._log(phase, 'symlink', src, dst)
                    emit('symlink', dst, src, 'ok')
            else:
                raise ReorgError(f'unknown rule {op}')

    def _clean(self, fs, phase: str) -> int:
        removed = 0
        if fs.virtual:
            for rel in [k for k, v in fs.kind.items()
                        if is_cache(k.rsplit('/', 1)[-1], v == 'dir')]:
                if rel in fs.kind:
                    fs.remove_tree(rel)
                    removed += 1
            return removed
        for base, dirs, files in os.walk(self.root):
            rel_base = Path(base).relative_to(self.root).as_posix()
            rel_base = '' if rel_base == '.' else rel_base
            keep = []
            for name in dirs:
                rel = f'{rel_base}/{name}' if rel_base else name
                if (Path(base) / name).is_symlink() or under(rel, self.opaque) or (not rel_base and name in SKIP_WALK):
                    continue
                if is_cache(name, True):
                    fs.remove_tree(rel)
                    self._log(phase, 'delete', rel)
                    removed += 1
                else:
                    keep.append(name)
            dirs[:] = keep
            for name in files:
                rel = f'{rel_base}/{name}' if rel_base else name
                if is_cache(name, False):
                    fs.unlink(rel)
                    self._log(phase, 'delete', rel)
                    removed += 1
        return removed

    def _walk_snapshot(self):
        for base, dirs, files in os.walk(self.root):
            rel_base = Path(base).relative_to(self.root).as_posix()
            rel_base = '' if rel_base == '.' else rel_base
            keep = []
            for name in sorted(dirs):
                rel = f'{rel_base}/{name}' if rel_base else name
                if under(rel, self.opaque) or under(rel, (TOOL_DIR,)) or (not rel_base and name in SKIP_WALK):
                    continue
                if (Path(base) / name).is_symlink():
                    yield rel, link_kind(Path(base) / name), 0, os.readlink(Path(base) / name)
                else:
                    keep.append(name)
            dirs[:] = keep
            for name in sorted(files):
                if name in CACHE_FILES:
                    continue
                path = Path(base) / name
                rel = f'{rel_base}/{name}' if rel_base else name
                if path.is_symlink():
                    yield rel, link_kind(path), 0, os.readlink(path)
                else:
                    yield rel, 'F', path.stat().st_size, sha256(path)

    def _snapshot(self) -> int:
        count = 0
        tmp = self.snapshot_path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as handle:
            handle.write('path\tkind\tbytes\tsha256_or_link\n')
            for rel, kind, size, value in self._walk_snapshot():
                handle.write(f'{rel}\t{kind}\t{size}\t{value}\n')
                count += 1
        os.replace(tmp, self.snapshot_path)
        return count

    # ----- commands -----
    def _check_order(self, phase: str, undo: bool) -> None:
        if phase not in PHASES:
            raise ReorgError(f'unknown phase {phase}; use one of {", ".join(PHASES)}')
        index = PHASES.index(phase)
        later = [p for p in PHASES[index + 1:] if self.phase_state(p) != 'not applied']
        if later:
            raise ReorgError(f'{", ".join(later)} must be undone first')
        if undo:
            if self.phase_state(phase) == 'not applied':
                raise ReorgError(f'{phase} has not been applied')
            return
        earlier = [p for p in PHASES[:index] if self.phase_state(p) != 'applied']
        if earlier:
            raise ReorgError(f'apply {", ".join(earlier)} first')
        if self.phase_state(phase) == 'applied':
            raise ReorgError(f'{phase} is already applied')

    def busy_processes(self) -> List[str]:
        try:
            result = subprocess.run(['pgrep', '-fl', BUSY_PATTERN], capture_output=True, text=True)
        except FileNotFoundError:
            return []
        return [line for line in result.stdout.splitlines() if 'reorg.py' not in line]

    def _check_idle(self, phase: str, force: bool) -> None:
        # From P2 on, files that running launchers write to (task/evidence, task/..., work/...) move.
        if phase != 'P1' and not force:
            busy = self.busy_processes()
            if busy:
                raise ReorgError('stop the simulator / robotd first (or pass --force):\n  ' + '\n  '.join(busy))

    def apply(self, phase: str, force: bool = False) -> List[Row]:
        self._check_order(phase, undo=False)
        self._check_idle(phase, force)
        if self.phase_state(phase) == 'not applied':
            self._log(phase, 'begin')
        rows: List[Row] = []
        self._run_phase(phase, RealFS(self.root, self.opaque), rows)
        self._log(phase, 'end')
        return rows

    def _undo_steps(self, phase: str) -> List[Tuple[Record, List[Tuple[str, str]]]]:
        """Each action of the phase, newest first, with the moves that came after it.

        Mapping a path through the later moves gives where that path is on disk now.
        """
        records = [r for r in self.effective() if r.phase == phase]
        steps = []
        for index, record in enumerate(records):
            later = [(r.src, r.dst) for r in records[index + 1:] if r.op == 'move']
            steps.append((record, later))
        return list(reversed(steps))

    def _undo_blockers(self, steps) -> List[str]:
        """Everything that would stop the undo, found before anything is changed."""
        fs = RealFS(self.root, self.opaque)
        blockers = []
        placed = set()      # paths (as on disk now) that undo will move, delete or unlink
        made = set()        # directories this phase created (as on disk now)
        for record, later in steps:
            if record.op in ('move', 'copy', 'symlink'):
                placed.add(map_through(record.dst, later))
            elif record.op == 'mkdir':
                made.add(map_through(record.src, later))
        for record, later in steps:
            if record.op == 'move':
                src, dst = map_through(record.src, later), map_through(record.dst, later)
                if not fs.lexists(dst):
                    blockers.append(f'{dst}: missing, cannot move it back to {record.src}')
                if fs.lexists(src):
                    blockers.append(f'{src}: exists again; move it elsewhere first (it would be overwritten '
                                    f'when {record.dst} moves back)')
            elif record.op == 'copy':
                src, dst = map_through(record.src, later), map_through(record.dst, later)
                if not fs.lexists(dst):
                    continue
                if not fs.lexists(src) or os.path.islink(fs.p(dst)) or not fs.same_file(src, dst):
                    blockers.append(f'{dst}: changed since it was copied from {record.src}; '
                                    'move it elsewhere first so undo does not delete it')
            elif record.op == 'symlink':
                dst = map_through(record.dst, later)
                if fs.lexists(dst) and (not os.path.islink(fs.p(dst)) or os.readlink(fs.p(dst)) != record.src):
                    blockers.append(f'{dst}: no longer the link the reorganisation made')
            elif record.op == 'rmdir':
                path = map_through(record.src, later)
                if fs.lexists(path):
                    blockers.append(f'{path}: exists again; undo needs to recreate it')
            elif record.op == 'mkdir':
                path = map_through(record.src, later)
                for name in fs.children(path):
                    rel = f'{path}/{name}'
                    full = fs.p(rel)
                    if rel in placed or rel in made or is_cache(name, full.is_dir() and not full.is_symlink()) \
                            or is_tool_manifest(full):
                        continue
                    blockers.append(f'{rel}: not placed there by the reorganisation; move it out of {path} first')
        return blockers

    def undo(self, phase: str, force: bool = False) -> int:
        self._check_order(phase, undo=True)
        self._check_idle(phase, force)
        steps = self._undo_steps(phase)
        blockers = self._undo_blockers(steps)
        if blockers:
            raise ReorgError(f'{phase} not undone, nothing changed:\n  ' + '\n  '.join(blockers))
        fs = RealFS(self.root, self.opaque)
        # Manifests are regenerable output of `manifest`; drop those inside what this phase placed.
        for record, later in steps:
            where = record.dst if record.op == 'move' else record.src if record.op == 'mkdir' else None
            if where is not None and is_tool_manifest(fs.p(map_through(where, later)) / 'MANIFEST.tsv'):
                (fs.p(map_through(where, later)) / 'MANIFEST.tsv').unlink()
        count = 0
        for record, _ in steps:
            if record.op == 'move':
                fs.rename(record.dst, record.src)
            elif record.op == 'copy':
                if fs.lexists(record.dst):
                    fs.unlink(record.dst)
            elif record.op == 'mkdir':
                for name in fs.children(record.src):
                    full = fs.p(f'{record.src}/{name}')
                    if is_cache(name, True) and full.is_dir() and not full.is_symlink():
                        fs.remove_tree(f'{record.src}/{name}')
                    elif is_cache(name, False) or is_tool_manifest(full):
                        full.unlink()
                fs.rmdir(record.src)
            elif record.op == 'rmdir':
                fs.mkdir(record.src)
            elif record.op == 'symlink':
                if os.path.islink(fs.p(record.dst)):
                    fs.unlink(record.dst)
            elif record.op == 'snapshot':
                if self.snapshot_path.exists():
                    self.snapshot_path.unlink()
            self._log(phase, 'undo', record.op, '', str(record.id))
            count += 1
        return count

    def plan(self) -> List[Row]:
        fs = VirtualFS(self.root, self.opaque)
        rows: List[Row] = []
        for phase in PHASES:
            state = self.phase_state(phase)
            if state == 'applied':
                rows.append(Row(phase, 'phase', '', None, 'done (already applied)'))
                continue
            self._run_phase(phase, fs, rows)
        for name in fs.children(''):
            if name not in self.top_level and name not in CACHE_FILES:
                rows.append(Row('end', 'leftover', name, None, 'error: not part of the target layout'))
        with open(self.plan_path, 'w', encoding='utf-8') as handle:
            handle.write('phase\top\tsrc\tdst\tstatus\n')
            for row in rows:
                handle.write('\t'.join([row.phase, row.op, row.src, row.dst or '', row.status]) + '\n')
        return rows

    def check(self) -> List[str]:
        if not self.snapshot_path.exists():
            return ['no snapshot yet: run apply P1 first']
        moves = [(r.src, r.dst) for r in self.effective() if r.op == 'move']
        problems = []
        lines = self.snapshot_path.read_text(encoding='utf-8').splitlines()[1:]
        for line in lines:
            rel, kind, size, value = line.split('\t')
            new = rel
            for src, dst in moves:
                if new == src or new.startswith(src + '/'):
                    new = dst + new[len(src):]
            path = self.root / new
            if kind in ('L', 'B'):
                if not path.is_symlink() or os.readlink(path) != value:
                    problems.append(f'{rel} -> {new}: link missing or changed')
            elif not path.is_file() or path.is_symlink():
                problems.append(f'{rel} -> {new}: missing')
            elif path.stat().st_size != int(size) or sha256(path) != value:
                problems.append(f'{rel} -> {new}: content changed')
        return problems

    def dangling_links(self) -> List[str]:
        """Links that resolved before the reorganisation and point nowhere now (old -> new path)."""
        if not self.snapshot_path.exists():
            return []
        moves = [(r.src, r.dst) for r in self.effective() if r.op == 'move']
        out = []
        for line in self.snapshot_path.read_text(encoding='utf-8').splitlines()[1:]:
            rel, kind, _, _ = line.split('\t')
            if kind != 'L':
                continue
            new = map_through(rel, moves)
            path = self.root / new
            if path.is_symlink() and not os.path.exists(path):
                out.append(f'{rel} -> {new}')
        return out

    def write_manifests(self, dirs: Iterable[str] = LOCAL_ONLY) -> List[str]:
        written = []
        for rel in dirs:
            base = self.root / rel
            if not base.is_dir():
                continue
            entries = []
            for current, subdirs, files in os.walk(base):
                for name in list(subdirs):
                    if (Path(current) / name).is_symlink():
                        files.append(name)
                subdirs[:] = sorted(d for d in subdirs if not (Path(current) / d).is_symlink())
                for name in files:
                    path = Path(current) / name
                    item = path.relative_to(base).as_posix()
                    if name in CACHE_FILES or item == 'MANIFEST.tsv':
                        continue
                    if path.is_symlink():
                        entries.append(f'{item}\t0\tsymlink:{os.readlink(path)}')
                    else:
                        entries.append(f'{item}\t{path.stat().st_size}\t{sha256(path)}')
            out = base / 'MANIFEST.tsv'
            out.write_text(MANIFEST_HEADER + '\n' + '\n'.join(sorted(entries)) + '\n', encoding='utf-8')
            written.append(f'{rel}/MANIFEST.tsv ({len(entries)} files)')
        return written


def summarize(rows: List[Row], verbose: bool) -> int:
    errors = [r for r in rows if r.status.startswith('error')]
    for phase in PHASES + ('end',):
        phase_rows = [r for r in rows if r.phase == phase]
        if not phase_rows:
            continue
        counts: Dict[str, int] = {}
        for row in phase_rows:
            key = row.status.split(' ')[0].rstrip(':')
            counts[key] = counts.get(key, 0) + 1
        print(f'{phase}: ' + ', '.join(f'{k} {v}' for k, v in sorted(counts.items())))
        if verbose:
            for row in phase_rows:
                arrow = f' -> {row.dst}' if row.dst else ''
                print(f'    {row.op:9} {row.src}{arrow}   [{row.status}]')
    for row in errors:
        arrow = f' -> {row.dst}' if row.dst else ''
        print(f'ERROR {row.phase} {row.op} {row.src}{arrow}: {row.status}')
    return 1 if errors else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['plan', 'apply', 'undo', 'check', 'manifest', 'status'])
    parser.add_argument('phase', nargs='?')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--force', action='store_true',
                        help='apply or undo even if simulator/robotd processes are running')
    parser.add_argument('-v', '--verbose', action='store_true', help='print every action')
    args = parser.parse_args(argv)
    tool = Reorg(Path(args.root))
    try:
        if args.command == 'plan':
            rows = tool.plan()
            status = summarize(rows, args.verbose)
            print(f'Full plan: {tool.plan_path}')
            return status
        if args.command == 'apply':
            rows = tool.apply(args.phase or '', force=args.force)
            summarize(rows, args.verbose)
            print(f'{args.phase} applied. Log: {tool.log_path}')
            return 0
        if args.command == 'undo':
            count = tool.undo(args.phase or '', force=args.force)
            print(f'{args.phase} undone ({count} actions reversed).')
            return 0
        if args.command == 'check':
            problems = tool.check()
            for line in problems[:50]:
                print(line)
            if problems:
                print(f'{len(problems)} problems')
                return 1
            print('check passed: every file from the P1 snapshot is unchanged at its new path')
            dangling = tool.dangling_links()
            if dangling:
                print(f'note: {len(dangling)} symlinks resolved before and point nowhere now '
                      '(old absolute or relative targets; expected inside archive/):')
                for line in dangling:
                    print('  ' + line)
            return 0
        if args.command == 'manifest':
            for line in tool.write_manifests():
                print(line)
            return 0
        for phase in PHASES:
            print(f'{phase}: {tool.phase_state(phase)}')
        return 0
    except ReorgError as error:
        print(f'stopped: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
