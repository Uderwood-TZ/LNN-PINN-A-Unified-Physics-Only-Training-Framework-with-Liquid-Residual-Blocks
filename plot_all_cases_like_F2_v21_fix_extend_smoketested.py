#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import io
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec, font_manager
from matplotlib.ticker import MaxNLocator, ScalarFormatter, LogLocator, LogFormatterMathtext
from matplotlib.colors import Normalize, LogNorm, PowerNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore', category=UserWarning)


def setup_font_environment():
    candidate_families = [
        'Microsoft YaHei', 'SimHei', 'SimSun', 'Noto Sans CJK SC',
        'Source Han Sans SC', 'WenQuanYi Micro Hei', 'PingFang SC',
        'Arial Unicode MS', 'DejaVu Sans',
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [f for f in candidate_families if f in available]
    if not chosen:
        chosen = ['DejaVu Sans']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = chosen + ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'


setup_font_environment()

CASE_ORDER_HINTS = [
    'Laplace2D_UnitSquare_DirTopBot_NeuLeftRight',
    '一维对流–反应方程（Drift–Decay）',
    '各向异性泊松–梁耦合方程Anisotropic Poisson–Beam Equation (APBE)',
    '相对论流体2',
    '相对论流体3',
    '稳态硅板加热',
]

METHOD_ORDER = [
    'PICS',
    'LNN-PINN',
    'RA-PINN',
    'XPINN',
    'DGM',
    'DRM',
    'PINN',
    'FEM',
]

EXCLUDED_METHOD_TOKENS = [
    'matlab有限元',
    'matlab-fem',
]

EXCLUDED_DIR_TOKENS = [
    'for gpt',
    '多参数性能',
    '多性能参数',
]

def is_excluded_method_name(name: str) -> bool:
    s = str(name).strip().lower()
    return any(tok in s for tok in EXCLUDED_METHOD_TOKENS)

def is_excluded_dir_name(name: str) -> bool:
    s = str(name).strip().lower()
    return any(tok in s for tok in EXCLUDED_DIR_TOKENS)

FIELD_DISPLAY_MAP = {
    'phi': r'$\phi$',
    'u': 'u',
    'uy': r'$u_y$',
    'epsilon': r'$\epsilon$',
    'n': 'n',
    'Pi': r'$\Pi$',
    'pi_xx': r'$\pi_{xx}$',
    'pi_xy': r'$\pi_{xy}$',
    'pi_yy': r'$\pi_{yy}$',
    'vx': r'$v_x$',
    'vy': r'$v_y$',
    'Temperature': 'T',
    'temperature': 'T',
    'T': 'T',
}

KNOWN_FIELD_TOKENS = [
    'Temperature', 'temperature',
    'pi_xx', 'pi_xy', 'pi_yy',
    'epsilon', 'phi', 'uy', 'vx', 'vy', 'Pi', 'u', 'n', 'T',
]

LOSS_MARKERS = ['loss_per_epoch', 'loss_history_step_by_step', 'loss_history', 'loss_curve']
SKIP_MARKERS = ['compute_cost', 'metrics', 'summary', 'runtime', 'run_summary', 'error.txt']
GENERATED_DIR_NAMES = {
    '_F2_style_summary_outputs', '_F2_style_summary_outputs_v3',
    '_F2_style_summary_outputs_v2', '_F2_style_summary_outputs_latest',
    '_F2_style_summary_outputs_v4', '_F2_style_summary_outputs_v5', '_F2_style_summary_outputs_v6', '_F2_style_summary_outputs_v8',
    '_F2_style_summary_outputs_v9', '_F2_style_summary_outputs_v18', '_F2_style_summary_outputs_v18',
}


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    name = re.sub(r'\s+', '_', name)
    return name


def case_label_from_index(idx: int) -> str:
    return f'Case {idx}'


def corner_case_label(idx: int) -> str:
    return f'Case {idx}'


def subplot_letter_tag(index_zero_based: int) -> str:
    n = int(index_zero_based)
    letters = []
    while True:
        letters.append(chr(ord('a') + (n % 26)))
        n = n // 26 - 1
        if n < 0:
            break
    return '(' + ''.join(reversed(letters)) + ')'


def annotate_subplot_tag(ax, tag: str):
    ax.text(
        0.0, 1.018, tag,
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=8.8, fontweight='bold',
        clip_on=False, zorder=20,
    )


def nice_method_name(name: str) -> str:
    display_map = {
        'RA-PINN': 'RA-PINN',
        'XPINN': 'XPINN',
    }
    return display_map.get(name, name)


def display_field_name(field_key: str) -> str:
    if field_key in FIELD_DISPLAY_MAP:
        return FIELD_DISPLAY_MAP[field_key]
    if '_' in field_key:
        return '$' + field_key.replace('_', r'\_') + '$'
    return field_key


def display_field_name_bold(field_key: str) -> str:
    bold_map = {
        'phi': r'$\boldsymbol{\phi}$',
        'u': r'$\mathbf{u}$',
        'uy': r'$\mathbf{u}_{y}$',
        'epsilon': r'$\boldsymbol{\epsilon}$',
        'n': r'$\mathbf{n}$',
        'Pi': r'$\boldsymbol{\Pi}$',
        'pi_xx': r'$\boldsymbol{\pi}_{xx}$',
        'pi_xy': r'$\boldsymbol{\pi}_{xy}$',
        'pi_yy': r'$\boldsymbol{\pi}_{yy}$',
        'vx': r'$\mathbf{v}_{x}$',
        'vy': r'$\mathbf{v}_{y}$',
        'Temperature': r'$\mathbf{T}$',
        'temperature': r'$\mathbf{T}$',
        'T': r'$\mathbf{T}$',
    }
    if field_key in bold_map:
        return bold_map[field_key]
    if '_' in field_key:
        head, tail = field_key.split('_', 1)
        return rf'$\mathbf{{{head}}}_{{\mathbf{{{tail}}}}}$'
    return rf'$\mathbf{{{field_key}}}$'


def safe_numeric_read(path: Path) -> Optional[np.ndarray]:
    if not path.exists() or path.is_dir():
        return None
    if path.stat().st_size == 0:
        return None
    seps = [r'\s+', r'[\s,]+', r',', r'\t']
    for sep in seps:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                engine='python',
                header=None,
                comment='#',
                skip_blank_lines=True,
            )
            if df.empty:
                continue
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna(how='all').dropna(axis=1, how='all')
            if df.empty:
                continue
            arr = df.to_numpy(dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            keep = np.isfinite(arr).all(axis=1)
            arr = arr[keep]
            if arr.size == 0:
                continue
            return arr
        except Exception:
            continue
    return None


def file_shape_hint(path: Path) -> Tuple[int, int]:
    arr = safe_numeric_read(path)
    if arr is None:
        return (0, 0)
    if arr.ndim == 1:
        return (arr.shape[0], 1)
    return int(arr.shape[0]), int(arr.shape[1])


def load_curve(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    arr = safe_numeric_read(path)
    if arr is None:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] == 1:
        y = arr[:, 0].astype(float)
        x = np.arange(1, len(y) + 1, dtype=float)
    else:
        x = arr[:, 0].astype(float)
        y = arr[:, 1].astype(float)
        if len(x) >= 3 and np.nanmean(np.diff(x)) <= 0:
            x = np.arange(1, len(y) + 1, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return None
    return x, y


def determine_dim(arr: np.ndarray) -> int:
    if arr.ndim == 1:
        return 1
    cols = arr.shape[1]
    if cols >= 3:
        return 2
    return 1


def prepare_1d(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if arr.ndim == 1:
        y = arr.astype(float)
        x = np.arange(len(y), dtype=float)
        return x, y
    if arr.shape[1] == 1:
        y = arr[:, 0].astype(float)
        x = np.arange(len(y), dtype=float)
        return x, y
    x = arr[:, 0].astype(float)
    y = arr[:, 1].astype(float)
    order = np.argsort(x)
    return x[order], y[order]


def prepare_2d(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = arr[:, 0].astype(float)
    y = arr[:, 1].astype(float)
    z = arr[:, 2].astype(float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[mask], y[mask], z[mask]


def basic_array_stats(arr: Optional[np.ndarray]) -> str:
    if arr is None:
        return 'missing'
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = arr.shape[1]
    if cols == 1:
        v = arr[:, 0]
        return f'rows={arr.shape[0]}, cols=1, vmin={np.nanmin(v):.8g}, vmax={np.nanmax(v):.8g}'
    if cols == 2:
        x, v = arr[:, 0], arr[:, 1]
        return (
            f'rows={arr.shape[0]}, cols=2, '
            f'xmin={np.nanmin(x):.8g}, xmax={np.nanmax(x):.8g}, '
            f'vmin={np.nanmin(v):.8g}, vmax={np.nanmax(v):.8g}'
        )
    x, y, v = arr[:, 0], arr[:, 1], arr[:, 2]
    return (
        f'rows={arr.shape[0]}, cols=3, '
        f'xmin={np.nanmin(x):.8g}, xmax={np.nanmax(x):.8g}, '
        f'ymin={np.nanmin(y):.8g}, ymax={np.nanmax(y):.8g}, '
        f'vmin={np.nanmin(v):.8g}, vmax={np.nanmax(v):.8g}'
    )


def infer_field_key(path: Path) -> Optional[str]:
    name = path.name
    if name.lower() == 'real.txt':
        return None
    stem = path.stem
    full = '/'.join(path.parts)
    for token in KNOWN_FIELD_TOKENS:
        if re.search(rf'(?:^|[_/\\]){re.escape(token)}(?:$|[_./\\])', full):
            return token
    pieces = re.split(r'[_\-\s]+', stem)
    pieces = [p for p in pieces if p]
    if len(pieces) >= 2 and pieces[-2] == 'pi' and pieces[-1] in {'xx', 'xy', 'yy'}:
        return f'pi_{pieces[-1]}'
    return None


def classify_kind(path: Path) -> Optional[str]:
    low = path.name.lower()
    full = '/'.join(path.parts).lower()
    if path.suffix.lower() not in {'.txt', '.csv'}:
        return None
    if any(m in low for m in LOSS_MARKERS) or any(m in full for m in LOSS_MARKERS):
        return 'loss'
    if any(m in low for m in SKIP_MARKERS):
        return None
    if low == 'real.txt':
        return 'aux_real'
    if 'abs_error' in low:
        return 'err'
    if 'maxerror' in low:
        return 'err'
    if '_true' in low or 'true_' in low or low.endswith('_true.txt'):
        return 'true'
    if '_pred' in low or 'pred_' in low or low.endswith('_pred.txt'):
        return 'pred'
    return None


def case_sort_key(path: Path) -> Tuple[int, str]:
    name = path.name
    if name in CASE_ORDER_HINTS:
        return (CASE_ORDER_HINTS.index(name), name)
    return (999, name)


def find_case_directories(root: Path) -> List[Path]:
    candidates = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith('.'):
            continue
        if p.name in GENERATED_DIR_NAMES:
            continue
        if is_excluded_dir_name(p.name):
            continue
        candidates.append(p)
    candidates.sort(key=case_sort_key)
    return candidates


def find_method_directories(case_dir: Path) -> List[Tuple[str, Path]]:
    methods = []
    for p in case_dir.iterdir():
        if not p.is_dir() or p.name.startswith('.'):
            continue
        if is_excluded_dir_name(p.name):
            continue
        methods.append((p.name, p))
    methods.sort(key=lambda x: ((METHOD_ORDER.index(x[0]) if x[0] in METHOD_ORDER else 999), x[0]))
    return methods


def choose_best_file(paths: List[Path], kind: str) -> Optional[Path]:
    if not paths:
        return None

    def score(p: Path) -> Tuple[int, int, int, int]:
        low = p.name.lower()
        full = '/'.join(p.parts).lower()
        size = int(p.stat().st_size)
        rows, cols = file_shape_hint(p)
        s = 0
        if 'data' in full:
            s -= 40
        if 'logs' in full and kind != 'loss':
            s += 20
        if kind == 'loss':
            if 'loss_per_epoch' in low:
                s -= 30
            elif 'loss_history_step_by_step' in low:
                s -= 25
            elif 'loss_history' in low:
                s -= 20
            elif 'loss_curve' in low:
                s += 10
        elif kind == 'true':
            if 'true_xyz' in low:
                s -= 40
            elif '_true' in low:
                s -= 30
            elif low == 'real.txt':
                s += 80
        elif kind == 'pred':
            if 'pred_xyz' in low:
                s -= 35
            elif '_pred' in low:
                s -= 25
            elif '_pinn' in low:
                s += 15
        elif kind == 'err':
            if 'abs_error_xyz' in low:
                s -= 45
            elif 'abs_error' in low:
                s -= 35
            elif 'maxerror_xyz' in low:
                s += 10
            elif 'maxerror' in low:
                s += 15
            if size < 256 or rows <= 5:
                s += 200
            if rows >= 100:
                s -= 10
            if cols >= 3:
                s -= 5
        return (s, -size, -rows, len(str(p)))

    return sorted(paths, key=score)[0]


def collect_method_files(method_dir: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {'loss': [], 'true': [], 'pred': [], 'err': [], 'aux_real': []}
    for p in method_dir.rglob('*'):
        if not p.is_file():
            continue
        kind = classify_kind(p)
        if kind is None:
            continue
        out.setdefault(kind, []).append(p)
    return out


def build_case_bundle(case_dir: Path) -> Dict:
    bundle: Dict = {
        'case_name': case_dir.name,
        'case_dir': case_dir,
        'methods': [],
        'losses': {},
        'fields': {},
        'field_truth': {},
        'pred_methods': [],
    }

    for method_name, method_dir in find_method_directories(case_dir):
        files = collect_method_files(method_dir)
        method_entry = {
            'name': method_name,
            'dir': method_dir,
            'fields': {},
            'aux_real': choose_best_file(files.get('aux_real', []), 'true'),
        }

        loss_file = choose_best_file(files.get('loss', []), 'loss')
        if loss_file is not None:
            bundle['losses'][method_name] = loss_file

        grouped: Dict[str, Dict[str, List[Path]]] = {}
        for kind in ['true', 'pred', 'err']:
            for p in files.get(kind, []):
                field_key = infer_field_key(p)
                if field_key is None:
                    continue
                grouped.setdefault(field_key, {}).setdefault(kind, []).append(p)

        for field_key, kind_map in grouped.items():
            info = {}
            for kind in ['true', 'pred', 'err']:
                best = choose_best_file(kind_map.get(kind, []), kind)
                if best is not None:
                    info[kind] = best
            if info:
                method_entry['fields'][field_key] = info
                bundle['fields'].setdefault(field_key, {})[method_name] = info

        bundle['methods'].append(method_entry)

    pred_methods = []
    for method in bundle['methods']:
        if is_excluded_method_name(method['name']):
            continue
        if any('pred' in info for info in method['fields'].values()):
            pred_methods.append(method['name'])
    ordered = [m for m in METHOD_ORDER if m in pred_methods]
    ordered += [m for m in pred_methods if m not in ordered]
    bundle['pred_methods'] = ordered

    for field_key, by_method in bundle['fields'].items():
        truth_candidates = []
        for _method_name, info in by_method.items():
            if 'true' in info:
                truth_candidates.append(info['true'])
        truth_file = choose_best_file(truth_candidates, 'true')
        if truth_file is not None:
            bundle['field_truth'][field_key] = truth_file

    return bundle



GENERIC_REL_PARTS = {
    'data', 'logs', 'log', 'figures', 'figure', 'imgs', 'images',
    'results', 'result', 'outputs', 'output', 'pred', 'true', 'err',
    'errors', 'maxerror', 'checkpoints', 'ckpt', 'models', 'model'
}

CASE_KEY_HINTS = [
    'drift', 'decay', 'apbe', 'poisson', 'beam', 'laplace', 'unit', 'square',
    'relativistic', 'casea', 'caseb', 'casec', 'heat', 'silicon',
    '对流', '反应', '泊松', '梁', '相对论', '流体', '稳态', '加热', '硅板'
]


def is_lr_or_numeric_token(token: str) -> bool:
    low = token.lower()
    if re.fullmatch(r'[\d.]+(?:e[+-]?\d+)?', low):
        return True
    if re.fullmatch(r'lr[\d.e+-]+', low):
        return True
    if re.fullmatch(r'[\d.]+e[+-]?\d+', low):
        return True
    return False


def sanitize_case_key_token(token: str) -> str:
    token = token.strip().strip('_- ')
    token = re.sub(r'\s+', '_', token)
    return token or 'case'


def infer_case_key_from_method_file(method_dir: Path, file_path: Path) -> str:
    rel = file_path.relative_to(method_dir)
    parts = list(rel.parts[:-1])
    cleaned = []
    for part in parts:
        low = part.lower()
        if low in GENERIC_REL_PARTS:
            continue
        if is_lr_or_numeric_token(low):
            continue
        cleaned.append(part)
    for part in cleaned:
        low = part.lower()
        if any(h in low for h in CASE_KEY_HINTS):
            return sanitize_case_key_token(part)
    for part in cleaned:
        low = part.lower()
        if low.startswith('outputs_') and any(h in low for h in CASE_KEY_HINTS):
            return sanitize_case_key_token(part)
    if cleaned:
        return sanitize_case_key_token(cleaned[0])
    stem = file_path.stem
    low = stem.lower()
    m = re.search(r'(case[a-z0-9]+)', low)
    if m:
        return sanitize_case_key_token(m.group(1))
    for h in CASE_KEY_HINTS:
        if h in low:
            return sanitize_case_key_token(h)
    return method_dir.name


def find_case_bundles_from_method_root(root: Path) -> List[Dict]:
    case_groups: Dict[str, Dict] = {}
    method_dirs = find_method_directories(root)
    for method_name, method_dir in method_dirs:
        files = collect_method_files(method_dir)
        grouped_by_case: Dict[str, Dict[str, List[Path]]] = {}
        for kind, paths in files.items():
            for p in paths:
                case_key = infer_case_key_from_method_file(method_dir, p)
                grouped_by_case.setdefault(case_key, {'loss': [], 'true': [], 'pred': [], 'err': [], 'aux_real': []})
                grouped_by_case[case_key].setdefault(kind, []).append(p)
        for case_key, per_kind in grouped_by_case.items():
            bundle = case_groups.setdefault(case_key, {
                'case_name': case_key,
                'case_dir': root / case_key,
                'methods': [],
                'losses': {},
                'fields': {},
                'field_truth': {},
                'pred_methods': [],
            })
            method_entry = {
                'name': method_name,
                'dir': method_dir,
                'fields': {},
                'aux_real': choose_best_file(per_kind.get('aux_real', []), 'true'),
            }
            loss_file = choose_best_file(per_kind.get('loss', []), 'loss')
            if loss_file is not None:
                bundle['losses'][method_name] = loss_file
            grouped_fields: Dict[str, Dict[str, List[Path]]] = {}
            for kind in ['true', 'pred', 'err']:
                for p in per_kind.get(kind, []):
                    field_key = infer_field_key(p)
                    if field_key is None:
                        continue
                    grouped_fields.setdefault(field_key, {}).setdefault(kind, []).append(p)
            for field_key, kind_map in grouped_fields.items():
                info = {}
                for kind in ['true', 'pred', 'err']:
                    best = choose_best_file(kind_map.get(kind, []), kind)
                    if best is not None:
                        info[kind] = best
                if info:
                    method_entry['fields'][field_key] = info
                    bundle['fields'].setdefault(field_key, {})[method_name] = info
            bundle['methods'].append(method_entry)
    bundles: List[Dict] = []
    for case_key, bundle in case_groups.items():
        pred_methods = []
        for method in bundle['methods']:
            if method['name'] == 'matlab有限元':
                continue
            if any('pred' in info for info in method['fields'].values()):
                pred_methods.append(method['name'])
        ordered = [m for m in METHOD_ORDER if m in pred_methods]
        ordered += [m for m in pred_methods if m not in ordered]
        bundle['pred_methods'] = ordered
        for field_key, by_method in bundle['fields'].items():
            truth_candidates = []
            for _method_name, info in by_method.items():
                if 'true' in info:
                    truth_candidates.append(info['true'])
            truth_file = choose_best_file(truth_candidates, 'true')
            if truth_file is not None:
                bundle['field_truth'][field_key] = truth_file
        bundles.append(bundle)
    def bundle_key(b: Dict):
        name = b['case_name'].lower()
        score = 999
        for i, hint in enumerate(CASE_ORDER_HINTS):
            if hint.lower() in name:
                score = i
                break
        return (score, b['case_name'])
    bundles.sort(key=bundle_key)
    return bundles


def detect_root_layout(root: Path) -> str:
    dirs = [
        p for p in root.iterdir()
        if p.is_dir()
        and not p.name.startswith('.')
        and p.name not in GENERATED_DIR_NAMES
        and not is_excluded_dir_name(p.name)
    ]
    if not dirs:
        return 'empty'
    names = {p.name for p in dirs}
    method_hits = sum(1 for n in names if n in METHOD_ORDER)
    if method_hits >= max(2, len(names) // 2):
        return 'method_first'
    case_like = 0
    for d in dirs:
        subdirs = [p for p in d.iterdir() if p.is_dir() and not p.name.startswith('.')]
        sub_names = {p.name for p in subdirs}
        sub_method_hits = sum(1 for n in sub_names if n in METHOD_ORDER)
        if sub_method_hits >= 1:
            case_like += 1
    if case_like >= max(1, len(dirs) // 2):
        return 'case_first'
    return 'unknown'

def canonical_field_order(fields: List[str]) -> List[str]:
    order = ['phi', 'u', 'uy', 'epsilon', 'n', 'Pi', 'pi_xx', 'pi_xy', 'pi_yy', 'vx', 'vy', 'Temperature', 'temperature', 'T']
    score = {k: i for i, k in enumerate(order)}
    return sorted(fields, key=lambda x: (score.get(x, 999), x))


def clean_axes(ax, show_x: bool, show_y: bool, xlabel: str = 'x', ylabel: Optional[str] = None):
    ax.tick_params(labelsize=7, pad=0.8, length=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.margins(x=0, y=0)
    ax.set_xlabel('')
    ax.set_ylabel('')
    if not show_x:
        ax.tick_params(labelbottom=False)
    if not show_y:
        ax.tick_params(labelleft=False)

def simplify_colorbar(cb, scale_mode: str = 'linear'):
    if scale_mode.startswith('log'):
        cb.locator = LogLocator(base=10.0, subs=(1.0,), numticks=4)
        cb.formatter = LogFormatterMathtext(base=10.0)
    else:
        cb.locator = MaxNLocator(nbins=3, min_n_ticks=2)
        cb.formatter = FuncFormatter(lambda x, pos: f'{x:.2g}')
    cb.update_ticks()
    cb.ax.tick_params(labelsize=5.6, pad=1.2, length=1.8)
    cb.ax.yaxis.get_offset_text().set_visible(False)
    cb.ax.minorticks_off()
    for spine in cb.ax.spines.values():
        spine.set_linewidth(0.5)

def plot_panel_1d(ax, arr_true: Optional[np.ndarray], arr_plot: np.ndarray, title: str, is_error: bool):
    if arr_true is not None and not is_error:
        xt, yt = prepare_1d(arr_true)
        ax.plot(xt, yt, linewidth=1.25, label='Exact')
    xp, yp = prepare_1d(arr_plot)
    ax.plot(xp, yp, linewidth=1.3, label=('Error' if is_error else 'Pred'))
    ax.set_title(title, fontsize=8.2, pad=1.4)
    ax.grid(alpha=0.18, linewidth=0.45)
    ax.margins(x=0)
    if arr_true is not None and not is_error:
        ax.legend(fontsize=6.5, loc='best', frameon=False, handlelength=1.2)

def plot_panel_2d(ax, arr: np.ndarray, title: str, vmin: float, vmax: float, cmap: str = 'jet', norm=None):
    x, y, z = prepare_2d(arr)
    ax.set_title(title, fontsize=8.2, pad=1.4)
    if len(z) == 0:
        ax.text(0.5, 0.5, 'empty', ha='center', va='center', fontsize=10)
        ax.set_axis_off()
        return None
    if norm is None:
        if not np.isfinite(vmin):
            vmin = float(np.nanmin(z))
        if not np.isfinite(vmax):
            vmax = float(np.nanmax(z))
        if abs(vmax - vmin) < 1e-14:
            vmin -= 1e-12
            vmax += 1e-12
        norm = Normalize(vmin=vmin, vmax=vmax)
    if isinstance(norm, LogNorm):
        z_plot = np.asarray(z, dtype=float).copy()
        z_plot[~np.isfinite(z_plot)] = norm.vmin
        z_plot[z_plot <= 0] = norm.vmin
    else:
        z_plot = z

    x_round = np.round(x.astype(float), 12)
    y_round = np.round(y.astype(float), 12)
    xu = np.unique(x_round)
    yu = np.unique(y_round)

    mappable = None
    can_grid = (
        xu.size >= 2 and yu.size >= 2
        and xu.size * yu.size <= 400000
        and len(x) >= max(100, int(0.35 * xu.size * yu.size))
    )
    if can_grid:
        try:
            ix = np.searchsorted(xu, x_round)
            iy = np.searchsorted(yu, y_round)
            grid = np.full((yu.size, xu.size), np.nan, dtype=float)
            grid[iy, ix] = z_plot
            grid = np.ma.masked_invalid(grid)
            mappable = ax.pcolormesh(xu, yu, grid, cmap=cmap, norm=norm, shading='auto')
        except Exception:
            mappable = None

    unique_pts = len({(round(float(a), 12), round(float(b), 12)) for a, b in zip(x, y)})
    use_tri = mappable is None and len(x) >= 3 and unique_pts >= 3 and len(x) <= 20000
    if use_tri:
        x0, y0 = float(x[0]), float(y[0])
        non_collinear = False
        max_i = min(len(x), 50)
        max_j = min(len(x), 100)
        for i in range(1, max_i):
            for j in range(i + 1, max_j):
                area2 = (float(x[i]) - x0) * (float(y[j]) - y0) - (float(x[j]) - x0) * (float(y[i]) - y0)
                if abs(area2) > 1e-14:
                    non_collinear = True
                    break
            if non_collinear:
                break
        use_tri = non_collinear

    if mappable is None and use_tri:
        try:
            mappable = ax.tricontourf(x, y, z_plot, levels=120, cmap=cmap, norm=norm)
        except Exception:
            mappable = None
    if mappable is None:
        mappable = ax.scatter(x, y, c=z_plot, cmap=cmap, norm=norm, s=8, linewidths=0, rasterized=True)

    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    if np.isfinite(xmin) and np.isfinite(xmax):
        ax.set_xlim(xmin, xmax)
    if np.isfinite(ymin) and np.isfinite(ymax):
        ax.set_ylim(ymin, ymax)
    ax.margins(x=0, y=0)
    return mappable

def add_zoom_inset(ax, curves: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    if not curves:
        return
    axins = ax.inset_axes([0.68, 0.42, 0.28, 0.46])
    drawn = False
    for name, (x, y) in curves.items():
        if len(x) < 20:
            continue
        k0 = max(int(0.85 * len(x)), 0)
        ysub = y[k0:]
        pos = ysub > 0
        if np.count_nonzero(pos) >= 2:
            axins.semilogy(x[k0:][pos], ysub[pos], linewidth=1.0, label=nice_method_name(name))
        else:
            axins.plot(x[k0:], ysub, linewidth=1.0, label=nice_method_name(name))
        drawn = True
    if not drawn:
        axins.remove()
        return
    axins.set_title('zoom', fontsize=7, pad=1)
    axins.tick_params(labelsize=6, pad=1)
    axins.grid(alpha=0.20, linewidth=0.4)
    axins.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axins.yaxis.set_major_locator(MaxNLocator(nbins=3))


def construct_error_array(arr_true: np.ndarray, arr_pred: np.ndarray, dim: int) -> Optional[np.ndarray]:
    try:
        if dim == 1:
            xt, yt = prepare_1d(arr_true)
            xp, yp = prepare_1d(arr_pred)
            if len(xt) == len(xp) and np.allclose(xt, xp):
                return np.column_stack([xt, np.abs(yt - yp)])
            yp_interp = np.interp(xt, xp, yp)
            return np.column_stack([xt, np.abs(yt - yp_interp)])
        xt, yt, zt = prepare_2d(arr_true)
        xp, yp, zp = prepare_2d(arr_pred)
        same = len(xt) == len(xp) and np.allclose(xt, xp) and np.allclose(yt, yp)
        if same:
            return np.column_stack([xt, yt, np.abs(zt - zp)])
        return None
    except Exception:
        return None


def choose_error_color_scale(err_arrays: Dict[str, np.ndarray], dim: int) -> Dict[str, object]:
    if not err_arrays:
        return {'norm': Normalize(vmin=0.0, vmax=1.0), 'vmin': 0.0, 'vmax': 1.0, 'mode': 'linear_fallback', 'extend': 'neither'}

    vals = []
    method_q005 = []
    method_q05 = []
    method_q95 = []
    method_q99 = []
    for arr in err_arrays.values():
        if dim == 2:
            z = prepare_2d(arr)[2]
        else:
            z = prepare_1d(arr)[1]
        z = np.asarray(z, dtype=float)
        z = z[np.isfinite(z)]
        z = np.abs(z)
        if z.size:
            vals.append(z)
            pos_m = z[z > 0]
            if pos_m.size:
                method_q005.append(float(np.nanquantile(pos_m, 0.005)))
                method_q05.append(float(np.nanquantile(pos_m, 0.05)))
                method_q95.append(float(np.nanquantile(pos_m, 0.95)))
                method_q99.append(float(np.nanquantile(pos_m, 0.99)))
    if not vals:
        return {'norm': Normalize(vmin=0.0, vmax=1.0), 'vmin': 0.0, 'vmax': 1.0, 'mode': 'linear_fallback', 'extend': 'neither'}

    allv = np.concatenate(vals)
    vmax_raw = float(np.nanmax(allv))
    if not np.isfinite(vmax_raw) or vmax_raw <= 0:
        return {'norm': Normalize(vmin=0.0, vmax=1.0), 'vmin': 0.0, 'vmax': 1.0, 'mode': 'linear_fallback', 'extend': 'neither'}

    if dim == 1:
        return {'norm': None, 'vmin': 0.0, 'vmax': vmax_raw, 'mode': 'line_only', 'extend': 'neither'}

    pos = allv[allv > 0]
    if pos.size < 10:
        return {'norm': Normalize(vmin=0.0, vmax=vmax_raw), 'vmin': 0.0, 'vmax': vmax_raw, 'mode': 'linear_max', 'extend': 'neither'}

    q001 = float(np.nanquantile(pos, 0.001))
    q005 = float(np.nanquantile(pos, 0.005))
    q01 = float(np.nanquantile(pos, 0.01))
    q05 = float(np.nanquantile(pos, 0.05))
    q10 = float(np.nanquantile(pos, 0.10))
    q95 = float(np.nanquantile(pos, 0.95))
    q98 = float(np.nanquantile(pos, 0.98))
    q99 = float(np.nanquantile(pos, 0.99))
    q995 = float(np.nanquantile(pos, 0.995))
    min_pos = float(np.nanmin(pos))

    cross_ratio_95 = 1.0
    if method_q95:
        pos95 = [v for v in method_q95 if np.isfinite(v) and v > 0]
        if len(pos95) >= 2:
            cross_ratio_95 = max(pos95) / max(min(pos95), 1e-300)

    cross_ratio_99 = 1.0
    if method_q99:
        pos99 = [v for v in method_q99 if np.isfinite(v) and v > 0]
        if len(pos99) >= 2:
            cross_ratio_99 = max(pos99) / max(min(pos99), 1e-300)

    dynamic_ratio = vmax_raw / max(min_pos, 1e-300)
    left_heavy = q99 / max(q01, 1e-300)

    if (
        (cross_ratio_95 >= 20.0)
        or (cross_ratio_99 >= 30.0)
        or (dynamic_ratio >= 300.0)
        or (left_heavy >= 150.0)
        or (q995 / max(q005, 1e-300) >= 250.0)
    ):
        vmin = max(min_pos, min(q001, q005, q01))
        vmax = q995 if vmax_raw / max(q995, 1e-300) >= 1.10 else vmax_raw
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = max(min_pos, q005)
            vmax = max(q99, vmin * 10.0)
        return {'norm': LogNorm(vmin=vmin, vmax=vmax), 'vmin': vmin, 'vmax': vmax, 'mode': 'log_cross_method_robust', 'extend': 'max' if vmax < vmax_raw else 'neither'}

    if q99 > 0 and q01 > 0 and (q99 / q01 >= 120.0 or vmax_raw / max(q99, 1e-300) >= 2.5):
        vmin = max(min_pos, min(q005, q01))
        vmax = q99 if vmax_raw / max(q99, 1e-300) >= 1.15 else vmax_raw
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = max(min_pos, min(q01, q05))
            vmax = max(q99, vmin * 10.0)
        return {'norm': LogNorm(vmin=vmin, vmax=vmax), 'vmin': vmin, 'vmax': vmax, 'mode': 'log_q01_q99_aggressive', 'extend': 'max' if vmax < vmax_raw else 'neither'}

    if q98 > 0 and (vmax_raw / max(q98, 1e-300) >= 1.6 or q98 / max(q10, 1e-300) >= 25.0):
        vmax = q98
        return {'norm': PowerNorm(gamma=0.38, vmin=0.0, vmax=vmax), 'vmin': 0.0, 'vmax': vmax, 'mode': 'power_q98_g038', 'extend': 'max'}

    if q95 > 0 and vmax_raw / max(q95, 1e-300) >= 3.0:
        vmax = q95
        return {'norm': PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax), 'vmin': 0.0, 'vmax': vmax, 'mode': 'power_q95_g045', 'extend': 'max'}

    return {'norm': Normalize(vmin=0.0, vmax=vmax_raw), 'vmin': 0.0, 'vmax': vmax_raw, 'mode': 'linear_max', 'extend': 'neither'}




def choose_local_error_display_scale(err_arr: np.ndarray, case_name: str = '', field_name: str = '') -> Dict[str, object]:
    z = np.asarray(prepare_2d(err_arr)[2], dtype=float)
    z = np.abs(z[np.isfinite(z)])
    if z.size == 0:
        return {'norm': Normalize(vmin=0.0, vmax=1.0), 'vmin': 0.0, 'vmax': 1.0, 'mode': 'linear_empty', 'extend': 'neither'}

    vmax_raw = float(np.nanmax(z))
    if not np.isfinite(vmax_raw) or vmax_raw <= 0:
        return {'norm': Normalize(vmin=0.0, vmax=1.0), 'vmin': 0.0, 'vmax': 1.0, 'mode': 'linear_zero', 'extend': 'neither'}

    pos = z[z > 0]
    if pos.size < 8:
        return {'norm': Normalize(vmin=0.0, vmax=vmax_raw), 'vmin': 0.0, 'vmax': vmax_raw, 'mode': 'linear_raw', 'extend': 'neither'}

    q001 = float(np.nanquantile(pos, 0.001))
    q005 = float(np.nanquantile(pos, 0.005))
    q01  = float(np.nanquantile(pos, 0.01))
    q02  = float(np.nanquantile(pos, 0.02))
    q05  = float(np.nanquantile(pos, 0.05))
    q10  = float(np.nanquantile(pos, 0.10))
    q50  = float(np.nanquantile(pos, 0.50))
    q90  = float(np.nanquantile(pos, 0.90))
    q95  = float(np.nanquantile(pos, 0.95))
    q98  = float(np.nanquantile(pos, 0.98))
    q99  = float(np.nanquantile(pos, 0.99))
    q995 = float(np.nanquantile(pos, 0.995))
    min_pos = float(np.nanmin(pos))

    span_rel = (q99 - q01) / max(q50, 1e-300)
    dynamic_ratio = vmax_raw / max(min_pos, 1e-300)
    tail_ratio = q995 / max(q005, 1e-300)

    case_low = str(case_name).lower()
    field_low = str(field_name).lower()
    aggressive = ('laplace' in case_low) or ('稳态硅板加热' in case_name) or (field_low in {'t', 'temperature'})

    if aggressive and tail_ratio >= 15.0:
        vmin = max(min_pos, min(q001, q005, q01))
        vmax = max(q99, vmin * 10.0)
        if vmax > vmin:
            return {'norm': LogNorm(vmin=vmin, vmax=vmax), 'vmin': vmin, 'vmax': vmax, 'mode': 'local_log_aggressive', 'extend': 'max' if vmax < vmax_raw else 'neither'}

    if dynamic_ratio >= 200.0 or tail_ratio >= 80.0:
        vmin = max(min_pos, min(q001, q005, q01))
        vmax = q995 if vmax_raw / max(q995, 1e-300) >= 1.05 else vmax_raw
        if vmax <= vmin:
            vmax = max(q99, vmin * 10.0)
        return {'norm': LogNorm(vmin=vmin, vmax=vmax), 'vmin': vmin, 'vmax': vmax, 'mode': 'local_log', 'extend': 'max' if vmax < vmax_raw else 'neither'}

    if span_rel <= 0.30:
        vmin = max(0.0, q01)
        vmax = q99
        if vmax <= vmin:
            vmax = max(q995, vmin + 1e-12)
        return {'norm': PowerNorm(gamma=0.55, vmin=vmin, vmax=vmax), 'vmin': vmin, 'vmax': vmax, 'mode': 'local_power_narrow', 'extend': 'both'}

    if aggressive and span_rel <= 0.80:
        vmin = max(0.0, q005)
        vmax = q995
        if vmax <= vmin:
            vmax = max(q99, vmin + 1e-12)
        return {'norm': PowerNorm(gamma=0.65, vmin=vmin, vmax=vmax), 'vmin': vmin, 'vmax': vmax, 'mode': 'local_power_aggressive', 'extend': 'both'}

    if vmax_raw / max(q98, 1e-300) >= 1.2:
        return {'norm': PowerNorm(gamma=0.72, vmin=0.0, vmax=q98), 'vmin': 0.0, 'vmax': q98, 'mode': 'local_power_q98', 'extend': 'max'}

    return {'norm': Normalize(vmin=0.0, vmax=vmax_raw), 'vmin': 0.0, 'vmax': vmax_raw, 'mode': 'local_linear_raw', 'extend': 'neither'}


def add_panel_local_colorbar(ax, mappable, scale_mode: str = 'linear', extend: str = 'neither'):
    if mappable is None:
        return
    cax = ax.inset_axes([1.045, 0.08, 0.032, 0.84])
    cb = plt.colorbar(mappable, cax=cax, extend=extend)
    simplify_colorbar(cb, scale_mode=scale_mode)

def choose_prediction_display_config(arr_true: np.ndarray, pred_arrays: Dict[str, np.ndarray], dim: int) -> Dict[str, object]:
    if dim != 2:
        return {'mode': 'raw', 'offset': 0.0, 'pred_vmin': 0.0, 'pred_vmax': 1.0}
    z_true = np.asarray(prepare_2d(arr_true)[2], dtype=float)
    z_true = z_true[np.isfinite(z_true)]
    z_stack = [z_true]
    for arr in pred_arrays.values():
        z = np.asarray(prepare_2d(arr)[2], dtype=float)
        z = z[np.isfinite(z)]
        if z.size:
            z_stack.append(z)
    all_z = np.concatenate([z for z in z_stack if z.size]) if z_stack else np.array([], dtype=float)
    if all_z.size == 0 or z_true.size == 0:
        return {'mode': 'raw', 'offset': 0.0, 'pred_vmin': 0.0, 'pred_vmax': 1.0}
    raw_vmin = float(np.nanmin(all_z))
    raw_vmax = float(np.nanmax(all_z))
    if abs(raw_vmax - raw_vmin) < 1e-14:
        raw_vmin -= 1e-12
        raw_vmax += 1e-12

    true_ref = float(np.nanmedian(z_true))
    true_span = float(np.nanmax(z_true) - np.nanmin(z_true))
    true_amp = max(abs(true_ref), float(np.nanpercentile(np.abs(z_true), 95)), 1.0)
    true_rel_span = true_span / true_amp
    global_rel_span = abs(raw_vmax - raw_vmin) / max(abs(true_ref), float(np.nanpercentile(np.abs(all_z), 95)), 1.0)

    centered_abs_q = []
    for z in z_stack:
        if not z.size:
            continue
        zc = z - float(np.nanmedian(z))
        zc = zc[np.isfinite(zc)]
        if zc.size:
            centered_abs_q.append(float(np.nanquantile(np.abs(zc), 0.995)))
    true_centered = z_true - true_ref
    true_centered_abs_q = float(np.nanquantile(np.abs(true_centered), 0.995)) if true_centered.size else 0.0

    if true_rel_span < 2.0e-3:
        amp_candidates = [v for v in centered_abs_q if np.isfinite(v) and v > 0]
        amp = max([true_centered_abs_q, 1.0e-12] + amp_candidates)
        amp = max(amp, true_span * 0.55, 1.0e-6)
        return {
            'mode': 'panel_median_removed',
            'offset': 0.0,
            'pred_vmin': -amp,
            'pred_vmax': amp,
            'panel_centering': 'per_panel_median',
            'reference_value': true_ref,
        }

    if global_rel_span < 5e-3:
        centered = []
        for z in z_stack:
            zc = z - true_ref
            zc = zc[np.isfinite(zc)]
            if zc.size:
                centered.append(zc)
        if centered:
            zc_all = np.concatenate(centered)
            vmin = float(np.nanmin(zc_all))
            vmax = float(np.nanmax(zc_all))
            if abs(vmax - vmin) < 1e-14:
                vmin -= 1e-12
                vmax += 1e-12
            return {'mode': 'offset_removed', 'offset': true_ref, 'pred_vmin': vmin, 'pred_vmax': vmax}
    return {'mode': 'raw', 'offset': 0.0, 'pred_vmin': raw_vmin, 'pred_vmax': raw_vmax}


def prepare_prediction_display_array(arr: np.ndarray, dim: int, display_cfg: Dict[str, object]) -> np.ndarray:
    if dim != 2:
        return arr
    mode = display_cfg.get('mode')
    if mode not in {'offset_removed', 'panel_median_removed'}:
        return arr
    arr2 = np.asarray(arr, dtype=float).copy()
    if arr2.ndim == 2 and arr2.shape[1] >= 3:
        if mode == 'offset_removed':
            arr2[:, 2] = arr2[:, 2] - float(display_cfg.get('offset', 0.0))
        elif mode == 'panel_median_removed':
            z = np.asarray(arr2[:, 2], dtype=float)
            zf = z[np.isfinite(z)]
            if zf.size:
                arr2[:, 2] = z - float(np.nanmedian(zf))
    return arr2


def write_case_manifest(out_path: Path, case_name: str, case_label: str, pred_methods: List[str], fields: List[str], manifest_lines: List[str]):
    lines = [
        f'case = {case_name}',
        f'label = {case_label}',
        'pred_methods = ' + ', '.join(pred_methods),
        'fields = ' + ', '.join(fields),
        '',
    ]
    lines.extend(manifest_lines)
    out_path.write_text('\n'.join(lines), encoding='utf-8')



def plot_case_summary(bundle: Dict, out_dir: Path, case_index: int) -> Optional[Path]:
    case_name = bundle['case_name']
    case_label = case_label_from_index(case_index)
    pred_methods = bundle['pred_methods']
    fields = [
        f for f in canonical_field_order(list(bundle['fields'].keys()))
        if f in bundle['field_truth'] and any(m in bundle['fields'].get(f, {}) and 'pred' in bundle['fields'][f][m] for m in pred_methods)
    ]
    if not pred_methods or not fields:
        print(f'[跳过] {case_name}：没有足够的真解/预测数据')
        return None

    field_dims: Dict[str, int] = {}
    truth_arrays: Dict[str, np.ndarray] = {}
    for field in fields:
        arr = safe_numeric_read(bundle['field_truth'][field])
        if arr is None:
            continue
        truth_arrays[field] = arr
        field_dims[field] = determine_dim(arr)
    fields = [f for f in fields if f in truth_arrays]
    if not fields:
        print(f'[跳过] {case_name}：真解文件无法读取')
        return None

    loss_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    manifest_lines: List[str] = []
    row_label_specs: List[Tuple[plt.Axes, str]] = []
    panel_counter = 0

    for method_name, p in bundle['losses'].items():
        curve = load_curve(p)
        if curve is not None:
            loss_curves[method_name] = curve
            x, y = curve
            manifest_lines.append(f'[loss] method={method_name}')
            manifest_lines.append(f'  source = {p}')
            manifest_lines.append(f'  rows = {len(x)}, xmin = {np.nanmin(x):.8g}, xmax = {np.nanmax(x):.8g}, ymin = {np.nanmin(y):.8g}, ymax = {np.nanmax(y):.8g}')

    row_infos: List[Dict] = []
    for field in fields:
        arr_true = truth_arrays[field]
        dim = field_dims[field]
        pred_arrays: Dict[str, np.ndarray] = {}
        err_arrays: Dict[str, np.ndarray] = {}
        pred_sources: Dict[str, str] = {}
        err_sources: Dict[str, str] = {}

        raw_err_arrays: Dict[str, np.ndarray] = {}
        raw_err_sources: Dict[str, str] = {}

        for method in pred_methods:
            info = bundle['fields'].get(field, {}).get(method, {})
            if 'pred' in info:
                arrp = safe_numeric_read(info['pred'])
                if arrp is not None:
                    pred_arrays[method] = arrp
                    pred_sources[method] = str(info['pred'])
            if 'err' in info:
                arre = safe_numeric_read(info['err'])
                if arre is not None:
                    rows, cols = arre.shape if arre.ndim == 2 else (arre.shape[0], 1)
                    if dim == 2 and rows <= 5:
                        pass
                    else:
                        raw_err_arrays[method] = arre
                        raw_err_sources[method] = str(info['err'])

        row_methods = [m for m in pred_methods if m in pred_arrays]
        if not row_methods:
            continue

        for method in row_methods:
            constructed = construct_error_array(arr_true, pred_arrays[method], dim)
            if constructed is not None:
                err_arrays[method] = constructed
                err_sources[method] = 'constructed_abs_error_from_true_pred'
            elif method in raw_err_arrays:
                err_arrays[method] = raw_err_arrays[method]
                err_sources[method] = raw_err_sources[method]

        if dim == 2:
            display_cfg = choose_prediction_display_config(arr_true, {m: pred_arrays[m] for m in row_methods}, dim=2)
            pred_vmin = float(display_cfg['pred_vmin'])
            pred_vmax = float(display_cfg['pred_vmax'])
            err_scale = choose_error_color_scale({m: err_arrays[m] for m in row_methods if m in err_arrays}, dim=2)
            err_norm = err_scale['norm']
            err_vmin = float(err_scale['vmin'])
            err_vmax = float(err_scale['vmax'])
            err_scale_mode = str(err_scale['mode'])
            err_extend = str(err_scale['extend'])
            if err_vmax <= 0:
                err_vmax = 1.0
                err_norm = Normalize(vmin=0.0, vmax=1.0)
                err_scale_mode = 'linear'
                err_extend = 'neither'
        else:
            display_cfg = {'mode': 'raw', 'offset': 0.0, 'pred_vmin': 0.0, 'pred_vmax': 1.0}
            pred_vmin = pred_vmax = 0.0
            err_scale = choose_error_color_scale({m: err_arrays[m] for m in row_methods if m in err_arrays}, dim=1)
            err_norm = err_scale['norm']
            err_vmin = float(err_scale['vmin'])
            err_vmax = float(err_scale['vmax'])
            err_scale_mode = str(err_scale['mode'])
            err_extend = str(err_scale['extend'])

        row_infos.append({
            'field': field,
            'arr_true': arr_true,
            'dim': dim,
            'row_methods': row_methods,
            'pred_arrays': pred_arrays,
            'err_arrays': err_arrays,
            'pred_sources': pred_sources,
            'err_sources': err_sources,
            'pred_vmin': pred_vmin,
            'pred_vmax': pred_vmax,
            'display_cfg': display_cfg,
            'err_norm': err_norm,
            'err_vmin': err_vmin,
            'err_vmax': err_vmax,
            'err_scale_mode': err_scale_mode,
            'err_extend': err_extend,
        })

    if not row_infos:
        print(f'[跳过] {case_name}：没有可绘制的字段数据')
        return None

    max_methods = max(len(r['row_methods']) for r in row_infos)
    n_rows = len(row_infos)
    max_cols = (1 + max_methods) + 1 + max_methods + 1
    height = 2.35 * n_rows + 2.9
    width = max(15.0, 2.08 * max_cols + 1.2)

    fig = plt.figure(figsize=(width, height), constrained_layout=False)
    outer = gridspec.GridSpec(
        nrows=n_rows + 1,
        ncols=1,
        figure=fig,
        height_ratios=[1.15] + [1.55] * n_rows,
        hspace=0.28,
    )

    ax_loss = fig.add_subplot(outer[0, 0])
    annotate_subplot_tag(ax_loss, subplot_letter_tag(panel_counter))
    panel_counter += 1
    if loss_curves:
        for method_name in pred_methods:
            if method_name not in loss_curves:
                continue
            x, y = loss_curves[method_name]
            pos = y > 0
            if np.count_nonzero(pos) >= 2:
                ax_loss.semilogy(x[pos], y[pos], linewidth=1.25, label=nice_method_name(method_name))
            else:
                ax_loss.plot(x, y, linewidth=1.25, label=nice_method_name(method_name))
        add_zoom_inset(ax_loss, loss_curves)
        ax_loss.set_ylabel('Loss', fontsize=10)
        ax_loss.set_title('', fontsize=1)
        ax_loss.grid(alpha=0.20, linewidth=0.5)
        ax_loss.legend(fontsize=8, ncol=max(1, min(len(pred_methods), 5)), frameon=False, loc='upper right', handlelength=1.5)
        ax_loss.tick_params(labelsize=8)
        ax_loss.xaxis.set_major_locator(MaxNLocator(nbins=6))
    else:
        ax_loss.text(0.5, 0.5, f'{case_label}\n(no loss history found)', ha='center', va='center', fontsize=12)
        ax_loss.set_axis_off()

    for i, row in enumerate(row_infos, start=1):
        field = row['field']
        arr_true = row['arr_true']
        dim = row['dim']
        row_methods = row['row_methods']
        pred_arrays = row['pred_arrays']
        err_arrays = row['err_arrays']
        pred_sources = row['pred_sources']
        err_sources = row['err_sources']
        pred_vmin = row['pred_vmin']
        pred_vmax = row['pred_vmax']
        display_cfg = row['display_cfg']
        err_norm = row['err_norm']
        err_vmin = row['err_vmin']
        err_vmax = row['err_vmax']
        err_scale_mode = row['err_scale_mode']
        err_extend = row['err_extend']

        field_label = display_field_name(field)
        field_label_bold = display_field_name_bold(field)
        show_x_row = (i == len(row_infos))
        row_cols = (1 + len(row_methods)) + 1 + len(row_methods) + 1

        inner = outer[i, 0].subgridspec(
            1,
            row_cols,
            width_ratios=[1.0] * (1 + len(row_methods)) + [0.060] + [1.0] * len(row_methods) + [0.060],
            wspace=0.090,
        )

        manifest_lines.append('')
        manifest_lines.append(f'[field] {field} ({field_label}), dim = {dim}')
        manifest_lines.append(f'  truth_source = {bundle["field_truth"][field]}')
        manifest_lines.append(f'  truth_stats = {basic_array_stats(arr_true)}')
        manifest_lines.append(f'  plotted_pred_methods = {", ".join(row_methods)}')
        manifest_lines.append(f'  prediction_display = {display_cfg.get("mode", "raw")}, offset = {float(display_cfg.get("offset", 0.0)):.8g}, vmin = {pred_vmin:.8g}, vmax = {pred_vmax:.8g}')
        manifest_lines.append(f'  error_colorbar = {err_scale_mode}, vmin = {err_vmin:.8g}, vmax = {err_vmax:.8g}, extend = {err_extend}')

        pred_last_mappable = None
        for j, title in enumerate(['Exact'] + [nice_method_name(m) for m in row_methods]):
            ax = fig.add_subplot(inner[0, j])
            annotate_subplot_tag(ax, subplot_letter_tag(panel_counter))
            panel_counter += 1
            show_y = (j == 0)
            if j == 0:
                if dim == 2:
                    arr_show = prepare_prediction_display_array(arr_true, dim, display_cfg)
                    mappable = plot_panel_2d(ax, arr_show, title, pred_vmin, pred_vmax, cmap='jet', norm=Normalize(vmin=pred_vmin, vmax=pred_vmax))
                else:
                    plot_panel_1d(ax, None, arr_true, title, is_error=False)
                    mappable = None
                clean_axes(ax, show_x=show_x_row, show_y=show_y, xlabel='x', ylabel=None)
                row_label_specs.append((ax, field_label_bold))
            else:
                method = row_methods[j - 1]
                arrp = pred_arrays.get(method)
                if arrp is None:
                    ax.set_axis_off()
                    continue
                if dim == 2:
                    arr_show = prepare_prediction_display_array(arrp, dim, display_cfg)
                    mappable = plot_panel_2d(ax, arr_show, title, pred_vmin, pred_vmax, cmap='jet', norm=Normalize(vmin=pred_vmin, vmax=pred_vmax))
                else:
                    plot_panel_1d(ax, arr_true, arrp, title, is_error=False)
                    mappable = None
                clean_axes(ax, show_x=show_x_row, show_y=False)
                manifest_lines.append(f'  pred[{method}]_source = {pred_sources.get(method, "missing")}')
                manifest_lines.append(f'  pred[{method}]_stats = {basic_array_stats(arrp)}')
            if dim == 2:
                ax.set_aspect('auto')
                ax.ticklabel_format(style='plain', useOffset=False, axis='both')
                if mappable is not None:
                    pred_last_mappable = mappable

        cax_pred = fig.add_subplot(inner[0, 1 + len(row_methods)])
        if dim == 2 and pred_last_mappable is not None:
            cb = fig.colorbar(pred_last_mappable, cax=cax_pred)
            simplify_colorbar(cb, scale_mode='linear')
        else:
            cax_pred.set_axis_off()

        err_start = 2 + len(row_methods)
        err_last_mappable = None
        for k, method in enumerate(row_methods):
            ax = fig.add_subplot(inner[0, err_start + k])
            annotate_subplot_tag(ax, subplot_letter_tag(panel_counter))
            panel_counter += 1
            arre = err_arrays.get(method)
            title = nice_method_name(method)
            if arre is None:
                ax.set_axis_off()
                manifest_lines.append(f'  err[{method}]_source = missing')
                continue
            if dim == 2:
                mappable_err = plot_panel_2d(ax, arre, title, err_vmin, err_vmax, cmap='jet', norm=err_norm)
                if mappable_err is not None:
                    err_last_mappable = mappable_err
                manifest_lines.append(f'  err[{method}]_colorbar = shared_{err_scale_mode}, vmin = {err_vmin:.8g}, vmax = {err_vmax:.8g}, extend = {err_extend}')
            else:
                plot_panel_1d(ax, None, arre, title, is_error=True)
            clean_axes(ax, show_x=show_x_row, show_y=False)
            manifest_lines.append(f'  err[{method}]_source = {err_sources.get(method, "missing")}')
            manifest_lines.append(f'  err[{method}]_stats = {basic_array_stats(arre)}')
            if dim == 2:
                ax.set_aspect('auto')
                ax.ticklabel_format(style='plain', useOffset=False, axis='both')

        cax_err = fig.add_subplot(inner[0, err_start + len(row_methods)])
        if dim == 2 and err_last_mappable is not None:
            cb_err = fig.colorbar(err_last_mappable, cax=cax_err, extend=err_extend)
            simplify_colorbar(cb_err, scale_mode=('log' if 'log' in err_scale_mode else 'linear'))
        else:
            cax_err.set_axis_off()

    fig.subplots_adjust(top=0.988, bottom=0.020, left=0.058, right=0.972, hspace=0.20)
    for ax_row, row_label in row_label_specs:
        pos = ax_row.get_position()
        x_text = max(0.003, pos.x0 - 0.020)
        y_text = 0.5 * (pos.y0 + pos.y1)
        fig.text(x_text, y_text, row_label, ha='right', va='center', fontsize=13.0, rotation=90)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(case_name)
    png_path = out_dir / f'{stem}_F2_style_summary_v18.png'
    pdf_path = out_dir / f'{stem}_F2_style_summary_v18.pdf'
    txt_path = out_dir / f'{stem}_F2_style_summary_v18.txt'
    save_figure_as_png_and_pdf(fig, png_path, pdf_path, dpi=300)
    plt.close(fig)
    write_case_manifest(txt_path, case_name, case_label, pred_methods, [r['field'] for r in row_infos], manifest_lines)
    print(f'[完成] {case_label} {case_name}')
    return png_path


def load_font(size: int):
    candidates = [
        'C:/Windows/Fonts/msyhbd.ttc',
        'C:/Windows/Fonts/msyh.ttc',
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/arialuni.ttf',
        '/System/Library/Fonts/PingFang.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        'arial.ttf',
        'DejaVuSans.ttf',
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def save_png_to_pdf(png_path: Path, pdf_path: Path, resolution: float = 300.0):
    img = Image.open(png_path).convert('RGB')
    img.save(pdf_path, 'PDF', resolution=resolution)


def save_figure_as_png_and_pdf(fig, png_path: Path, pdf_path: Path, dpi: int = 300):
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0.065, facecolor='white')
    save_png_to_pdf(png_path, pdf_path, resolution=float(dpi))


def build_vertical_stack(case_pngs: List[Tuple[int, str, Path]], out_dir: Path) -> Optional[Path]:
    if not case_pngs:
        return None
    images = []
    for idx, label, path in case_pngs:
        if not path.exists():
            continue
        img = Image.open(path).convert('RGB')
        images.append((idx, label, path, img))
    if not images:
        return None

    margin = 4
    max_w = max(img.width for _, _, _, img in images)
    resized = []
    total_h = margin
    max_label_width = 0
    font_size = max(24, max_w // 70)
    font = load_font(font_size)

    for idx, label, path, img in images:
        if img.width != max_w:
            new_h = int(round(img.height * max_w / img.width))
            img = img.resize((max_w, new_h), Image.Resampling.LANCZOS)
        bbox = font.getbbox(label)
        label_w = max(1, bbox[2] - bbox[0])
        max_label_width = max(max_label_width, label_w)
        resized.append((idx, label, path, img))
        total_h += img.height + margin

    label_gutter = max(70, min(140, max_label_width + 18))
    canvas = Image.new('RGB', (label_gutter + max_w + 2 * margin, total_h), 'white')
    y = margin
    manifest_lines = []
    for idx, label, path, img in resized:
        canvas.paste(img, (label_gutter + margin, y))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, y + 8), label, fill='black', font=font)
        manifest_lines.append(f'{label} = {path.name}')
        manifest_lines.append(f'  source_png = {path}')
        manifest_lines.append(f'  width = {img.width}, height = {img.height}')
        manifest_lines.append(f'  label_gutter = {label_gutter}')
        y += img.height + margin

    png_path = out_dir / 'all_cases_stacked_vertical_v18.png'
    pdf_path = out_dir / 'all_cases_stacked_vertical_v18.pdf'
    txt_path = out_dir / 'all_cases_stacked_vertical_v18.txt'
    canvas.save(png_path)
    save_png_to_pdf(png_path, pdf_path, resolution=300.0)
    txt_path.write_text('\n'.join(manifest_lines), encoding='utf-8')
    return png_path

def write_case_mapping(case_names: List[str], out_dir: Path):
    lines = []
    for idx, case_name in enumerate(case_names, start=1):
        lines.append(f'Case {idx}: {case_name}')
    (out_dir / 'case_index_mapping_v18.txt').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='按照 F2 样式批量重绘所有 case，排除指定文件夹并对近常量场启用增强显示，同时对二维绝对误差图启用逐面板细节增强')
    parser.add_argument('--root', type=str, default=None, help='根目录。默认取脚本所在目录')
    parser.add_argument('--out', type=str, default=None, help='输出目录。默认 root/_F2_style_summary_outputs_v18')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else script_dir
    out_dir = Path(args.out).resolve() if args.out else (root / '_F2_style_summary_outputs_v18')
    out_dir.mkdir(parents=True, exist_ok=True)

    layout = detect_root_layout(root)
    bundles: List[Dict] = []
    if layout == 'case_first':
        case_dirs = find_case_directories(root)
        bundles = [build_case_bundle(case_dir) for case_dir in case_dirs]
    elif layout == 'method_first':
        bundles = find_case_bundles_from_method_root(root)
    else:
        raise FileNotFoundError(
            f'没有在 {root} 下识别到可用的目录结构。支持 root/case/method/... 或 root/method/case/... 两种结构。'
        )

    bundles = [b for b in bundles if b.get('pred_methods') and b.get('fields')]
    if not bundles:
        raise FileNotFoundError(
            f'没有在 {root} 下找到可用的真解/预测数据。'
        )

    write_case_mapping([b['case_name'] for b in bundles], out_dir)

    generated: List[Tuple[int, str, Path]] = []
    for idx, bundle in enumerate(bundles, start=1):
        png_path = plot_case_summary(bundle, out_dir, case_index=idx)
        if png_path is not None:
            generated.append((idx, case_label_from_index(idx), png_path))

    stacked = build_vertical_stack(generated, out_dir)
    print('全部完成。')
    print(f'输出目录: {out_dir}')
    print(f'识别布局: {layout}')
    if stacked is not None:
        print(f'竖向总图: {stacked}')


if __name__ == '__main__':
    main()
