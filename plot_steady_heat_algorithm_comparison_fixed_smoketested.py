#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import zipfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator

CASE_NAME = '稳态硅板加热'
FIELD_NAME = 'T'
ALGO_ORDER = ['LNN-PINN', 'RA-PINN', 'XPINN', 'PINN']
ALGO_ALIASES = {
    'LNN-PINN': 'LNN-PINN',
    'LNN_PINN': 'LNN-PINN',
    'PINN': 'PINN',
    'XPINN': 'XPINN',
    'RA-PINN': 'RA-PINN',
    'RA_PINN': 'RA-PINN',
    'RA-PIINN': 'RA-PINN',
    'RAPINN': 'RA-PINN',
}

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', message='Glyph .* missing from font')


def canonical_algo_name(name: str) -> Optional[str]:
    key = name.strip()
    if key in ALGO_ALIASES:
        return ALGO_ALIASES[key]
    up = re.sub(r'[^A-Za-z0-9]+', '', key).upper()
    if up == 'LNNPINN':
        return 'LNN-PINN'
    if up == 'PINN':
        return 'PINN'
    if up == 'XPINN':
        return 'XPINN'
    if up in {'RAPINN', 'RAPIINN'}:
        return 'RA-PINN'
    return None


def detect_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / CASE_NAME).is_dir():
        return (path / CASE_NAME).resolve()
    return path


def maybe_unzip(root_arg: Optional[str]) -> Path:
    if root_arg:
        return detect_root(Path(root_arg))
    cwd = Path.cwd()
    if (cwd / CASE_NAME).is_dir():
        return (cwd / CASE_NAME).resolve()
    if (cwd / f'{CASE_NAME}.zip').is_file():
        out_dir = cwd / f'{CASE_NAME}_unzipped'
        if not out_dir.exists():
            with zipfile.ZipFile(cwd / f'{CASE_NAME}.zip', 'r') as zf:
                zf.extractall(out_dir)
        return detect_root(out_dir)
    return detect_root(cwd)


def normalize_lr_label(s: str) -> str:
    s = s.strip()
    m = re.fullmatch(r'1e-0*(\d+)', s)
    if m:
        return f'1e-{int(m.group(1))}'
    return s


def lr_sort_key(s: str) -> Tuple[int, str]:
    m = re.fullmatch(r'1e-(\d+)', s)
    return (int(m.group(1)) if m else 999, s)


def find_algorithm_dirs(root: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name == 'matlab有限元':
            continue
        canon = canonical_algo_name(p.name)
        if canon:
            found[canon] = p
    return {k: found[k] for k in ALGO_ORDER if k in found}


def find_lr_subdirs(algo_dir: Path) -> Dict[str, Path]:
    lr_dirs: Dict[str, Path] = {}
    for p in algo_dir.iterdir():
        if p.is_dir() and re.fullmatch(r'1e-\d+', p.name):
            lr_dirs[normalize_lr_label(p.name)] = p
    if lr_dirs:
        return lr_dirs
    return {'1e-3': algo_dir}


def choose_file_recursive(folder: Path, patterns: List[str]) -> Optional[Path]:
    files = [p for p in folder.rglob('*') if p.is_file()]
    lowered = [(p, p.name.lower()) for p in files]
    for pat in patterns:
        rx = re.compile(pat)
        for p, name in lowered:
            if rx.search(name):
                return p
    return None


def load_numeric_txt(path: Path) -> np.ndarray:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline().strip().lower()
    skiprows = 1 if any(tok in first for tok in ['x', 'y', 'value', 'epoch', 'loss']) and any(c.isalpha() for c in first) else 0
    try:
        arr = np.loadtxt(path, dtype=float, skiprows=skiprows)
    except Exception:
        arr = np.genfromtxt(path, dtype=float, skip_header=skiprows, invalid_raise=False)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        return arr.reshape(0, 1)
    mask = np.all(np.isfinite(arr), axis=1)
    return arr[mask]


def extract_algo_files(folder: Path) -> Dict[str, Optional[Path]]:
    return {
        'true': choose_file_recursive(folder, [r'temperature_true.*\.txt$', r'\breal\.txt$']),
        'pred': choose_file_recursive(folder, [r'temperature_pred.*\.txt$']),
        'err': choose_file_recursive(folder, [r'temperature_maxerror.*\.txt$', r'temperature_abs_error.*\.txt$']),
        'loss': choose_file_recursive(folder, [r'loss_history_step_by_step.*\.txt$', r'loss_per_epoch.*\.txt$']),
    }


def parse_loss(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    arr = load_numeric_txt(path)
    if arr.size == 0:
        return None
    if arr.shape[1] == 1:
        x = np.arange(1, len(arr) + 1, dtype=float)
        y = arr[:, 0]
    else:
        x = arr[:, 0]
        y = arr[:, 1]
    m = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) == 0:
        return None
    return np.column_stack([x[m], y[m]])


def field_values(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2 and a.shape[1] >= 3:
        return a[:, 2]
    return a.ravel()


def copy_with_new_z(arr: np.ndarray, znew: np.ndarray) -> np.ndarray:
    out = np.array(arr, dtype=float, copy=True)
    out[:, 2] = np.asarray(znew, dtype=float)
    return out


def make_grid(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = a[:, 0]
    y = a[:, 1]
    z = a[:, 2]
    xu = np.unique(np.round(x, 12))
    yu = np.unique(np.round(y, 12))
    nx, ny = len(xu), len(yu)
    if nx * ny == len(a):
        x_map = {v: i for i, v in enumerate(xu)}
        y_map = {v: i for i, v in enumerate(yu)}
        Z = np.full((ny, nx), np.nan)
        for xi, yi, zi in zip(np.round(x, 12), np.round(y, 12), z):
            Z[y_map[yi], x_map[xi]] = zi
        X, Y = np.meshgrid(xu, yu)
        return X, Y, Z
    raise ValueError('irregular grid')


def detrend_plane_display(arr: np.ndarray) -> np.ndarray:
    aa = np.asarray(arr, dtype=float)
    x = aa[:, 0]
    y = aa[:, 1]
    z = aa[:, 2]
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.count_nonzero(mask) < 6:
        return arr
    A = np.column_stack([x[mask], y[mask], np.ones(np.count_nonzero(mask))])
    coef, *_ = np.linalg.lstsq(A, z[mask], rcond=None)
    trend = aa[:, 0] * coef[0] + aa[:, 1] * coef[1] + coef[2]
    return copy_with_new_z(arr, z - trend)


def signed_power_display(arr: np.ndarray, gamma: float = 0.45) -> np.ndarray:
    vals = field_values(arr)
    out = np.sign(vals) * np.power(np.abs(vals), gamma)
    return copy_with_new_z(arr, out)


def enhance_temperature_display(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    vals = field_values(arr)
    centered = copy_with_new_z(arr, vals - float(np.nanmedian(vals)))
    centered = detrend_plane_display(centered)
    centered = signed_power_display(centered, gamma=0.45)
    return centered


def data_range_for_prediction(arrays: List[np.ndarray]) -> Tuple[float, float]:
    vals = []
    for a in arrays:
        if a is None:
            continue
        z = field_values(a)
        z = z[np.isfinite(z)]
        if z.size:
            vals.append(z)
    if not vals:
        return 0.0, 1.0
    zall = np.concatenate(vals)
    vmin = float(np.nanmin(zall))
    vmax = float(np.nanmax(zall))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return 0.0, 1.0
    return vmin, vmax


def prediction_display_range(display_true: Optional[np.ndarray], display_preds: List[Optional[np.ndarray]]) -> Tuple[float, float]:
    if display_true is None and not any(a is not None for a in display_preds):
        return 0.0, 1.0
    if display_true is None:
        return data_range_for_prediction([a for a in display_preds if a is not None])
    tv = field_values(display_true)
    tv = tv[np.isfinite(tv)]
    if tv.size == 0:
        return data_range_for_prediction([display_true] + [a for a in display_preds if a is not None])
    absq = float(np.nanpercentile(np.abs(tv), 95.0))
    if not np.isfinite(absq) or absq <= 0:
        absq = float(np.nanpercentile(np.abs(tv), 99.0)) if tv.size else 1.0
    if not np.isfinite(absq) or absq <= 0:
        absq = float(np.nanmax(np.abs(tv))) if tv.size else 1.0
    if not np.isfinite(absq) or absq <= 0:
        absq = 1.0
    return -1.00 * absq, 1.00 * absq


def compute_error(true_arr: Optional[np.ndarray], pred_arr: Optional[np.ndarray], err_arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if true_arr is not None and pred_arr is not None and true_arr.shape == pred_arr.shape and true_arr.shape[1] >= 3:
        out = np.array(pred_arr, dtype=float, copy=True)
        out[:, 2] = np.abs(pred_arr[:, 2] - true_arr[:, 2])
        return out
    return err_arr


def plot_2d(ax, a: np.ndarray, title: str, vmin: float, vmax: float, *, norm=None):
    try:
        X, Y, Z = make_grid(a)
        m = ax.pcolormesh(X, Y, Z, shading='auto', cmap='jet', vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm)
    except Exception:
        m = ax.tricontourf(a[:, 0], a[:, 1], a[:, 2], levels=150, cmap='jet', vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm)
    ax.set_title(title, fontsize=8, pad=2)
    ax.tick_params(labelsize=5.4, length=1.4, pad=1)
    ax.set_aspect('equal', adjustable='box')
    return m


def simplify_colorbar(cb, *, log_mode: bool = False):
    cb.ax.tick_params(labelsize=3.9, length=1.0, pad=0.9, labelright=True, labelleft=False, right=True, left=False)
    cb.ax.yaxis.set_ticks_position('right')
    cb.ax.yaxis.set_label_position('right')
    cb.ax.yaxis.get_offset_text().set_visible(False)
    if log_mode:
        cb.ax.minorticks_off()
        cb.formatter = FuncFormatter(lambda x, pos: f'{x:.0e}')
    else:
        cb.ax.yaxis.set_major_locator(MaxNLocator(2))
        cb.formatter = FuncFormatter(lambda x, pos: f'{x:.2g}')
    cb.update_ticks()


def collect_case(root: Path):
    algo_dirs = find_algorithm_dirs(root)
    if not algo_dirs:
        raise RuntimeError(f'No algorithm folders found under {root}')
    per_algo_lr = {algo: find_lr_subdirs(d) for algo, d in algo_dirs.items()}
    all_lrs = sorted({lr for d in per_algo_lr.values() for lr in d.keys()}, key=lr_sort_key)
    case: Dict[str, Dict[str, Dict[str, Optional[Path]]]] = {}
    for lr in all_lrs:
        by_algo: Dict[str, Dict[str, Optional[Path]]] = {}
        for algo, lr_map in per_algo_lr.items():
            if lr not in lr_map:
                continue
            by_algo[algo] = extract_algo_files(lr_map[lr])
        case[lr] = by_algo
    return case


def plot_lr_comparison(root: Path, lr: str, by_algo: Dict[str, Dict[str, Optional[Path]]], out_dir: Path) -> Path:
    loss_map: Dict[str, Optional[np.ndarray]] = {}
    true_map: Dict[str, Optional[np.ndarray]] = {}
    pred_map: Dict[str, Optional[np.ndarray]] = {}
    err_map: Dict[str, Optional[np.ndarray]] = {}

    ref_true = None
    for algo in ALGO_ORDER:
        info = by_algo.get(algo)
        if not info:
            continue
        loss_map[algo] = parse_loss(info.get('loss'))
        true_arr = load_numeric_txt(info['true']) if info.get('true') else None
        pred_arr = load_numeric_txt(info['pred']) if info.get('pred') else None
        err_arr = load_numeric_txt(info['err']) if info.get('err') else None
        true_map[algo] = true_arr
        pred_map[algo] = pred_arr
        err_map[algo] = compute_error(true_arr, pred_arr, err_arr)
        if ref_true is None and true_arr is not None:
            ref_true = true_arr

    algos_present = [a for a in ALGO_ORDER if a in by_algo]

    display_true = enhance_temperature_display(ref_true)
    display_preds = {a: enhance_temperature_display(pred_map.get(a)) for a in algos_present}
    pred_vmin, pred_vmax = prediction_display_range(display_true, [display_preds[a] for a in algos_present])

    valid_err = [err_map[a] for a in algos_present if err_map.get(a) is not None]
    if valid_err:
        zerr = np.concatenate([field_values(a) for a in valid_err])
        zerr = zerr[np.isfinite(zerr) & (zerr >= 0)]
        emax = float(np.nanmax(zerr)) if zerr.size else 1.0
        pos = zerr[zerr > 0]
        emin = float(np.nanpercentile(pos, 5)) if pos.size else max(emax * 1e-3, 1e-12)
        if not np.isfinite(emin) or emin <= 0:
            emin = max(emax * 1e-3, 1e-12)
        if not np.isfinite(emax) or emax <= 0:
            err_norm = Normalize(vmin=0.0, vmax=1.0)
        else:
            err_norm = LogNorm(vmin=min(emin, emax), vmax=emax)
    else:
        err_norm = Normalize(vmin=0.0, vmax=1.0)

    ncols = 1 + len(algos_present) + 1 + len(algos_present) + 1
    fig = plt.figure(figsize=(2.15 * ncols, 5.25), dpi=180)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55], hspace=0.26)

    ax_loss = fig.add_subplot(outer[0, 0])
    for algo in algos_present:
        loss = loss_map.get(algo)
        if loss is None:
            continue
        ax_loss.semilogy(loss[:, 0], np.abs(loss[:, 1]), lw=0.9, label=algo)
    ax_loss.set_ylabel('Loss', fontsize=8)
    ax_loss.set_title(f'{CASE_NAME}   algorithm comparison   {lr}', fontsize=10, pad=2)
    ax_loss.grid(True, alpha=0.22, lw=0.4)
    ax_loss.tick_params(labelsize=6)
    if algos_present:
        ax_loss.legend(loc='upper right', fontsize=6, ncol=min(len(algos_present), 4), frameon=False)

    inner = outer[1, 0].subgridspec(
        1,
        ncols,
        width_ratios=[1.0] * (1 + len(algos_present)) + [0.065] + [1.0] * len(algos_present) + [0.065],
        wspace=0.11,
    )

    m_pred = None
    ax = fig.add_subplot(inner[0, 0])
    if display_true is not None:
        m_pred = plot_2d(ax, display_true, 'Exact', pred_vmin, pred_vmax)
    else:
        ax.set_axis_off()
    ax.set_ylabel(FIELD_NAME, fontsize=8)

    for i, algo in enumerate(algos_present, start=1):
        ax = fig.add_subplot(inner[0, i])
        arr = display_preds.get(algo)
        if arr is not None:
            m_pred = plot_2d(ax, arr, algo, pred_vmin, pred_vmax)
        else:
            ax.set_axis_off()

    cax_pred = fig.add_subplot(inner[0, 1 + len(algos_present)])
    if m_pred is not None:
        cbp = fig.colorbar(m_pred, cax=cax_pred)
        simplify_colorbar(cbp, log_mode=False)
    else:
        cax_pred.set_axis_off()

    m_err = None
    start = 2 + len(algos_present)
    for i, algo in enumerate(algos_present):
        ax = fig.add_subplot(inner[0, start + i])
        arr = err_map.get(algo)
        if arr is not None:
            m_err = plot_2d(ax, arr, algo, 0.0, 1.0, norm=err_norm)
        else:
            ax.set_axis_off()

    cax_err = fig.add_subplot(inner[0, start + len(algos_present)])
    if m_err is not None:
        cbe = fig.colorbar(m_err, cax=cax_err)
        simplify_colorbar(cbe, log_mode=isinstance(err_norm, LogNorm))
    else:
        cax_err.set_axis_off()

    for ax0 in fig.axes:
        if ax0 in [cax_pred, cax_err, ax_loss]:
            continue
        ax0.tick_params(labelsize=5.2, length=1.3, pad=1)

    fig.subplots_adjust(left=0.036, right=0.948, top=0.95, bottom=0.088)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{CASE_NAME}_algorithm_comparison_{lr.replace('.', 'p')}.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default=None, help='case root containing the algorithm folders')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    root = maybe_unzip(args.root)
    if not find_algorithm_dirs(root) and (root / CASE_NAME).is_dir():
        root = root / CASE_NAME
    if not root.exists():
        raise SystemExit(f'Root not found: {root}')
    out_dir = Path(args.out) if args.out else root / 'algorithm_plot_outputs'

    case = collect_case(root)
    outputs = []
    for lr in sorted(case.keys(), key=lr_sort_key):
        outputs.append(plot_lr_comparison(root, lr, case[lr], out_dir))

    print(f'Root: {root}')
    print(f'Output directory: {out_dir}')
    for p in outputs:
        print(p)


if __name__ == '__main__':
    main()
