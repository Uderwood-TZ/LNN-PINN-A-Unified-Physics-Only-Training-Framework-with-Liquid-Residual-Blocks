#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import MaxNLocator, FuncFormatter

CASE_NAME = 'Laplace2D_UnitSquare_DirTopBot_NeuLeftRight'
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
    if path.is_dir() and canonical_algo_name(path.name) is None:
        algos = []
        for p in path.iterdir():
            if p.is_dir() and canonical_algo_name(p.name):
                algos.append(p)
        if algos:
            return path
    return path


def maybe_unzip(root_arg: Optional[str]) -> Path:
    if root_arg:
        return detect_root(Path(root_arg).expanduser().resolve())
    cwd = Path.cwd()
    direct = cwd / CASE_NAME
    if direct.is_dir():
        return detect_root(direct)
    zip_path = cwd / f'{CASE_NAME}.zip'
    if zip_path.is_file():
        out_dir = cwd / f'{CASE_NAME}_unzipped'
        if out_dir.exists():
            return detect_root(out_dir / CASE_NAME if (out_dir / CASE_NAME).is_dir() else out_dir)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(out_dir)
        return detect_root(out_dir / CASE_NAME if (out_dir / CASE_NAME).is_dir() else out_dir)
    return detect_root(cwd)


def find_algorithm_dirs(root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in root.iterdir():
        if not p.is_dir():
            continue
        canon = canonical_algo_name(p.name)
        if canon:
            out[canon] = p
    return {k: out[k] for k in ALGO_ORDER if k in out}


def find_lr_subdirs(algo_dir: Path) -> Dict[str, Path]:
    lr_dirs: Dict[str, Path] = {}
    for p in algo_dir.iterdir():
        if p.is_dir() and re.fullmatch(r'1e-\d+', p.name):
            lr_dirs[normalize_lr_label(p.name)] = p
    if lr_dirs:
        return lr_dirs
    # flat layout: infer lr from file names; default to 1e-3 if absent
    files = list(algo_dir.iterdir())
    lrs = set()
    for f in files:
        m = re.search(r'lr(1e-\d+)', f.name)
        if m:
            lrs.add(normalize_lr_label(m.group(1)))
    if not lrs:
        lrs.add(normalize_lr_label('1e-3'))
    return {lr: algo_dir for lr in sorted(lrs, key=lr_sort_key)}




def normalize_lr_label(s: str) -> str:
    m = re.fullmatch(r'1e-0*(\d+)', s.strip())
    if m:
        return f'1e-{int(m.group(1))}'
    return s.strip()

def lr_sort_key(s: str) -> Tuple[int, str]:
    m = re.fullmatch(r'1e-(\d+)', s)
    return (int(m.group(1)) if m else 999, s)


def choose_file(folder: Path, patterns: List[str]) -> Optional[Path]:
    candidates = [p for p in folder.iterdir() if p.is_file()]
    lowered = [(p, p.name.lower()) for p in candidates]
    for pat in patterns:
        rx = re.compile(pat)
        for p, name in lowered:
            if rx.search(name):
                return p
    return None


def load_numeric_txt(path: Path) -> np.ndarray:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first = f.readline().strip().lower()
    skiprows = 1 if any(tok in first for tok in ['x', 'y', 'z', 'epoch', 'loss']) and any(c.isalpha() for c in first) else 0
    try:
        arr = np.loadtxt(path, dtype=float, skiprows=skiprows)
    except Exception:
        arr = np.genfromtxt(path, dtype=float, skip_header=skiprows, invalid_raise=False)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    mask = np.all(np.isfinite(arr), axis=1)
    return arr[mask]


def extract_field_data(folder: Path, algo: str, lr: str) -> Dict[str, Optional[Path]]:
    d: Dict[str, Optional[Path]] = {}
    # true / pred / error
    d['true'] = choose_file(folder, [r'phi_true.*\.txt$', r'true_xyz.*\.txt$'])
    d['pred'] = choose_file(folder, [r'phi_pred.*\.txt$', r'pred_xyz.*\.txt$'])
    d['err'] = choose_file(folder, [r'phi_abs_error.*\.txt$', r'phi_maxerror.*\.txt$', r'error_xyz.*\.txt$', r'maxerror_xyz.*\.txt$'])
    d['loss'] = choose_file(folder, [r'loss_per_epoch.*\.txt$'])
    return d


def parse_loss(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    arr = load_numeric_txt(path)
    if arr.size == 0:
        return None
    if arr.shape[1] == 1:
        y = arr[:, 0]
        x = np.arange(1, len(y) + 1)
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


def plot_2d(ax, a: np.ndarray, title: str, vmin: float, vmax: float, *, norm=None):
    try:
        X, Y, Z = make_grid(a)
        m = ax.pcolormesh(X, Y, Z, shading='auto', cmap='jet', vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm)
    except Exception:
        m = ax.tricontourf(a[:, 0], a[:, 1], a[:, 2], levels=150, cmap='jet', vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm)
    ax.set_title(title, fontsize=8, pad=2)
    ax.tick_params(labelsize=6, length=1.8, pad=1)
    return m


def simplify_colorbar(cb, *, log_mode: bool = False):
    cb.ax.tick_params(labelsize=5.2, length=1.5, pad=1.1, labelright=True, labelleft=False, right=True, left=False)
    cb.ax.yaxis.set_ticks_position('right')
    cb.ax.yaxis.set_label_position('right')
    cb.ax.yaxis.get_offset_text().set_visible(False)
    if log_mode:
        cb.ax.minorticks_off()
        cb.formatter = FuncFormatter(lambda x, pos: f'{x:.0e}')
    else:
        cb.ax.yaxis.set_major_locator(MaxNLocator(3))
        cb.formatter = FuncFormatter(lambda x, pos: f'{x:.2g}')
    cb.update_ticks()


def collect_case(root: Path):
    algo_dirs = find_algorithm_dirs(root)
    if not algo_dirs:
        raise RuntimeError(f'No algorithm folders found under {root}')
    per_algo_lr = {algo: find_lr_subdirs(d) for algo, d in algo_dirs.items()}
    all_lrs = sorted({lr for d in per_algo_lr.values() for lr in d.keys()}, key=lr_sort_key)
    case = {}
    for lr in all_lrs:
        by_algo = {}
        for algo, lr_map in per_algo_lr.items():
            if lr not in lr_map:
                continue
            folder = lr_map[lr]
            files = extract_field_data(folder, algo, lr)
            by_algo[algo] = files
        case[lr] = by_algo
    return case


def compute_error(true_arr: Optional[np.ndarray], pred_arr: Optional[np.ndarray], err_arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if true_arr is not None and pred_arr is not None and true_arr.shape == pred_arr.shape and true_arr.shape[1] >= 3:
        out = pred_arr.copy()
        out[:, 2] = np.abs(pred_arr[:, 2] - true_arr[:, 2])
        return out
    return err_arr


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
        loss_map[algo] = parse_loss(info['loss'])
        true_arr = load_numeric_txt(info['true']) if info.get('true') else None
        pred_arr = load_numeric_txt(info['pred']) if info.get('pred') else None
        err_arr = load_numeric_txt(info['err']) if info.get('err') else None
        true_map[algo] = true_arr
        pred_map[algo] = pred_arr
        err_map[algo] = compute_error(true_arr, pred_arr, err_arr)
        if ref_true is None and true_arr is not None:
            ref_true = true_arr

    algos_present = [a for a in ALGO_ORDER if a in by_algo]
    pred_arrays = [ref_true] + [pred_map.get(a) for a in algos_present]
    valid_pred = [a for a in pred_arrays if a is not None]
    z_pred = np.concatenate([field_values(a) for a in valid_pred])
    pred_vmin, pred_vmax = float(np.nanmin(z_pred)), float(np.nanmax(z_pred))
    if not np.isfinite(pred_vmin) or not np.isfinite(pred_vmax) or pred_vmin == pred_vmax:
        pred_vmin, pred_vmax = 0.0, 1.0

    valid_err = [err_map[a] for a in algos_present if err_map.get(a) is not None]
    if valid_err:
        zerr = np.concatenate([field_values(a) for a in valid_err])
        zerr = zerr[np.isfinite(zerr) & (zerr >= 0)]
        emax = float(np.nanmax(zerr)) if zerr.size else 1.0
        positive = zerr[zerr > 0]
        emin = float(np.nanpercentile(positive, 5)) if positive.size else max(emax * 1e-3, 1e-12)
        if not np.isfinite(emin) or emin <= 0:
            emin = max(emax * 1e-3, 1e-12)
        if not np.isfinite(emax) or emax <= 0:
            err_norm = Normalize(vmin=0.0, vmax=1.0)
        else:
            err_norm = LogNorm(vmin=min(emin, emax), vmax=emax)
    else:
        err_norm = Normalize(vmin=0.0, vmax=1.0)

    ncols = 1 + len(algos_present) + 1 + len(algos_present) + 1
    fig = plt.figure(figsize=(2.2 * ncols, 5.0), dpi=180)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.5], hspace=0.28)

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

    inner = outer[1, 0].subgridspec(1, ncols, width_ratios=[1.0] * (1 + len(algos_present)) + [0.065] + [1.0] * len(algos_present) + [0.065], wspace=0.08)

    m_pred = None
    ax = fig.add_subplot(inner[0, 0])
    if ref_true is not None:
        m_pred = plot_2d(ax, ref_true, 'Exact', pred_vmin, pred_vmax)
    else:
        ax.set_axis_off()
    ax.set_ylabel('phi', fontsize=8)

    for i, algo in enumerate(algos_present, start=1):
        ax = fig.add_subplot(inner[0, i])
        arr = pred_map.get(algo)
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

    for ax in fig.axes:
        if ax in [cax_pred, cax_err, ax_loss]:
            continue
        ax.tick_params(labelsize=5.5, length=1.5, pad=1)

    fig.subplots_adjust(left=0.036, right=0.965, top=0.95, bottom=0.09)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{CASE_NAME}_algorithm_comparison_{lr.replace(".", "p")}.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default=None, help='case root containing the four algorithm folders')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    root = maybe_unzip(args.root)
    # descend one level if zip unpack root contains the case folder
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
