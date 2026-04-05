
import argparse
import math
import re
import zipfile
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator, FuncFormatter, LogFormatterSciNotation
from matplotlib import font_manager

try:
    from PIL import Image
except Exception:
    Image = None

ROOT_CANDIDATES = ["多参数性能", "multi_param_performance"]
ZIP_CANDIDATES = ["多参数性能.zip", "multi_param_performance.zip"]
JET = "jet"
TXT_CACHE = {}

FIELD_ORDER_HINT = {
    "u": 0, "uy": 1, "epsilon": 2, "n": 3, "pi": 4,
    "pi_xx": 5, "pi_xy": 6, "pi_yy": 7, "vx": 8, "vy": 9, "phi": 10, "t": 11
}


def configure_fonts():
    warnings.filterwarnings("ignore", message=r"Glyph .* missing from font", category=UserWarning)
    warnings.filterwarnings("ignore", message=r"findfont: Font family .* not found\.", category=UserWarning)
    names = {f.name for f in font_manager.fontManager.ttflist}
    preferred = [n for n in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "PingFang SC", "Heiti SC"] if n in names]
    if preferred:
        matplotlib.rcParams["font.family"] = preferred[0]
    else:
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False


def sanitize_name(name: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]+', "_", name)
    s = re.sub(r"\s+", " ", s).strip()
    return s or "untitled"


def numeric_lr_value(s: str) -> float:
    try:
        return float(s)
    except Exception:
        pass
    m = re.search(r"([0-9.]+e[+\-]?\d+)", s.lower())
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return -math.inf


def resolve_root(cli_root: str | None) -> Path:
    if cli_root:
        p = Path(cli_root).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"root not found: {p}")
    here = Path(__file__).resolve().parent
    for name in ROOT_CANDIDATES:
        p = here / name
        if p.is_dir():
            return p
    for name in ZIP_CANDIDATES:
        z = here / name
        if z.is_file():
            out = here / z.stem
            if not out.is_dir():
                with zipfile.ZipFile(z, "r") as f:
                    f.extractall(here)
            if out.is_dir():
                return out
            subs = [d for d in here.iterdir() if d.is_dir() and d.name != "__pycache__"]
            for d in subs:
                if d.name == z.stem or "多参数性能" in d.name or "multi" in d.name.lower():
                    return d
    raise FileNotFoundError("未找到多参数性能文件夹，也未找到可自动解压的多参数性能.zip")


def is_numeric_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith("#"):
        return False
    s = s.replace(",", " ").replace("\t", " ")
    parts = [p for p in s.split() if p]
    if not parts:
        return False
    good = 0
    for p in parts:
        try:
            float(p)
            good += 1
        except Exception:
            return False
    return good > 0


def count_skiprows(path: Path) -> int:
    skip = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if is_numeric_line(line):
                break
            skip += 1
            if skip > 8:
                break
    return skip


def load_numeric_txt(path: Path):
    key = str(path.resolve())
    if key in TXT_CACHE:
        return TXT_CACHE[key]
    try:
        skiprows = count_skiprows(path)
        arr = np.loadtxt(path, dtype=float, comments="#", skiprows=skiprows)
        arr = np.asarray(arr)
        if arr.size == 0:
            TXT_CACHE[key] = None
            return None
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim > 1:
            arr = arr[np.isfinite(arr).all(axis=1)]
        else:
            arr = arr[np.isfinite(arr)]
        TXT_CACHE[key] = arr if arr.size else None
        return TXT_CACHE[key]
    except Exception:
        TXT_CACHE[key] = None
        return None


def is_loss_file(name: str) -> bool:
    n = name.lower()
    if not n.endswith(".txt"):
        return False
    if "loss" not in n:
        return False
    bad = ["metrics", "runtime", "summary", "time", "cost", "compute_cost"]
    return not any(b in n for b in bad)


def classify_txt_kind(path: Path):
    n = path.name.lower()
    if is_loss_file(path.name):
        return "loss"
    if not n.endswith(".txt"):
        return None
    bad = ["metrics", "runtime", "summary", "compute_cost", "run_summary", "real_time"]
    if any(b in n for b in bad):
        return None
    if "pred" in n:
        return "pred"
    if "abs_error" in n or "maxerror" in n:
        return "abs_error"
    if "true" in n or "exact" in n or n == "real.txt":
        return "true"
    return None


def infer_field(stem: str, kind: str) -> str:
    s = stem.lower().replace("-", "_")
    if "temperature" in s or s == "real":
        return "T"
    for token in ["pi_xx", "pi_xy", "pi_yy", "epsilon", "uy", "vx", "vy", "phi", "pi", "u", "p", "n", "t"]:
        if re.search(rf"(^|_)({re.escape(token)})(_|$)", s):
            return token
    s = re.sub(r"lr[0-9eE.+\-]+", "", s)
    for word in ["true", "pred", "abs", "error", "maxerror", "xyz", "lnn", "pinn", "result", "data"]:
        s = s.replace(word, "_")
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t]
    return toks[-1] if toks else stem


def parse_loss(path: Path):
    arr = load_numeric_txt(path)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        y = arr
    elif arr.shape[1] == 1:
        y = arr[:, 0]
    else:
        y = arr[:, -1]
    x = np.arange(1, len(y) + 1, dtype=float)
    mask = np.isfinite(y)
    if not np.any(mask):
        return None
    return x[mask], y[mask]


def normalize_points(arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        x = np.arange(arr.size, dtype=float)
        y = arr
        idx = np.argsort(x)
        return ("1d", x[idx], y[idx])
    if arr.shape[1] == 2:
        x = arr[:, 0]
        y = arr[:, 1]
        idx = np.argsort(x)
        return ("1d", x[idx], y[idx])
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]
    return ("2d", x, y, z)


def reconstruct_grid(x, y, z):
    xu, xi = np.unique(x, return_inverse=True)
    yu, yi = np.unique(y, return_inverse=True)
    nx = xu.size
    ny = yu.size
    if nx * ny > 600000:
        return None
    grid = np.full((yu.size, xu.size), np.nan, dtype=float)
    grid[yi, xi] = z
    X, Y = np.meshgrid(xu, yu)
    return X, Y, grid


def discover_bundle(case_dir: Path):
    method_dirs = [d for d in case_dir.iterdir() if d.is_dir()]
    if not method_dirs:
        return None
    method_dir = sorted(method_dirs, key=lambda p: p.name)[0]
    lr_dirs = [d for d in method_dir.iterdir() if d.is_dir()]
    if not lr_dirs:
        return None
    lr_dirs = sorted(lr_dirs, key=lambda p: numeric_lr_value(p.name), reverse=True)

    fields = {}
    loss_paths = {}
    for lr_dir in lr_dirs:
        lr = lr_dir.name
        for p in lr_dir.rglob("*.txt"):
            if not p.is_file():
                continue
            kind = classify_txt_kind(p)
            if kind is None:
                continue
            if kind == "loss":
                old = loss_paths.get(lr)
                if old is None or ("history" in p.name.lower()) or ("epoch" in p.name.lower()) or ("step" in p.name.lower()):
                    loss_paths[lr] = p
                continue
            field = infer_field(p.stem, kind)
            fields.setdefault(field, {}).setdefault(lr, {})[kind] = p

    if not fields:
        return None

    field_names = sorted(fields.keys(), key=lambda k: (FIELD_ORDER_HINT.get(k.lower(), 999), k.lower()))
    return {
        "case_name": case_dir.name,
        "case_dir": case_dir,
        "lr_dirs": lr_dirs,
        "field_paths": fields,
        "field_names": field_names,
        "loss_paths": loss_paths,
    }


def align_1d(true_xy, pred_xy):
    xt, yt = true_xy
    xp, yp = pred_xy
    if xt.size == xp.size and np.allclose(xt, xp, rtol=1e-9, atol=1e-12):
        return xt, yt, yp
    yp2 = np.interp(xt, xp, yp)
    return xt, yt, yp2


def align_2d(true_xyz, pred_xyz):
    xt, yt, zt = true_xyz
    xp, yp, zp = pred_xyz
    if xt.size == xp.size and np.allclose(xt, xp, rtol=1e-9, atol=1e-12) and np.allclose(yt, yp, rtol=1e-9, atol=1e-12):
        return xt, yt, zt, zp
    key_t = np.round(np.column_stack([xt, yt]), 12)
    key_p = np.round(np.column_stack([xp, yp]), 12)
    map_p = {(a, b): v for (a, b), v in zip(key_p, zp)}
    keep_x = []
    keep_y = []
    keep_t = []
    keep_p = []
    for (a, b), v in zip(key_t, zt):
        vv = map_p.get((a, b))
        if vv is not None:
            keep_x.append(a)
            keep_y.append(b)
            keep_t.append(v)
            keep_p.append(vv)
    return np.asarray(keep_x), np.asarray(keep_y), np.asarray(keep_t), np.asarray(keep_p)


def load_case_data(bundle):
    loaded = {
        "case_name": bundle["case_name"],
        "lr_values": [d.name for d in bundle["lr_dirs"]],
        "losses": {},
        "fields": {},
        "dims": {},
    }
    for lr, p in bundle["loss_paths"].items():
        val = parse_loss(p)
        if val is not None:
            loaded["losses"][lr] = val

    for field in bundle["field_names"]:
        loaded["fields"][field] = {}
        true_cache = None
        for lr in loaded["lr_values"]:
            trip = bundle["field_paths"].get(field, {}).get(lr, {})
            true_path = trip.get("true")
            pred_path = trip.get("pred")
            err_path = trip.get("abs_error")

            if true_cache is None and true_path is not None:
                true_arr = load_numeric_txt(true_path)
                true_cache = true_arr
            else:
                true_arr = true_cache

            pred_arr = load_numeric_txt(pred_path) if pred_path is not None else None
            err_arr = None

            if true_arr is None and pred_arr is None:
                continue

            dim = 0
            if true_arr is not None:
                dim = 2 if (np.asarray(true_arr).ndim > 1 and np.asarray(true_arr).shape[-1] >= 3) else 1
            elif pred_arr is not None:
                dim = 2 if (np.asarray(pred_arr).ndim > 1 and np.asarray(pred_arr).shape[-1] >= 3) else 1

            if true_arr is not None and pred_arr is not None:
                true_norm = normalize_points(true_arr)
                pred_norm = normalize_points(pred_arr)
                if true_norm[0] == "1d":
                    x, yt, yp = align_1d((true_norm[1], true_norm[2]), (pred_norm[1], pred_norm[2]))
                    err_arr = np.column_stack([x, np.abs(yp - yt)])
                    true_plot = np.column_stack([x, yt])
                    pred_plot = np.column_stack([x, yp])
                    dim = 1
                else:
                    x, y, zt, zp = align_2d((true_norm[1], true_norm[2], true_norm[3]), (pred_norm[1], pred_norm[2], pred_norm[3]))
                    err_arr = np.column_stack([x, y, np.abs(zp - zt)])
                    true_plot = np.column_stack([x, y, zt])
                    pred_plot = np.column_stack([x, y, zp])
                    dim = 2
            else:
                true_plot = true_arr
                pred_plot = pred_arr
                if err_path is not None:
                    err_arr = load_numeric_txt(err_path)
                dim = 2 if ((true_arr is not None and np.asarray(true_arr).ndim > 1 and np.asarray(true_arr).shape[-1] >= 3) or (pred_arr is not None and np.asarray(pred_arr).ndim > 1 and np.asarray(pred_arr).shape[-1] >= 3)) else 1

            loaded["fields"][field][lr] = {"true": true_plot, "pred": pred_plot, "err": err_arr}
            loaded["dims"][field] = dim
    return loaded


def nice_lr(lr: str) -> str:
    return f"lr={lr}"


def choose_err_norm(arrays):
    vals = []
    for a in arrays:
        if a is None:
            continue
        aa = np.asarray(a, dtype=float)
        if aa.ndim == 1:
            v = aa
        elif aa.shape[1] == 2:
            v = aa[:, 1]
        else:
            v = aa[:, 2]
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)
    if not vals:
        return Normalize(vmin=0.0, vmax=1.0), 0.0, 1.0, "linear"
    v = np.concatenate(vals)
    vmax = float(np.nanmax(v))
    if vmax <= 0:
        return Normalize(vmin=0.0, vmax=1.0), 0.0, 1.0, "linear"
    positive = v[v > 0]
    if positive.size and vmax / max(float(np.nanmin(positive)), 1e-30) > 1e3:
        vmin = max(float(np.nanpercentile(positive, 5)), 1e-16)
        return LogNorm(vmin=vmin, vmax=vmax), vmin, vmax, "log"
    return Normalize(vmin=0.0, vmax=vmax), 0.0, vmax, "linear"


def copy_with_new_z(arr, new_z):
    aa = np.asarray(arr, dtype=float)
    if aa.ndim == 1:
        return np.asarray(new_z, dtype=float)
    out = aa.copy()
    if out.shape[1] == 2:
        out[:, 1] = np.asarray(new_z, dtype=float)
    else:
        out[:, 2] = np.asarray(new_z, dtype=float)
    return out


def field_values(arr):
    aa = np.asarray(arr, dtype=float)
    if aa.ndim == 1:
        return aa
    if aa.shape[1] == 2:
        return aa[:, 1]
    return aa[:, 2]


def maybe_center_prediction_displays(case_name, field, base_true, pred_arrays, dim):
    if dim != 2:
        return base_true, pred_arrays, False
    key = (case_name + ' ' + field).lower()
    arrays = [a for a in [base_true] + [a for a in pred_arrays if a is not None] if a is not None]
    if not arrays:
        return base_true, pred_arrays, False
    vals = np.concatenate([field_values(a)[np.isfinite(field_values(a))] for a in arrays if field_values(a).size])
    if vals.size == 0:
        return base_true, pred_arrays, False
    median = float(np.nanmedian(vals))
    span = float(np.nanmax(vals) - np.nanmin(vals))
    rel = span / max(abs(median), 1.0)
    heat_like = ('硅板加热' in case_name) or ('heat' in key and field.lower() in ['t', 'temperature']) or ('temperature' in key)
    if (not heat_like) and rel >= 5e-3:
        return base_true, pred_arrays, False
    true2 = None
    if base_true is not None:
        true_vals = field_values(base_true)
        true2 = copy_with_new_z(base_true, true_vals - float(np.nanmedian(true_vals)))
    pred2 = []
    for a in pred_arrays:
        if a is None:
            pred2.append(None)
            continue
        vals_a = field_values(a)
        pred2.append(copy_with_new_z(a, vals_a - float(np.nanmedian(vals_a))))
    return true2, pred2, True


def prediction_display_range(case_name, field, base_true, display_true, display_preds, used_centering):
    if display_true is None and not any(a is not None for a in display_preds):
        return 0.0, 1.0
    key = (case_name + ' ' + field).lower()
    heat_like = ('硅板加热' in case_name) or ('heat' in key and field.lower() in ['t', 'temperature']) or ('temperature' in key)
    if (not used_centering) or (not heat_like) or (display_true is None):
        return data_range_for_prediction([display_true] + [a for a in display_preds if a is not None])
    tv = field_values(display_true)
    tv = tv[np.isfinite(tv)]
    if tv.size == 0:
        return data_range_for_prediction([display_true] + [a for a in display_preds if a is not None])
    absq = float(np.nanpercentile(np.abs(tv), 99.5))
    if not np.isfinite(absq) or absq <= 0:
        absq = float(np.nanmax(np.abs(tv))) if tv.size else 1.0
    if not np.isfinite(absq) or absq <= 0:
        absq = 1.0
    return -1.12 * absq, 1.12 * absq


def data_range_for_prediction(arrays):
    vals = []
    for a in arrays:
        if a is None:
            continue
        aa = np.asarray(a, dtype=float)
        if aa.ndim == 1:
            v = aa
        elif aa.shape[1] == 2:
            v = aa[:, 1]
        else:
            v = aa[:, 2]
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(v)
    if not vals:
        return 0.0, 1.0
    v = np.concatenate(vals)
    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))
    if vmin == vmax:
        eps = 1.0 if vmin == 0 else abs(vmin) * 1e-6
        return vmin - eps, vmax + eps
    return vmin, vmax


def plot_1d(ax, arr, title, color=None, is_error=False):
    aa = np.asarray(arr, dtype=float)
    if aa.ndim == 1:
        x = np.arange(aa.size, dtype=float)
        y = aa
    else:
        x = aa[:, 0]
        y = aa[:, 1]
    ax.plot(x, y, lw=1.2, color=color if color is not None else None)
    ax.set_title(title, fontsize=8, pad=1.5)
    ax.tick_params(labelsize=6, length=2, pad=1)
    if is_error:
        ax.yaxis.set_major_locator(MaxNLocator(3))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))


def plot_2d(ax, arr, title, vmin, vmax, cmap=JET, norm=None):
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]
    grid = reconstruct_grid(x, y, z)
    if grid is not None:
        X, Y, Z = grid
        m = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto", vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm)
    else:
        m = ax.tripcolor(x, y, z, cmap=cmap, shading="gouraud", vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm)
    ax.set_title(title, fontsize=8, pad=1.5)
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    return m


def style_colorbar(cb, mode="linear"):
    cb.ax.tick_params(labelsize=3.6, length=1.2, pad=1.3, labelright=True, labelleft=False, right=True, left=False)
    cb.outline.set_linewidth(0.5)
    cb.ax.yaxis.set_ticks_position('right')
    cb.ax.yaxis.set_label_position('right')
    if mode == "linear":
        cb.ax.yaxis.set_major_locator(MaxNLocator(2))
        cb.formatter = FuncFormatter(lambda x, pos: f"{x:.2g}")
        cb.update_ticks()
    else:
        cb.formatter = LogFormatterSciNotation(base=10, labelOnlyBase=False)
        cb.ax.minorticks_off()
        cb.update_ticks()
    try:
        cb.ax.yaxis.get_offset_text().set_visible(False)
    except Exception:
        pass
    return cb


def add_colorbar_in_cax(fig, cax, mappable, mode="linear"):
    cb = fig.colorbar(mappable, cax=cax)
    return style_colorbar(cb, mode=mode)


def plot_case_summary(bundle, out_dir: Path, case_index: int):
    lrs = bundle["lr_values"]
    fields = [f for f in bundle["fields"].keys() if bundle["fields"][f]]
    if not fields:
        return None

    n_lr = len(lrs)
    ncols = 1 + n_lr + 1 + n_lr + 1
    nrows = 1 + len(fields)
    width_ratios = [1.0] * (1 + n_lr) + [0.078] + [1.0] * n_lr + [0.078]
    fig = plt.figure(figsize=(max(18, 2.22 * ncols), 2.8 + 2.15 * len(fields)))
    gs = fig.add_gridspec(
        nrows,
        ncols,
        width_ratios=width_ratios,
        height_ratios=[1.0] + [1.0] * len(fields),
        hspace=0.34,
        wspace=0.110,
    )

    loss_ax = fig.add_subplot(gs[0, :])
    plotted_loss = False
    for lr in lrs:
        loss = bundle["losses"].get(lr)
        if loss is None:
            continue
        x, y = loss
        loss_ax.plot(x, y, lw=1.0, label=nice_lr(lr))
        plotted_loss = True
    if plotted_loss:
        loss_ax.set_yscale("log")
        loss_ax.legend(fontsize=7, ncol=min(6, max(1, len(lrs))), frameon=False, loc="upper right")
    loss_ax.set_title(f"Case {case_index}: {bundle['case_name']}", fontsize=10, pad=3)
    loss_ax.set_ylabel("Loss", fontsize=8)
    loss_ax.tick_params(labelsize=7, length=2, pad=1)
    loss_ax.grid(alpha=0.18, lw=0.4)

    pred_cbar_col = 1 + n_lr
    err_start_col = pred_cbar_col + 1
    err_cbar_col = err_start_col + n_lr

    for i, field in enumerate(fields, start=1):
        dim = bundle["dims"].get(field, 2)
        row = bundle["fields"][field]
        base_true = None
        pred_arrays = []
        err_arrays = []
        for lr in lrs:
            entry = row.get(lr)
            if entry is None:
                pred_arrays.append(None)
                err_arrays.append(None)
                continue
            if base_true is None and entry.get("true") is not None:
                base_true = entry["true"]
            pred_arrays.append(entry.get("pred"))
            err_arrays.append(entry.get("err"))

        display_true, display_preds, used_centering = maybe_center_prediction_displays(bundle['case_name'], field, base_true, pred_arrays, dim)
        pred_vmin, pred_vmax = prediction_display_range(bundle['case_name'], field, base_true, display_true, display_preds, used_centering)
        err_norm, _, _, err_mode = choose_err_norm([a for a in err_arrays if a is not None])

        pred_last = None
        err_last = None

        ax0 = fig.add_subplot(gs[i, 0])
        if dim == 1:
            if base_true is not None:
                plot_1d(ax0, base_true, "Exact")
        else:
            if display_true is not None:
                pred_last = plot_2d(ax0, display_true, "Exact", pred_vmin, pred_vmax, cmap=JET)
        ax0.set_ylabel(field, fontsize=8)

        for j, lr in enumerate(lrs, start=1):
            ax = fig.add_subplot(gs[i, j])
            arr = display_preds[j - 1]
            if arr is not None:
                if dim == 1:
                    plot_1d(ax, arr, nice_lr(lr))
                else:
                    pred_last = plot_2d(ax, arr, nice_lr(lr), pred_vmin, pred_vmax, cmap=JET)
            else:
                ax.set_axis_off()

        pred_cax = fig.add_subplot(gs[i, pred_cbar_col])
        if dim == 2 and pred_last is not None:
            add_colorbar_in_cax(fig, pred_cax, pred_last, mode="linear")
        else:
            pred_cax.set_axis_off()

        for j, lr in enumerate(lrs):
            ax = fig.add_subplot(gs[i, err_start_col + j])
            arr = row.get(lr, {}).get("err")
            if arr is not None:
                if dim == 1:
                    plot_1d(ax, arr, nice_lr(lr), is_error=True)
                else:
                    err_last = plot_2d(ax, arr, nice_lr(lr), 0.0, 1.0, cmap=JET, norm=err_norm)
            else:
                ax.set_axis_off()

        err_cax = fig.add_subplot(gs[i, err_cbar_col])
        if dim == 2 and err_last is not None:
            add_colorbar_in_cax(fig, err_cax, err_last, mode=err_mode)
        else:
            err_cax.set_axis_off()

    fig.subplots_adjust(left=0.038, right=0.947, top=0.97, bottom=0.035)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = sanitize_name(bundle["case_name"]) + "_lr_sweep_summary.png"
    out_path = out_dir / name
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    return out_path

def stack_images_vertically(image_paths, out_path: Path):
    if not image_paths:
        return None
    if Image is None:
        return None
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    w = max(im.width for im in imgs)
    gap = 16
    total_h = sum(im.height for im in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, total_h), (255, 255, 255))
    y = 0
    for im in imgs:
        x = (w - im.width) // 2
        canvas.paste(im, (x, y))
        y += im.height + gap
    canvas.save(out_path)
    return out_path


def main():
    configure_fonts()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    root = resolve_root(args.root)
    out_dir = Path(args.out).expanduser().resolve() if args.out else (Path(__file__).resolve().parent / "lr_sweep_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = [d for d in root.iterdir() if d.is_dir()]
    case_dirs = sorted(case_dirs, key=lambda p: p.name)

    pngs = []
    idx = 1
    for case_dir in case_dirs:
        bundle0 = discover_bundle(case_dir)
        if bundle0 is None:
            continue
        bundle = load_case_data(bundle0)
        png = plot_case_summary(bundle, out_dir, idx)
        if png is not None:
            pngs.append(png)
            idx += 1

    if pngs:
        stack_images_vertically(pngs, out_dir / "all_cases_lr_sweep_stacked_vertical.png")


if __name__ == "__main__":
    main()
