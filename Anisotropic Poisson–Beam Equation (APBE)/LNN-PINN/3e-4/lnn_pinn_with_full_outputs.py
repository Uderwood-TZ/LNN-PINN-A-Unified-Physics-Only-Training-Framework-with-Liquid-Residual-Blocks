import os
import time
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


# ==========================================================
# Hyperparameters (kept consistent with the uploaded script)
# ==========================================================
epochs = 5000
LR = 3e-4
N = 1000          # interior collocation points
N1 = 1000         # boundary collocation points per boundary condition
h = 200           # visualization grid density
FIELD_NAME = "uy"
METHOD_TAG = "LNN-PINN"
SEED = 888888
PRINT_EVERY = 100
DPI = 300

# ==========================================================
# Paths
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(SCRIPT_DIR, f"outputs_{METHOD_TAG.replace('-', '_')}")
FIG_DIR = os.path.join(OUT_ROOT, "figures")
DATA_DIR = os.path.join(OUT_ROOT, "data")
LOG_DIR = os.path.join(OUT_ROOT, "logs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================================
# Reproducibility and device
# ==========================================================
def setup_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device():
    if os.environ.get("PINN_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


setup_seed(SEED)
device = select_device()


def cuda_warmup():
    global device
    if device.type != "cuda":
        return
    try:
        torch.cuda.set_device(0)
        torch.cuda.init()
        a = torch.randn(64, 64, device=device, requires_grad=True)
        b = torch.randn(64, 64, device=device, requires_grad=True)
        (a @ b).sum().backward()
        torch.cuda.synchronize()
    except Exception as e:
        print("[CUDA warmup warning]", e, "-> fallback to CPU")
        device = torch.device("cpu")


cuda_warmup()

# ==========================================================
# Sampling
# ==========================================================
def interior(n: int = N):
    x = torch.rand(n, 1, device=device)
    y = torch.rand(n, 1, device=device)
    cond = (2.0 - x ** 2) * torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def down_yy(n: int = N1):
    x = torch.rand(n, 1, device=device)
    y = torch.zeros_like(x)
    cond = x ** 2
    return x.requires_grad_(True), y.requires_grad_(True), cond


def up_yy(n: int = N1):
    x = torch.rand(n, 1, device=device)
    y = torch.ones_like(x)
    cond = x ** 2 / math.e
    return x.requires_grad_(True), y.requires_grad_(True), cond


def down(n: int = N1):
    x = torch.rand(n, 1, device=device)
    y = torch.zeros_like(x)
    cond = x ** 2
    return x.requires_grad_(True), y.requires_grad_(True), cond


def up(n: int = N1):
    x = torch.rand(n, 1, device=device)
    y = torch.ones_like(x)
    cond = x ** 2 / math.e
    return x.requires_grad_(True), y.requires_grad_(True), cond


def left(n: int = N1):
    y = torch.rand(n, 1, device=device)
    x = torch.zeros_like(y)
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def right(n: int = N1):
    y = torch.rand(n, 1, device=device)
    x = torch.ones_like(y)
    cond = torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond

# ==========================================================
# Autograd helpers
# ==========================================================
MSELoss = torch.nn.MSELoss()


def gradients(u, x, order: int = 1):
    if order == 1:
        return torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
    return gradients(gradients(u, x, 1), x, order - 1)

# ==========================================================
# LNN operator and network
# ==========================================================
class LiquidNeuron(torch.nn.Module):
    """
    Width-preserving residual liquid operator:
        h_out = h + alpha * tanh(W h + b)
    This preserves the hidden dimension 64, so the linear topology remains unchanged.
    """
    def __init__(self, size: int, alpha_init: float = 0.5):
        super().__init__()
        self.W = torch.nn.Linear(size, size)
        self.alpha = torch.nn.Parameter(torch.ones(size) * alpha_init)

    def forward(self, h):
        return h + self.alpha * torch.tanh(self.W(h))


class MLP_LNN(torch.nn.Module):
    """
    Original linear topology:
        2 -> 64 -> 64 -> 64 -> 64 -> 1
    We insert one liquid operator after each hidden tanh without changing widths.
    """
    def __init__(self):
        super().__init__()
        self.inp = torch.nn.Linear(2, 64)
        self.fc1 = torch.nn.Linear(64, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.out = torch.nn.Linear(64, 1)

        self.ln1 = LiquidNeuron(64)
        self.ln2 = LiquidNeuron(64)
        self.ln3 = LiquidNeuron(64)
        self.ln4 = LiquidNeuron(64)

    def forward(self, x):
        x = torch.tanh(self.inp(x))
        x = self.ln1(x)
        x = torch.tanh(self.fc1(x))
        x = self.ln2(x)
        x = torch.tanh(self.fc2(x))
        x = self.ln3(x)
        x = torch.tanh(self.fc3(x))
        x = self.ln4(x)
        x = self.out(x)
        return x

# ==========================================================
# Physics losses
# ==========================================================
@torch.enable_grad()
def L_interior(model):
    x, y, cond = interior()
    uxy = model(torch.cat([x, y], dim=1))
    pde_res = gradients(uxy, x, 2) - gradients(uxy, y, 4)
    return MSELoss(pde_res, cond)


@torch.enable_grad()
def L_down_yy(model):
    x, y, cond = down_yy()
    uxy = model(torch.cat([x, y], dim=1))
    return MSELoss(gradients(uxy, y, 2), cond)


@torch.enable_grad()
def L_up_yy(model):
    x, y, cond = up_yy()
    uxy = model(torch.cat([x, y], dim=1))
    return MSELoss(gradients(uxy, y, 2), cond)


@torch.enable_grad()
def L_down(model):
    x, y, cond = down()
    uxy = model(torch.cat([x, y], dim=1))
    return MSELoss(uxy, cond)


@torch.enable_grad()
def L_up(model):
    x, y, cond = up()
    uxy = model(torch.cat([x, y], dim=1))
    return MSELoss(uxy, cond)


@torch.enable_grad()
def L_left(model):
    x, y, cond = left()
    uxy = model(torch.cat([x, y], dim=1))
    return MSELoss(uxy, cond)


@torch.enable_grad()
def L_right(model):
    x, y, cond = right()
    uxy = model(torch.cat([x, y], dim=1))
    return MSELoss(uxy, cond)

# ==========================================================
# Exact solution and export helpers
# ==========================================================
def analytic_u(x, y):
    return (x ** 2) * torch.exp(-y)


def save_xyz_txt(x_grid, y_grid, value_grid, txt_path):
    arr = np.column_stack([
        x_grid.reshape(-1),
        y_grid.reshape(-1),
        value_grid.reshape(-1),
    ])
    header = "x y value"
    np.savetxt(txt_path, arr, fmt="%.12e", header=header, comments="")


def save_field_figure(x_grid, y_grid, value_grid, png_path, title, cbar_label, vmin=None, vmax=None):
    plt.figure(figsize=(6.5, 5.2))
    plt.imshow(
        value_grid.T,
        origin="lower",
        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        aspect="auto",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label=cbar_label)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def save_loss_curve(epochs_arr, loss_arr, png_path, title, yscale="linear"):
    plt.figure(figsize=(7.0, 5.0))
    plt.plot(epochs_arr, loss_arr, linewidth=2.0, color=plt.cm.jet(0.20))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    if yscale == "log":
        plt.yscale("log")
        plt.grid(True, which="both", alpha=0.3)
    else:
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def write_run_summary(summary_path, training_time_seconds, metrics_dict):
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("LNN-PINN run summary\n")
        f.write("===================\n")
        f.write(f"method_tag: {METHOD_TAG}\n")
        f.write("problem: u_xx - u_yyyy = (2 - x^2) exp(-y) on (0,1)^2\n")
        f.write("exact_solution: u(x,y) = x^2 exp(-y)\n")
        f.write("network_linear_topology: 2 -> 64 -> 64 -> 64 -> 64 -> 1\n")
        f.write("liquid_operator: residual liquid cell after each hidden tanh, width preserved\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"learning_rate: {LR:.12e}\n")
        f.write(f"interior_points: {N}\n")
        f.write(f"boundary_points_each: {N1}\n")
        f.write(f"grid_density: {h}\n")
        f.write(f"seed: {SEED}\n")
        f.write(f"device: {device}\n")
        f.write(f"training_time_seconds: {training_time_seconds:.6f}\n")
        f.write("\nmetrics\n")
        for k, v in metrics_dict.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.12e}\n")
            else:
                f.write(f"{k}: {v}\n")

# ==========================================================
# Evaluation
# ==========================================================
@torch.no_grad()
def evaluate_and_save_results(model, training_time_seconds: float, lr_value: float, grid_n: int = 200):
    model.eval()
    lr_str = f"{lr_value:.0e}"

    xs = torch.linspace(0.0, 1.0, grid_n, device=device)
    ys = torch.linspace(0.0, 1.0, grid_n, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)

    pred = model(XY).reshape(grid_n, grid_n)
    true = analytic_u(Xg, Yg)
    abs_err = (pred - true).abs()

    x_np = Xg.detach().cpu().numpy()
    y_np = Yg.detach().cpu().numpy()
    true_np = true.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    abs_np = abs_err.detach().cpu().numpy()
    diff_np = pred_np - true_np

    mse = float(np.mean(diff_np ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff_np)))
    l2_abs = float(np.sqrt(np.sum(diff_np ** 2)))
    l2_rel = float(np.sqrt(np.sum(diff_np ** 2) / np.sum(true_np ** 2)))
    max_abs = float(np.max(np.abs(diff_np)))

    vmin = float(min(true_np.min(), pred_np.min()))
    vmax = float(max(true_np.max(), pred_np.max()))

    true_png = os.path.join(FIG_DIR, f"{FIELD_NAME}_true_{METHOD_TAG}_lr{lr_str}.png")
    pred_png = os.path.join(FIG_DIR, f"{FIELD_NAME}_pred_{METHOD_TAG}_lr{lr_str}.png")
    err_png = os.path.join(FIG_DIR, f"{FIELD_NAME}_maxerror_{METHOD_TAG}_lr{lr_str}.png")

    true_txt = os.path.join(DATA_DIR, f"{FIELD_NAME}_true_{METHOD_TAG}_lr{lr_str}.txt")
    pred_txt = os.path.join(DATA_DIR, f"{FIELD_NAME}_pred_{METHOD_TAG}_lr{lr_str}.txt")
    err_txt = os.path.join(DATA_DIR, f"{FIELD_NAME}_maxerror_{METHOD_TAG}_lr{lr_str}.txt")
    metrics_txt = os.path.join(LOG_DIR, f"{FIELD_NAME}_metrics_{METHOD_TAG}_lr{lr_str}.txt")
    summary_txt = os.path.join(LOG_DIR, f"run_summary_{METHOD_TAG}_lr{lr_str}.txt")

    save_xyz_txt(x_np, y_np, true_np, true_txt)
    save_xyz_txt(x_np, y_np, pred_np, pred_txt)
    save_xyz_txt(x_np, y_np, abs_np, err_txt)

    save_field_figure(
        x_np,
        y_np,
        true_np,
        true_png,
        title=f"True solution of {FIELD_NAME}",
        cbar_label=f"{FIELD_NAME}_true",
        vmin=vmin,
        vmax=vmax,
    )
    save_field_figure(
        x_np,
        y_np,
        pred_np,
        pred_png,
        title=f"Predicted solution of {FIELD_NAME}",
        cbar_label=f"{FIELD_NAME}_pred",
        vmin=vmin,
        vmax=vmax,
    )
    save_field_figure(
        x_np,
        y_np,
        abs_np,
        err_png,
        title=f"Maxerror / absolute error of {FIELD_NAME}",
        cbar_label=f"|{FIELD_NAME}_pred - {FIELD_NAME}_true|",
    )

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "L2_absolute_error": l2_abs,
        "L2_relative_error": l2_rel,
        "Max_absolute_error": max_abs,
        "training_time_seconds": float(training_time_seconds),
    }

    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"field_name: {FIELD_NAME}\n")
        f.write(f"method_tag: {METHOD_TAG}\n")
        f.write(f"learning_rate: {lr_value:.12e}\n")
        f.write(f"MSE: {mse:.12e}\n")
        f.write(f"RMSE: {rmse:.12e}\n")
        f.write(f"MAE: {mae:.12e}\n")
        f.write(f"L2_absolute_error: {l2_abs:.12e}\n")
        f.write(f"L2_relative_error: {l2_rel:.12e}\n")
        f.write(f"Max_absolute_error: {max_abs:.12e}\n")
        f.write(f"training_time_seconds: {training_time_seconds:.12e}\n")

    write_run_summary(summary_txt, training_time_seconds, metrics)

    return {
        "true_png": true_png,
        "pred_png": pred_png,
        "err_png": err_png,
        "true_txt": true_txt,
        "pred_txt": pred_txt,
        "err_txt": err_txt,
        "metrics_txt": metrics_txt,
        "summary_txt": summary_txt,
        "metrics": metrics,
    }

# ==========================================================
# Training
# ==========================================================
def train():
    print("Using device:", device)
    model = MLP_LNN().to(device)
    cuda_warmup()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_history = []
    component_history = []

    time_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        loss_interior = L_interior(model)
        loss_down_yy = L_down_yy(model)
        loss_up_yy = L_up_yy(model)
        loss_down_bc = L_down(model)
        loss_up_bc = L_up(model)
        loss_left_bc = L_left(model)
        loss_right_bc = L_right(model)

        total_loss = (
            loss_interior
            + loss_down_yy
            + loss_up_yy
            + loss_down_bc
            + loss_up_bc
            + loss_left_bc
            + loss_right_bc
        )

        total_loss.backward()
        optimizer.step()

        total_scalar = float(total_loss.detach().cpu().item())
        component_record = {
            "epoch": epoch,
            "total_loss": total_scalar,
            "L_interior": float(loss_interior.detach().cpu().item()),
            "L_down_yy": float(loss_down_yy.detach().cpu().item()),
            "L_up_yy": float(loss_up_yy.detach().cpu().item()),
            "L_down": float(loss_down_bc.detach().cpu().item()),
            "L_up": float(loss_up_bc.detach().cpu().item()),
            "L_left": float(loss_left_bc.detach().cpu().item()),
            "L_right": float(loss_right_bc.detach().cpu().item()),
        }

        loss_history.append(total_scalar)
        component_history.append(component_record)

        if epoch % PRINT_EVERY == 0 or epoch == 1 or epoch == epochs:
            print(
                f"epoch {epoch:5d} | total={component_record['total_loss']:.6e} | "
                f"int={component_record['L_interior']:.6e} | "
                f"downyy={component_record['L_down_yy']:.6e} | "
                f"upyy={component_record['L_up_yy']:.6e}"
            )

    training_time_seconds = time.perf_counter() - time_start

    lr_str = f"{LR:.0e}"
    epochs_arr = np.arange(1, len(loss_history) + 1)
    loss_array = np.asarray(loss_history, dtype=np.float64)

    # Save full loss history step by step
    loss_txt = os.path.join(LOG_DIR, f"loss_history_{METHOD_TAG}_lr{lr_str}.txt")
    header = "epoch total_loss L_interior L_down_yy L_up_yy L_down L_up L_left L_right"
    data_mat = np.array([
        [
            rec["epoch"],
            rec["total_loss"],
            rec["L_interior"],
            rec["L_down_yy"],
            rec["L_up_yy"],
            rec["L_down"],
            rec["L_up"],
            rec["L_left"],
            rec["L_right"],
        ]
        for rec in component_history
    ], dtype=np.float64)
    np.savetxt(loss_txt, data_mat, fmt="%.12e", header=header, comments="")

    # Save linear and logarithmic loss curves
    linear_loss_png = os.path.join(FIG_DIR, f"loss_curve_linear_{METHOD_TAG}_lr{lr_str}.png")
    log_loss_png = os.path.join(FIG_DIR, f"loss_curve_log_{METHOD_TAG}_lr{lr_str}.png")
    save_loss_curve(epochs_arr, loss_array, linear_loss_png, "Training loss curve (linear scale)", yscale="linear")
    positive_loss = np.clip(loss_array, 1e-300, None)
    save_loss_curve(epochs_arr, positive_loss, log_loss_png, "Training loss curve (log scale)", yscale="log")

    results = evaluate_and_save_results(
        model=model,
        training_time_seconds=training_time_seconds,
        lr_value=LR,
        grid_n=max(150, h),
    )

    print("Training finished.")
    print(f"Training time: {training_time_seconds:.6f} s")
    print(f"Loss history txt: {loss_txt}")
    print(f"Metrics txt: {results['metrics_txt']}")
    print(f"True figure: {results['true_png']}")
    print(f"Prediction figure: {results['pred_png']}")
    print(f"Maxerror figure: {results['err_png']}")

    return {
        "model": model,
        "loss_history": loss_array,
        "loss_history_txt": loss_txt,
        "linear_loss_png": linear_loss_png,
        "log_loss_png": log_loss_png,
        "results": results,
    }


if __name__ == "__main__":
    train()
