import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time  # 新增：用于计时

# =====================
# Hyperparameters
# =====================
epochs = 5000
LR = 3e-4
N = 1000     # interior points
N1 = 1000    # boundary points
h = 100      # (可选)可视化网格密度，当前脚本不绘制解，只绘制loss
FIELD_NAME = "uy"  # 文件命名前缀，与你的示例 uy_result_PINN_lr1e-03_error.txt 保持一致

# =====================
# Utils
# =====================
def setup_seed(seed: int = 888888):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(888888)

def select_device():
    # 环境变量 PINN_FORCE_CPU=1 可强制走 CPU
    if os.environ.get("PINN_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = select_device()

# 更稳健的 CUDA 预热：做一次小矩阵乘 + backward，提前创建 cuBLAS/cudnn 句柄
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
        print("[CUDA warmup warning]", e, "→ fallback to CPU")
        device = torch.device("cpu")

cuda_warmup()

def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def _write_xyz(filename: str, X: np.ndarray, Y: np.ndarray, V: np.ndarray):
    """
    保存 3 列格式的数据：x y value，第一行为表头。
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("x y value\n")
        H, W = V.shape
        for i in range(H):
            for j in range(W):
                x_ = float(X[i, j])
                y_ = float(Y[i, j])
                v_ = float(V[i, j])
                f.write(f"{x_:.8e} {y_:.8e} {v_:.8e}\n")

# =====================
# Sampling on domain/boundaries
# =====================
# 域: (x,y) ∈ (0,1)×(0,1)
# PDE: u_xx - u_yyyy = (2 - x^2) * exp(-y)

def interior(n: int = N):
    x = torch.rand(n, 1, device=device)
    y = torch.rand(n, 1, device=device)
    cond = (2 - x**2) * torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond

# y=0: u = x^2, u_yy = x^2
# y=1: u = x^2/e, u_yy = x^2/e

def down_yy(n: int = N1):
    x = torch.rand(n, 1, device=device)
    y = torch.zeros_like(x)
    cond = x**2
    return x.requires_grad_(True), y.requires_grad_(True), cond

def up_yy(n: int = N1):
    x = torch.rand(n, 1, device=device)
    y = torch.ones_like(x)
    cond = x**2 / torch.e
    return x.requires_grad_(True), y.requires_grad_(True), cond

def down(n: int = N1):
    x = torch.rand(n, 1, device=device)
    y = torch.zeros_like(x)
    cond = x**2
    return x.requires_grad_(True), y.requires_grad_(True), cond

def up(n: int = N1):
    x = torch.rand(n, 1, device=device)
    y = torch.ones_like(x)
    cond = x**2 / torch.e
    return x.requires_grad_(True), y.requires_grad_(True), cond

# x=0: u=0; x=1: u=exp(-y)

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

# =====================
# Model
# =====================
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# =====================
# Autograd helpers
# =====================
MSE = torch.nn.MSELoss()

def gradients(u, x, order: int = 1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True, only_inputs=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

# =====================
# Loss terms (无数据损失)
# =====================
@torch.enable_grad()
def L_interior(model):
    x, y, cond = interior()
    uxy = model(torch.cat([x, y], dim=1))
    return MSE(gradients(uxy, x, 2) - gradients(uxy, y, 4), cond)

@torch.enable_grad()
def L_down_yy(model):
    x, y, cond = down_yy()
    uxy = model(torch.cat([x, y], dim=1))
    return MSE(gradients(uxy, y, 2), cond)

@torch.enable_grad()
def L_up_yy(model):
    x, y, cond = up_yy()
    uxy = model(torch.cat([x, y], dim=1))
    return MSE(gradients(uxy, y, 2), cond)

@torch.enable_grad()
def L_down(model):
    x, y, cond = down()
    uxy = model(torch.cat([x, y], dim=1))
    return MSE(uxy, cond)

@torch.enable_grad()
def L_up(model):
    x, y, cond = up()
    uxy = model(torch.cat([x, y], dim=1))
    return MSE(uxy, cond)

@torch.enable_grad()
def L_left(model):
    x, y, cond = left()
    uxy = model(torch.cat([x, y], dim=1))
    return MSE(uxy, cond)

@torch.enable_grad()
def L_right(model):
    x, y, cond = right()
    uxy = model(torch.cat([x, y], dim=1))
    return MSE(uxy, cond)

# =====================
# Evaluate & Save Results
# =====================

def analytic_u(x, y):
    return (x ** 2) * torch.exp(-y)

@torch.no_grad()
def evaluate_and_save_results(model, lr_value: float, grid_n: int = 200):
    model.eval()
    lr_str = f"{lr_value:.0e}"  # e.g., 1e-03

    # ==== grid ====
    xs = torch.linspace(0, 1, grid_n, device=device)
    ys = torch.linspace(0, 1, grid_n, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)

    # PINN prediction
    Up = model(XY).reshape(grid_n, grid_n)
    # Analytic
    Ut = analytic_u(Xg, Yg)

    # Error
    AbsErr = (Up - Ut).abs()

    # ==== metrics ====
    err = (Up - Ut).detach().cpu().numpy()
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    maxae = float(np.max(np.abs(err)))
    max_idx = np.unravel_index(int(np.argmax(np.abs(err))), err.shape)
    max_i, max_j = int(max_idx[0]), int(max_idx[1])

    # ==== save metrics txt（原有） ====
    txt_name = f"{FIELD_NAME}_result_PINN_lr{lr_str}_error.txt"
    with open(txt_name, "w", encoding="utf-8") as f:
        f.write("u 分量误差分析\n")
        f.write(f"均方误差 (MSE): {mse:.6e}\n")
        f.write(f"均方根误差 (RMSE): {rmse:.6f}\n")
        f.write(f"平均绝对误差 (MAE): {mae:.6f}\n")
        f.write(f"最大绝对误差: {maxae:.6f}\n")

    # ==== plots（原有） ====
    Ut_np = Ut.detach().cpu().numpy()
    Up_np = Up.detach().cpu().numpy()
    Abs_np = AbsErr.detach().cpu().numpy()
    vmin = float(min(Ut_np.min(), Up_np.min()))
    vmax = float(max(Ut_np.max(), Up_np.max()))

    # 解析解
    plt.figure(figsize=(6, 5))
    plt.imshow(Ut_np.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
               cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(label=f"{FIELD_NAME}_true")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Analytic solution u(x,y)")
    plt.savefig(f"{FIELD_NAME}_true_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # PINN 预测
    plt.figure(figsize=(6, 5))
    plt.imshow(Up_np.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
               cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(label=f"{FIELD_NAME}_PINN")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("PINN prediction u_PINN(x,y)")
    plt.savefig(f"{FIELD_NAME}_pinn_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 绝对误差
    plt.figure(figsize=(6, 5))
    plt.imshow(Abs_np.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
               cmap="jet")
    plt.colorbar(label=f"|{FIELD_NAME}_PINN - {FIELD_NAME}_true|")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Absolute error")
    plt.savefig(f"{FIELD_NAME}_abs_error_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==== (3) 结果数据：x y value 三列表 ====
    X_np = Xg.detach().cpu().numpy()
    Y_np = Yg.detach().cpu().numpy()
    _write_xyz(f"{FIELD_NAME}_true_xyz_lr{lr_str}.txt", X_np, Y_np, Ut_np)
    _write_xyz(f"{FIELD_NAME}_pred_xyz_lr{lr_str}.txt", X_np, Y_np, Up_np)
    _write_xyz(f"{FIELD_NAME}_abs_error_xyz_lr{lr_str}.txt", X_np, Y_np, Abs_np)

    # ==== (4) Maxerror：单行三列 x y max_abs_error ====
    max_x = float(X_np[max_i, max_j])
    max_y = float(Y_np[max_i, max_j])
    with open(f"{FIELD_NAME}_maxerror_xyz_lr{lr_str}.txt", "w", encoding="utf-8") as f:
        f.write("x y value\n")
        f.write(f"{max_x:.8e} {max_y:.8e} {maxae:.8e}\n")

    # 回传关键文件名，便于调试
    return {
        "metrics_txt": txt_name,
        "fig_true": f"{FIELD_NAME}_true_lr{lr_str}.png",
        "fig_pinn": f"{FIELD_NAME}_pinn_lr{lr_str}.png",
        "fig_abs": f"{FIELD_NAME}_abs_error_lr{lr_str}.png",
        "xyz_true": f"{FIELD_NAME}_true_xyz_lr{lr_str}.txt",
        "xyz_pred": f"{FIELD_NAME}_pred_xyz_lr{lr_str}.txt",
        "xyz_abs": f"{FIELD_NAME}_abs_error_xyz_lr{lr_str}.txt",
        "xyz_max": f"{FIELD_NAME}_maxerror_xyz_lr{lr_str}.txt",
    }

# =====================
# Train
# =====================

def train():
    print("Using device:", device)
    model = MLP().to(device)
    # 再预热一次，确保当前上下文稳定
    cuda_warmup()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    loss_hist = []
    epoch_durations = []

    # 计算成本：训练起始时间、GPU 信息
    train_t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "N/A"

    param_count = _count_parameters(model)

    for i in range(1, epochs + 1):
        ep_t0 = time.perf_counter()

        opt.zero_grad()

        L1 = L_interior(model)
        L2 = L_down_yy(model)
        L3 = L_up_yy(model)
        L4 = L_down(model)
        L5 = L_up(model)
        L6 = L_left(model)
        L7 = L_right(model)

        l_total = L1 + L2 + L3 + L4 + L5 + L6 + L7
        l_total.backward()
        opt.step()

        loss_val = float(l_total.detach().cpu().item())
        loss_hist.append(loss_val)

        ep_t1 = time.perf_counter()
        epoch_durations.append(ep_t1 - ep_t0)

        if i % 100 == 0 or i == 1:
            print(f"epoch {i:5d} | total_loss = {loss_val:.6e}")

    train_t1 = time.perf_counter()
    total_train_sec = train_t1 - train_t0

    # 保存损失曲线（线性/对数，文件名包含 lr）
    lr_str = f"{LR:.0e}"
    epochs_arr = np.arange(1, len(loss_hist) + 1)

    plt.figure()
    plt.plot(epochs_arr, loss_hist, lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Training Loss (Linear)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"loss_curve_linear_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs_arr, loss_hist, lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Training Loss (Semilogy)")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(f"loss_curve_semilogy_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # (1) 每次训练 LOSS 记录：两列 epoch loss
    loss_txt = f"{FIELD_NAME}_loss_per_epoch_lr{lr_str}.txt"
    with open(loss_txt, "w", encoding="utf-8") as f:
        f.write("epoch loss\n")
        for ep, val in enumerate(loss_hist, start=1):
            f.write(f"{ep:d} {val:.8e}\n")

    # 评估并保存图与误差指标（使用 grid_n=h 或者更细密 200）
    eval_t0 = time.perf_counter()
    eval_files = evaluate_and_save_results(model, lr_value=LR, grid_n=max(150, h))
    eval_t1 = time.perf_counter()
    eval_sec = eval_t1 - eval_t0

    # (2) 计算成本：总训练时间、评估时间、每轮耗时等
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        peak_mem = -1

    cost_txt = f"{FIELD_NAME}_compute_cost_lr{lr_str}.txt"
    with open(cost_txt, "w", encoding="utf-8") as f:
        f.write(f"FieldName: {FIELD_NAME}\n")
        f.write(f"Device: {device.type}\n")
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"ParameterCount: {param_count}\n")
        f.write(f"TotalTrainingSeconds: {total_train_sec:.6f}\n")
        f.write(f"EvalSeconds: {eval_sec:.6f}\n")
        f.write("Epoch Durations (seconds):\n")
        for ep, dt in enumerate(epoch_durations, start=1):
            f.write(f"{ep:d} {dt:.8e}\n")
        f.write(f"MeanEpochSeconds: {np.mean(epoch_durations):.6f}\n")
        f.write(f"StdEpochSeconds:  {np.std(epoch_durations):.6f}\n")
        f.write(f"PeakGPUMemoryBytes: {peak_mem}\n")

    # 可选：返回内容
    return model, np.array(loss_hist), eval_files, loss_txt, cost_txt


if __name__ == "__main__":
    train()
