import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager

epochs = 5000
LR = 3e-4
N = 1000
N1 = 1000
h = 100
FIELD_NAME = "uy"
METHOD_NAME = "ResidualAttentionPINN"
MODEL_WIDTH = 64
NUM_BLOCKS = 4
ATTN_HIDDEN = 8
RES_SCALE = 0.12
ATTN_TEMPERATURE = 0.55
GATE_BIAS = -1.8
DROPOUT_P = 0.18
SEED = 888888

def _set_chinese_font():
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break
    if chosen is None:
        common_paths = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc"]
        for p in common_paths:
            if os.path.exists(p):
                try:
                    font_manager.fontManager.addfont(p)
                    chosen = font_manager.FontProperties(fname=p).get_name()
                    break
                except Exception:
                    pass
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

_set_chinese_font()

def setup_seed(seed: int = SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(SEED)

def select_device():
    if os.environ.get("PINN_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
MSE = torch.nn.MSELoss()

def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def _write_xyz(filename: str, X: np.ndarray, Y: np.ndarray, V: np.ndarray):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("x y value\n")
        H, W = V.shape
        for i in range(H):
            for j in range(W):
                f.write(f"{float(X[i, j]):.8e} {float(Y[i, j]):.8e} {float(V[i, j]):.8e}\n")

def interior(n: int = N):
    x = torch.rand(n, 1, device=device)
    y = torch.rand(n, 1, device=device)
    cond = (2 - x**2) * torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond

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

def analytic_u(x, y):
    return (x ** 2) * torch.exp(-y)

class ResidualAttentionBlock(torch.nn.Module):
    def __init__(self, width, attn_hidden, res_scale, attn_temperature, gate_bias, dropout_p):
        super().__init__()
        self.feature_1 = torch.nn.Linear(width, width)
        self.feature_2 = torch.nn.Linear(width, width)
        self.gate_1 = torch.nn.Linear(width, attn_hidden)
        self.gate_2 = torch.nn.Linear(attn_hidden, width)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.res_scale = res_scale
        self.attn_temperature = attn_temperature
        self.gate_bias = torch.nn.Parameter(torch.full((width,), float(gate_bias)))
    def forward(self, x):
        feat = torch.tanh(self.feature_1(x))
        feat = self.dropout(feat)
        feat = torch.tanh(self.feature_2(feat))
        gate = torch.tanh(self.gate_1(x))
        gate = torch.sigmoid(self.attn_temperature * self.gate_2(gate) + self.gate_bias)
        return x + self.res_scale * gate * feat

class ResidualAttentionPINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = torch.nn.Linear(2, MODEL_WIDTH)
        self.blocks = torch.nn.ModuleList([
            ResidualAttentionBlock(MODEL_WIDTH, ATTN_HIDDEN, RES_SCALE, ATTN_TEMPERATURE, GATE_BIAS, DROPOUT_P)
            for _ in range(NUM_BLOCKS)
        ])
        self.output_layer = torch.nn.Linear(MODEL_WIDTH, 1)
    def forward(self, x):
        z = torch.tanh(self.input_layer(x))
        for block in self.blocks:
            z = block(z)
        return self.output_layer(z)

def gradients(u, x, order: int = 1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, only_inputs=True)[0]
    return gradients(gradients(u, x), x, order=order - 1)

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

@torch.no_grad()
def evaluate_and_save_results(model, lr_value: float, grid_n: int = 200):
    model.eval()
    lr_str = f"{lr_value:.0e}"
    xs = torch.linspace(0, 1, grid_n, device=device)
    ys = torch.linspace(0, 1, grid_n, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
    Up = model(XY).reshape(grid_n, grid_n)
    Ut = analytic_u(Xg, Yg)
    MaxErr = (Up - Ut).abs()
    err = (Up - Ut).detach().cpu().numpy()
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    rel_l2 = float(np.linalg.norm(err.ravel(), 2) / (np.linalg.norm(Ut.detach().cpu().numpy().ravel(), 2) + 1e-12))
    maxae = float(np.max(np.abs(err)))
    Ut_np = Ut.detach().cpu().numpy()
    Up_np = Up.detach().cpu().numpy()
    Max_np = MaxErr.detach().cpu().numpy()
    X_np = Xg.detach().cpu().numpy()
    Y_np = Yg.detach().cpu().numpy()
    metrics_name = f"{FIELD_NAME}_result_{METHOD_NAME}_lr{lr_str}_error.txt"
    with open(metrics_name, "w", encoding="utf-8") as f:
        f.write(f"Method: {METHOD_NAME}\n")
        f.write(f"Field: {FIELD_NAME}\n")
        f.write(f"MSE: {mse:.8e}\n")
        f.write(f"RMSE: {rmse:.8e}\n")
        f.write(f"MAE: {mae:.8e}\n")
        f.write(f"RelativeL2: {rel_l2:.8e}\n")
        f.write(f"MaxAbsoluteError: {maxae:.8e}\n")
    vmin = float(min(Ut_np.min(), Up_np.min()))
    vmax = float(max(Ut_np.max(), Up_np.max()))
    plt.figure(figsize=(6, 5))
    plt.imshow(Ut_np.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"{FIELD_NAME}_true_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(6, 5))
    plt.imshow(Up_np.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"{FIELD_NAME}_{METHOD_NAME.lower()}_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(6, 5))
    plt.imshow(Max_np.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"{FIELD_NAME}_maxerror_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()
    _write_xyz(f"{FIELD_NAME}_true_xyz_lr{lr_str}.txt", X_np, Y_np, Ut_np)
    _write_xyz(f"{FIELD_NAME}_pred_xyz_lr{lr_str}.txt", X_np, Y_np, Up_np)
    _write_xyz(f"{FIELD_NAME}_maxerror_xyz_lr{lr_str}.txt", X_np, Y_np, Max_np)
    return {
        "metrics_txt": metrics_name,
        "fig_true": f"{FIELD_NAME}_true_lr{lr_str}.png",
        "fig_pred": f"{FIELD_NAME}_{METHOD_NAME.lower()}_lr{lr_str}.png",
        "fig_maxerror": f"{FIELD_NAME}_maxerror_lr{lr_str}.png",
        "xyz_true": f"{FIELD_NAME}_true_xyz_lr{lr_str}.txt",
        "xyz_pred": f"{FIELD_NAME}_pred_xyz_lr{lr_str}.txt",
        "xyz_max": f"{FIELD_NAME}_maxerror_xyz_lr{lr_str}.txt",
    }

def save_loss_outputs(loss_log, lr_value: float):
    lr_str = f"{lr_value:.0e}"
    epochs_arr = np.arange(1, loss_log.shape[0] + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_arr, loss_log[:, 1], lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"loss_curve_linear_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_arr, loss_log[:, 1], lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(f"loss_curve_semilogy_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_arr, loss_log[:, 2], lw=1.4, label="Interior")
    plt.plot(epochs_arr, loss_log[:, 3], lw=1.4, label="DownYY")
    plt.plot(epochs_arr, loss_log[:, 4], lw=1.4, label="UpYY")
    plt.plot(epochs_arr, loss_log[:, 5], lw=1.4, label="Down")
    plt.plot(epochs_arr, loss_log[:, 6], lw=1.4, label="Up")
    plt.plot(epochs_arr, loss_log[:, 7], lw=1.4, label="Left")
    plt.plot(epochs_arr, loss_log[:, 8], lw=1.4, label="Right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"loss_components_linear_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_arr, loss_log[:, 2], lw=1.4, label="Interior")
    plt.plot(epochs_arr, loss_log[:, 3], lw=1.4, label="DownYY")
    plt.plot(epochs_arr, loss_log[:, 4], lw=1.4, label="UpYY")
    plt.plot(epochs_arr, loss_log[:, 5], lw=1.4, label="Down")
    plt.plot(epochs_arr, loss_log[:, 6], lw=1.4, label="Up")
    plt.plot(epochs_arr, loss_log[:, 7], lw=1.4, label="Left")
    plt.plot(epochs_arr, loss_log[:, 8], lw=1.4, label="Right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(f"loss_components_semilogy_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()
    loss_txt = f"{FIELD_NAME}_loss_per_epoch_lr{lr_str}.txt"
    with open(loss_txt, "w", encoding="utf-8") as f:
        f.write("epoch total_loss interior_loss down_yy_loss up_yy_loss down_loss up_loss left_loss right_loss\n")
        for row in loss_log:
            f.write(f"{int(row[0])} {row[1]:.8e} {row[2]:.8e} {row[3]:.8e} {row[4]:.8e} {row[5]:.8e} {row[6]:.8e} {row[7]:.8e} {row[8]:.8e}\n")
    return loss_txt

def train():
    print("Using device:", device)
    print("Method:", METHOD_NAME)
    model = ResidualAttentionPINN().to(device)
    cuda_warmup()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_rows = []
    epoch_durations = []
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
        row = [
            i,
            float(l_total.detach().cpu().item()),
            float(L1.detach().cpu().item()),
            float(L2.detach().cpu().item()),
            float(L3.detach().cpu().item()),
            float(L4.detach().cpu().item()),
            float(L5.detach().cpu().item()),
            float(L6.detach().cpu().item()),
            float(L7.detach().cpu().item()),
        ]
        loss_rows.append(row)
        ep_t1 = time.perf_counter()
        epoch_durations.append(ep_t1 - ep_t0)
        if i % 100 == 0 or i == 1:
            print(f"epoch {i:5d} | total_loss = {row[1]:.6e} | interior = {row[2]:.6e} | downyy = {row[3]:.6e} | upyy = {row[4]:.6e} | down = {row[5]:.6e} | up = {row[6]:.6e} | left = {row[7]:.6e} | right = {row[8]:.6e}")
    train_t1 = time.perf_counter()
    total_train_sec = train_t1 - train_t0
    loss_log = np.array(loss_rows, dtype=float)
    loss_txt = save_loss_outputs(loss_log, LR)
    eval_t0 = time.perf_counter()
    eval_files = evaluate_and_save_results(model, lr_value=LR, grid_n=max(150, h))
    eval_t1 = time.perf_counter()
    eval_sec = eval_t1 - eval_t0
    peak_mem = torch.cuda.max_memory_allocated() if device.type == "cuda" else -1
    lr_str = f"{LR:.0e}"
    cost_txt = f"{FIELD_NAME}_compute_cost_lr{lr_str}.txt"
    with open(cost_txt, "w", encoding="utf-8") as f:
        f.write(f"Method: {METHOD_NAME}\n")
        f.write(f"FieldName: {FIELD_NAME}\n")
        f.write(f"Device: {device.type}\n")
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"ParameterCount: {param_count}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"LearningRate: {LR}\n")
        f.write(f"InteriorPoints: {N}\n")
        f.write(f"BoundaryPoints: {N1}\n")
        f.write(f"GridEval: {h}\n")
        f.write(f"ModelWidth: {MODEL_WIDTH}\n")
        f.write(f"NumBlocks: {NUM_BLOCKS}\n")
        f.write(f"AttentionHidden: {ATTN_HIDDEN}\n")
        f.write(f"ResidualScale: {RES_SCALE}\n")
        f.write(f"AttentionTemperature: {ATTN_TEMPERATURE}\n")
        f.write(f"GateBiasInit: {GATE_BIAS}\n")
        f.write(f"DropoutP: {DROPOUT_P}\n")
        f.write(f"TotalTrainingSeconds: {total_train_sec:.6f}\n")
        f.write(f"EvalSeconds: {eval_sec:.6f}\n")
        f.write(f"MeanEpochSeconds: {np.mean(epoch_durations):.6f}\n")
        f.write(f"StdEpochSeconds: {np.std(epoch_durations):.6f}\n")
        f.write(f"PeakGPUMemoryBytes: {peak_mem}\n")
        f.write("EpochDurationsSeconds:\n")
        for ep, dt in enumerate(epoch_durations, start=1):
            f.write(f"{ep} {dt:.8e}\n")
    return model, loss_log, eval_files, loss_txt, cost_txt

if __name__ == "__main__":
    train()
