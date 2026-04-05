
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time

epochs = 5000
LR = 3e-4
N = 1000
N1 = 1000
h = 100
FIELD_NAME = "uy"
METHOD_NAME = "XPINN"
PDE_NAME_SHORT = "BiharmonicMixedBC2D"
PDE_NAME = "u_xx - u_yyyy = (2 - x^2) exp(-y), u=x^2 exp(-y)"
NUM_SUB_X = 2
NUM_SUB_Y = 2
X_SPLITS = [0.0, 0.38, 1.0]
Y_SPLITS = [0.0, 0.62, 1.0]
SUBNET_WIDTH = 16
SUBNET_DEPTH = 2
INTERFACE_N = 24
W_PHYS = 1.0
W_BC_U = 1.0
W_BC_YY = 1.0
W_IF_VALUE = 1e-6
W_IF_DX = 1e-8
W_IF_DY = 1e-8
USE_LOCAL_INPUT = False
SEED = 888888

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

def analytic_u(x, y):
    return (x ** 2) * torch.exp(-y)

def gradients(u, x, order: int = 1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, only_inputs=True)[0]
    return gradients(gradients(u, x), x, order=order - 1)

def build_subdomains():
    bounds = []
    for j in range(NUM_SUB_Y):
        for i in range(NUM_SUB_X):
            bounds.append((X_SPLITS[i], X_SPLITS[i + 1], Y_SPLITS[j], Y_SPLITS[j + 1], i, j))
    return bounds

SUBDOMAIN_BOUNDS = build_subdomains()

def map_to_local(z, a, b):
    return 2.0 * (z - a) / (b - a) - 1.0

def preprocess_input(x, y, bounds):
    xl, xr, yl, yr, _, _ = bounds
    if USE_LOCAL_INPUT:
        xx = map_to_local(x, xl, xr)
        yy = map_to_local(y, yl, yr)
        return torch.cat([xx, yy], dim=1)
    return torch.cat([x, y], dim=1)

class SubNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = [torch.nn.Linear(2, SUBNET_WIDTH), torch.nn.Tanh()]
        for _ in range(SUBNET_DEPTH - 1):
            layers += [torch.nn.Linear(SUBNET_WIDTH, SUBNET_WIDTH), torch.nn.Tanh()]
        layers += [torch.nn.Linear(SUBNET_WIDTH, 1)]
        self.net = torch.nn.Sequential(*layers)
    def forward(self, z):
        return self.net(z)

class XPINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.subnets = torch.nn.ModuleList([SubNet() for _ in range(NUM_SUB_X * NUM_SUB_Y)])
    def forward_subdomain(self, idx, x, y):
        return self.subnets[idx](preprocess_input(x, y, SUBDOMAIN_BOUNDS[idx]))

def sample_interior(bounds, n):
    xl, xr, yl, yr, _, _ = bounds
    x = xl + (xr - xl) * torch.rand(n, 1, device=device)
    y = yl + (yr - yl) * torch.rand(n, 1, device=device)
    cond = (2.0 - x ** 2) * torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond

def sample_y0(bounds, n):
    xl, xr, _, _, _, _ = bounds
    x = xl + (xr - xl) * torch.rand(n, 1, device=device)
    y = torch.zeros_like(x)
    cond = x ** 2
    return x.requires_grad_(True), y.requires_grad_(True), cond

def sample_y1(bounds, n):
    xl, xr, _, _, _, _ = bounds
    x = xl + (xr - xl) * torch.rand(n, 1, device=device)
    y = torch.ones_like(x)
    cond = x ** 2 / torch.e
    return x.requires_grad_(True), y.requires_grad_(True), cond

def sample_x0(bounds, n):
    _, _, yl, yr, _, _ = bounds
    y = yl + (yr - yl) * torch.rand(n, 1, device=device)
    x = torch.zeros_like(y)
    cond = torch.zeros_like(y)
    return x.requires_grad_(True), y.requires_grad_(True), cond

def sample_x1(bounds, n):
    _, _, yl, yr, _, _ = bounds
    y = yl + (yr - yl) * torch.rand(n, 1, device=device)
    x = torch.ones_like(y)
    cond = torch.exp(-y)
    return x.requires_grad_(True), y.requires_grad_(True), cond

def sample_vertical_interface(x_fixed, y0, y1, n):
    y = y0 + (y1 - y0) * torch.rand(n, 1, device=device)
    x = x_fixed * torch.ones_like(y)
    return x.requires_grad_(True), y.requires_grad_(True)

def sample_horizontal_interface(y_fixed, x0, x1, n):
    x = x0 + (x1 - x0) * torch.rand(n, 1, device=device)
    y = y_fixed * torch.ones_like(x)
    return x.requires_grad_(True), y.requires_grad_(True)

def subdomain_pde_loss(model, idx, bounds):
    n = max(16, N // (NUM_SUB_X * NUM_SUB_Y))
    x, y, cond = sample_interior(bounds, n)
    u = model.forward_subdomain(idx, x, y)
    u_xx = gradients(u, x, 2)
    u_yyyy = gradients(u, y, 4)
    return MSE(u_xx - u_yyyy, cond)

def subdomain_boundary_u_loss(model, idx, bounds):
    xl, xr, yl, yr, i, j = bounds
    loss = torch.tensor(0.0, device=device)
    n_x = max(16, N1 // NUM_SUB_X)
    n_y = max(16, N1 // NUM_SUB_Y)
    if j == 0:
        x, y, cond = sample_y0(bounds, n_x)
        loss = loss + MSE(model.forward_subdomain(idx, x, y), cond)
    if j == NUM_SUB_Y - 1:
        x, y, cond = sample_y1(bounds, n_x)
        loss = loss + MSE(model.forward_subdomain(idx, x, y), cond)
    if i == 0:
        x, y, cond = sample_x0(bounds, n_y)
        loss = loss + MSE(model.forward_subdomain(idx, x, y), cond)
    if i == NUM_SUB_X - 1:
        x, y, cond = sample_x1(bounds, n_y)
        loss = loss + MSE(model.forward_subdomain(idx, x, y), cond)
    return loss

def subdomain_boundary_yy_loss(model, idx, bounds):
    _, _, _, _, _, j = bounds
    loss = torch.tensor(0.0, device=device)
    n_x = max(16, N1 // NUM_SUB_X)
    if j == 0:
        x, y, cond = sample_y0(bounds, n_x)
        u = model.forward_subdomain(idx, x, y)
        loss = loss + MSE(gradients(u, y, 2), cond)
    if j == NUM_SUB_Y - 1:
        x, y, cond = sample_y1(bounds, n_x)
        u = model.forward_subdomain(idx, x, y)
        loss = loss + MSE(gradients(u, y, 2), cond)
    return loss

def vertical_interface_loss(model, idx_left, idx_right, x_fixed, y0, y1):
    x, y = sample_vertical_interface(x_fixed, y0, y1, INTERFACE_N)
    u_l = model.forward_subdomain(idx_left, x, y)
    u_r = model.forward_subdomain(idx_right, x, y)
    ux_l = gradients(u_l, x, 1)
    ux_r = gradients(u_r, x, 1)
    uy_l = gradients(u_l, y, 1)
    uy_r = gradients(u_r, y, 1)
    return MSE(u_l, u_r), MSE(ux_l, ux_r), MSE(uy_l, uy_r)

def horizontal_interface_loss(model, idx_bottom, idx_top, y_fixed, x0, x1):
    x, y = sample_horizontal_interface(y_fixed, x0, x1, INTERFACE_N)
    u_b = model.forward_subdomain(idx_bottom, x, y)
    u_t = model.forward_subdomain(idx_top, x, y)
    ux_b = gradients(u_b, x, 1)
    ux_t = gradients(u_t, x, 1)
    uy_b = gradients(u_b, y, 1)
    uy_t = gradients(u_t, y, 1)
    return MSE(u_b, u_t), MSE(ux_b, ux_t), MSE(uy_b, uy_t)

def all_losses(model):
    phys_total = torch.tensor(0.0, device=device)
    bc_u_total = torch.tensor(0.0, device=device)
    bc_yy_total = torch.tensor(0.0, device=device)
    if_value_total = torch.tensor(0.0, device=device)
    if_dx_total = torch.tensor(0.0, device=device)
    if_dy_total = torch.tensor(0.0, device=device)
    for idx, bounds in enumerate(SUBDOMAIN_BOUNDS):
        phys_total = phys_total + subdomain_pde_loss(model, idx, bounds)
        bc_u_total = bc_u_total + subdomain_boundary_u_loss(model, idx, bounds)
        bc_yy_total = bc_yy_total + subdomain_boundary_yy_loss(model, idx, bounds)
    for j in range(NUM_SUB_Y):
        idx_left = j * NUM_SUB_X + 0
        idx_right = j * NUM_SUB_X + 1
        y0 = Y_SPLITS[j]
        y1 = Y_SPLITS[j + 1]
        lv, ldx, ldy = vertical_interface_loss(model, idx_left, idx_right, X_SPLITS[1], y0, y1)
        if_value_total = if_value_total + lv
        if_dx_total = if_dx_total + ldx
        if_dy_total = if_dy_total + ldy
    for i in range(NUM_SUB_X):
        idx_bottom = i
        idx_top = NUM_SUB_X + i
        x0 = X_SPLITS[i]
        x1 = X_SPLITS[i + 1]
        lv, ldx, ldy = horizontal_interface_loss(model, idx_bottom, idx_top, Y_SPLITS[1], x0, x1)
        if_value_total = if_value_total + lv
        if_dx_total = if_dx_total + ldx
        if_dy_total = if_dy_total + ldy
    total = W_PHYS * phys_total + W_BC_U * bc_u_total + W_BC_YY * bc_yy_total + W_IF_VALUE * if_value_total + W_IF_DX * if_dx_total + W_IF_DY * if_dy_total
    return total, phys_total, bc_u_total, bc_yy_total, if_value_total, if_dx_total, if_dy_total

@torch.no_grad()
def predict_grid(model, grid_n):
    model.eval()
    xs = torch.linspace(0.0, 1.0, grid_n, device=device)
    ys = torch.linspace(0.0, 1.0, grid_n, device=device)
    Xg, Yg = torch.meshgrid(xs, ys, indexing="xy")
    Up = torch.zeros_like(Xg)
    for idx, bounds in enumerate(SUBDOMAIN_BOUNDS):
        xl, xr, yl, yr, _, _ = bounds
        if xr < 1.0:
            x_mask = (Xg >= xl) & (Xg < xr)
        else:
            x_mask = (Xg >= xl) & (Xg <= xr)
        if yr < 1.0:
            y_mask = (Yg >= yl) & (Yg < yr)
        else:
            y_mask = (Yg >= yl) & (Yg <= yr)
        mask = x_mask & y_mask
        x_sub = Xg[mask].reshape(-1, 1)
        y_sub = Yg[mask].reshape(-1, 1)
        if x_sub.numel() > 0:
            Up[mask] = model.forward_subdomain(idx, x_sub, y_sub).reshape(-1)
    Ut = analytic_u(Xg, Yg)
    Err = (Up - Ut).abs()
    return Xg, Yg, Ut, Up, Err

@torch.no_grad()
def evaluate_and_save_results(model, lr_value: float, grid_n: int = 200):
    lr_str = f"{lr_value:.0e}"
    Xg, Yg, Ut, Up, AbsErr = predict_grid(model, grid_n)
    err = (Up - Ut).detach().cpu().numpy()
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    rel_l2 = float(np.linalg.norm(err.ravel(), 2) / (np.linalg.norm(Ut.detach().cpu().numpy().ravel(), 2) + 1e-12))
    maxae = float(np.max(np.abs(err)))
    max_idx = np.unravel_index(int(np.argmax(np.abs(err))), err.shape)
    max_i, max_j = int(max_idx[0]), int(max_idx[1])
    txt_name = f"{FIELD_NAME}_result_{METHOD_NAME}_lr{lr_str}_error.txt"
    with open(txt_name, "w", encoding="utf-8") as f:
        f.write(f"Method: {METHOD_NAME}\n")
        f.write(f"Field: {FIELD_NAME}\n")
        f.write(f"MSE: {mse:.8e}\n")
        f.write(f"RMSE: {rmse:.8e}\n")
        f.write(f"MAE: {mae:.8e}\n")
        f.write(f"RelativeL2: {rel_l2:.8e}\n")
        f.write(f"MaxAbsoluteError: {maxae:.8e}\n")
    Ut_np = Ut.detach().cpu().numpy()
    Up_np = Up.detach().cpu().numpy()
    Abs_np = AbsErr.detach().cpu().numpy()
    X_np = Xg.detach().cpu().numpy()
    Y_np = Yg.detach().cpu().numpy()
    vmin = float(min(Ut_np.min(), Up_np.min()))
    vmax = float(max(Ut_np.max(), Up_np.max()))
    plt.figure(figsize=(6, 5))
    plt.imshow(Ut_np.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(label=f"{FIELD_NAME}_true")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Analytic solution u(x,y)")
    plt.savefig(f"{FIELD_NAME}_true_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(6, 5))
    plt.imshow(Up_np.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(label=f"{FIELD_NAME}_{METHOD_NAME}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{METHOD_NAME} prediction")
    plt.savefig(f"{FIELD_NAME}_{METHOD_NAME.lower()}_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(6, 5))
    plt.imshow(Abs_np.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="jet")
    plt.colorbar(label=f"|{FIELD_NAME}_{METHOD_NAME} - {FIELD_NAME}_true|")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Maxerror")
    plt.savefig(f"{FIELD_NAME}_maxerror_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()
    _write_xyz(f"{FIELD_NAME}_true_xyz_lr{lr_str}.txt", X_np, Y_np, Ut_np)
    _write_xyz(f"{FIELD_NAME}_pred_xyz_lr{lr_str}.txt", X_np, Y_np, Up_np)
    _write_xyz(f"{FIELD_NAME}_maxerror_xyz_map_lr{lr_str}.txt", X_np, Y_np, Abs_np)
    max_x = float(X_np[max_i, max_j])
    max_y = float(Y_np[max_i, max_j])
    with open(f"{FIELD_NAME}_maxerror_xyz_lr{lr_str}.txt", "w", encoding="utf-8") as f:
        f.write("x y value\n")
        f.write(f"{max_x:.8e} {max_y:.8e} {maxae:.8e}\n")
    return {
        "metrics_txt": txt_name,
        "fig_true": f"{FIELD_NAME}_true_lr{lr_str}.png",
        "fig_pred": f"{FIELD_NAME}_{METHOD_NAME.lower()}_lr{lr_str}.png",
        "fig_maxerror": f"{FIELD_NAME}_maxerror_lr{lr_str}.png",
        "xyz_true": f"{FIELD_NAME}_true_xyz_lr{lr_str}.txt",
        "xyz_pred": f"{FIELD_NAME}_pred_xyz_lr{lr_str}.txt",
        "xyz_max_map": f"{FIELD_NAME}_maxerror_xyz_map_lr{lr_str}.txt",
        "xyz_max": f"{FIELD_NAME}_maxerror_xyz_lr{lr_str}.txt"
    }

def save_loss_outputs(loss_log, lr_value: float):
    lr_str = f"{lr_value:.0e}"
    epochs_arr = np.arange(1, loss_log.shape[0] + 1)
    plt.figure()
    plt.plot(epochs_arr, loss_log[:, 1], lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Training Loss (Linear)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"loss_curve_linear_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure()
    plt.plot(epochs_arr, loss_log[:, 1], lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Training Loss (Semilogy)")
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(f"loss_curve_semilogy_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure()
    plt.plot(epochs_arr, loss_log[:, 2], lw=1.4, label="physics")
    plt.plot(epochs_arr, loss_log[:, 3], lw=1.4, label="bc_u")
    plt.plot(epochs_arr, loss_log[:, 4], lw=1.4, label="bc_yy")
    plt.plot(epochs_arr, loss_log[:, 5], lw=1.4, label="if_value")
    plt.plot(epochs_arr, loss_log[:, 6], lw=1.4, label="if_dx")
    plt.plot(epochs_arr, loss_log[:, 7], lw=1.4, label="if_dy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Components (Linear)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"loss_components_linear_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure()
    plt.plot(epochs_arr, loss_log[:, 2], lw=1.4, label="physics")
    plt.plot(epochs_arr, loss_log[:, 3], lw=1.4, label="bc_u")
    plt.plot(epochs_arr, loss_log[:, 4], lw=1.4, label="bc_yy")
    plt.plot(epochs_arr, loss_log[:, 5], lw=1.4, label="if_value")
    plt.plot(epochs_arr, loss_log[:, 6], lw=1.4, label="if_dx")
    plt.plot(epochs_arr, loss_log[:, 7], lw=1.4, label="if_dy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Components (Semilogy)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(f"loss_components_semilogy_lr{lr_str}.png", dpi=300, bbox_inches="tight")
    plt.close()
    loss_txt = f"{FIELD_NAME}_loss_per_epoch_lr{lr_str}.txt"
    with open(loss_txt, "w", encoding="utf-8") as f:
        f.write("epoch total_loss physics_loss bc_u_loss bc_yy_loss interface_value_loss interface_dx_loss interface_dy_loss\n")
        for row in loss_log:
            f.write(f"{int(row[0])} {row[1]:.8e} {row[2]:.8e} {row[3]:.8e} {row[4]:.8e} {row[5]:.8e} {row[6]:.8e} {row[7]:.8e}\n")
    return loss_txt

def train():
    print("Using device:", device)
    print("Method:", METHOD_NAME)
    model = XPINN().to(device)
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
        total, phys, bc_u, bc_yy, ifv, ifdx, ifdy = all_losses(model)
        total.backward()
        opt.step()
        total_v = float(total.detach().cpu().item())
        phys_v = float(phys.detach().cpu().item())
        bc_u_v = float(bc_u.detach().cpu().item())
        bc_yy_v = float(bc_yy.detach().cpu().item())
        ifv_v = float(ifv.detach().cpu().item())
        ifdx_v = float(ifdx.detach().cpu().item())
        ifdy_v = float(ifdy.detach().cpu().item())
        loss_rows.append([i, total_v, phys_v, bc_u_v, bc_yy_v, ifv_v, ifdx_v, ifdy_v])
        ep_t1 = time.perf_counter()
        epoch_durations.append(ep_t1 - ep_t0)
        if i % 100 == 0 or i == 1:
            print(f"epoch {i:5d} | total_loss = {total_v:.6e} | phys = {phys_v:.6e} | bc_u = {bc_u_v:.6e} | bc_yy = {bc_yy_v:.6e} | ifv = {ifv_v:.6e} | ifdx = {ifdx_v:.6e} | ifdy = {ifdy_v:.6e}")
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
        f.write(f"FieldName: {FIELD_NAME}\n")
        f.write(f"Method: {METHOD_NAME}\n")
        f.write(f"PDENameShort: {PDE_NAME_SHORT}\n")
        f.write(f"PDENameFull: {PDE_NAME}\n")
        f.write(f"Device: {device.type}\n")
        f.write(f"GPU: {gpu_name}\n")
        f.write(f"ParameterCount: {param_count}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"LearningRate: {LR}\n")
        f.write(f"InteriorPoints: {N}\n")
        f.write(f"BoundaryPoints: {N1}\n")
        f.write(f"GridEval: {h}\n")
        f.write(f"NumSubX: {NUM_SUB_X}\n")
        f.write(f"NumSubY: {NUM_SUB_Y}\n")
        f.write(f"XSplits: {X_SPLITS}\n")
        f.write(f"YSplits: {Y_SPLITS}\n")
        f.write(f"SubNetWidth: {SUBNET_WIDTH}\n")
        f.write(f"SubNetDepth: {SUBNET_DEPTH}\n")
        f.write(f"InterfaceN: {INTERFACE_N}\n")
        f.write(f"W_PHYS: {W_PHYS}\n")
        f.write(f"W_BC_U: {W_BC_U}\n")
        f.write(f"W_BC_YY: {W_BC_YY}\n")
        f.write(f"W_IF_VALUE: {W_IF_VALUE}\n")
        f.write(f"W_IF_DX: {W_IF_DX}\n")
        f.write(f"W_IF_DY: {W_IF_DY}\n")
        f.write(f"USE_LOCAL_INPUT: {USE_LOCAL_INPUT}\n")
        f.write(f"TotalTrainingSeconds: {total_train_sec:.6f}\n")
        f.write(f"EvalSeconds: {eval_sec:.6f}\n")
        f.write("Epoch Durations (seconds):\n")
        for ep, dt in enumerate(epoch_durations, start=1):
            f.write(f"{ep:d} {dt:.8e}\n")
        f.write(f"MeanEpochSeconds: {np.mean(epoch_durations):.6f}\n")
        f.write(f"StdEpochSeconds: {np.std(epoch_durations):.6f}\n")
        f.write(f"PeakGPUMemoryBytes: {peak_mem}\n")
    return model, loss_log, eval_files, loss_txt, cost_txt

if __name__ == "__main__":
    train()
