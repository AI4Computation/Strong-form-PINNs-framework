import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import os
import sys
from contextlib import redirect_stdout
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 25

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L = 0.5
R = 0.1
E = 1.333
nu = 0.3333

def load_reference_data(P=None, P_top=None):
    try:
        if P is not None and P_top is not None:
            p_val = abs(P) if P != 0 else '1e-300'
            ptop_val = abs(P_top) if P_top != 0 else '1e-300'
            filename = f'abaqus-p_lateral{p_val}-p_top{ptop_val}.csv'
        else:
            filename = '圆孔.csv'
        
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        print(f"成功读取参考数据文件: {filename}")
        return data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    except Exception as e:
        print(f"警告：无法读取参考数据文件 {filename if P is not None else '圆孔.csv'}: {e}")
        return None, None, None, None

def compute_error_at_iteration(model, iteration, ref_x, ref_y, ref_u1, ref_u2, lam, mu, output_dir, phase=""):
    if ref_x is None:
        print(f"{phase}Iteration {iteration}: 无参考数据，跳过误差计算")
        return
    
    xy_tensor = torch.tensor(np.stack([ref_x, ref_y], axis=1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        u_theta = model(xy_tensor)
        u_pred = apply_hard_bc(xy_tensor, u_theta).cpu().numpy()
    
    error_u_vals = u_pred[:, 0] - ref_u1
    error_v_vals = u_pred[:, 1] - ref_u2
    
    mean_error_u = np.mean(np.abs(error_u_vals))
    mean_error_v = np.mean(np.abs(error_v_vals))
    max_error_u = np.max(np.abs(error_u_vals))
    max_error_v = np.max(np.abs(error_v_vals))
    
    print(f"{phase}Iteration {iteration}:")
    print(f"  平均误差 - U: {mean_error_u:.6f}, V: {mean_error_v:.6f}")
    print(f"  最大误差 - U: {max_error_u:.6f}, V: {max_error_v:.6f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].tricontourf(ref_x, ref_y, u_pred[:, 0], levels=70, cmap='RdBu_r')
    axes[0].set_title(f'Iter {iteration}')
    axes[0].set_xticks([-L, 0, L])
    axes[0].set_yticks([-L, 0, L])
    axes[0].set_aspect('equal')
    axes[0].tick_params(axis='both', pad=12)
    circle = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[0].add_patch(circle)
    cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.062, pad=0.08, aspect=14)
    cb1.locator = ticker.MaxNLocator(nbins=3)
    cb1.update_ticks()
    
    im2 = axes[1].tricontourf(ref_x, ref_y, u_pred[:, 1], levels=70, cmap='RdBu_r')
    axes[1].set_title(f'Iter {iteration}')
    axes[1].set_xticks([-L, 0, L])
    axes[1].set_yticks([-L, 0, L])
    axes[1].set_aspect('equal')
    axes[1].tick_params(axis='both', pad=12)
    circle = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[1].add_patch(circle)
    cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.062, pad=0.08, aspect=14)
    cb2.locator = ticker.MaxNLocator(nbins=3)
    cb2.update_ticks()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'displacement_iter_{iteration}.svg'), format='svg', bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].tricontourf(ref_x, ref_y, error_u_vals, levels=70, cmap='RdBu_r')
    axes[0].set_title(f'Iter {iteration}')
    axes[0].set_xticks([-L, 0, L])
    axes[0].set_yticks([-L, 0, L])
    axes[0].set_aspect('equal')
    axes[0].tick_params(axis='both', pad=12)
    circle = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[0].add_patch(circle)
    cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.062, pad=0.08, aspect=14)
    cb1.locator = ticker.MaxNLocator(nbins=3)
    cb1.update_ticks()
    
    im2 = axes[1].tricontourf(ref_x, ref_y, error_v_vals, levels=70, cmap='RdBu_r')
    axes[1].set_title(f'Iter {iteration}')
    axes[1].set_xticks([-L, 0, L])
    axes[1].set_yticks([-L, 0, L])
    axes[1].set_aspect('equal')
    axes[1].tick_params(axis='both', pad=12)
    circle = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[1].add_patch(circle)
    cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.062, pad=0.08, aspect=14)
    cb2.locator = ticker.MaxNLocator(nbins=3)
    cb2.update_ticks()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'error_iter_{iteration}.svg'), format='svg', bbox_inches='tight')
    plt.close()

class PINN_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 2)
        )
    
    def forward(self, x):
        return self.net(x)

def apply_hard_bc(xy, u_theta):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    u = (x**2 + (y + L)**2) * u_theta[:, 0:1]
    v = (y + L) * u_theta[:, 1:2]
    return torch.cat([u, v], dim=1)

def generate_points(n_domain, n_boundary_right, n_boundary_other):
    domain_pts = []
    pts = (torch.rand(n_domain*2, 2, device=DEVICE) * 2 - 1) * L
    r = torch.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    valid = (r > R)
    domain_pts.extend(pts[valid])
    domain_pts = domain_pts[:n_domain]
    domain_pts = torch.stack(domain_pts[:])
    
    y_right = (torch.rand(n_boundary_right, 1, device=DEVICE) * 2 - 1) * L
    x_right = torch.ones(n_boundary_right, 1, device=DEVICE) * (-L)
    boundary_left = torch.cat([x_right, y_right], dim=1)
    
    y_right = (torch.rand(n_boundary_right, 1, device=DEVICE) * 2 - 1) * L
    x_right = torch.ones(n_boundary_right, 1, device=DEVICE) * L
    boundary_right = torch.cat([x_right, y_right], dim=1)
    
    x_top = (torch.rand(n_boundary_right, 1, device=DEVICE) * 2 - 1) * L
    y_top = torch.ones(n_boundary_right, 1, device=DEVICE) * L
    boundary_top = torch.cat([x_top, y_top], dim=1)
    
    x_bottom = (torch.rand(n_boundary_right, 1, device=DEVICE) * 2 - 1) * L
    y_bottom = torch.ones(n_boundary_right, 1, device=DEVICE) * (-L)
    boundary_bottom = torch.cat([x_bottom, y_bottom], dim=1)

    theta = torch.linspace(0, 2*np.pi, n_boundary_other, device=DEVICE)
    x_hole = R * torch.cos(theta).unsqueeze(1)
    y_hole = R * torch.sin(theta).unsqueeze(1)
    boundary_hole = torch.cat([x_hole, y_hole], dim=1)
    
    return domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole

def compute_loss_strong(model, domain_points, boundary_left, boundary_right, boundary_top, 
                       boundary_bottom, boundary_hole, lam, mu, P, P_top):
    xy_domain = domain_points.clone().requires_grad_(True)
    u_theta = model(xy_domain)
    u = apply_hard_bc(xy_domain, u_theta)
    
    u_x = u[:, 0:1]
    u_y = u[:, 1:2]
    
    du_dx = torch.autograd.grad(u_x.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    du_dy = torch.autograd.grad(u_x.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    dv_dx = torch.autograd.grad(u_y.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    dv_dy = torch.autograd.grad(u_y.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    
    eps_xx = du_dx
    eps_yy = dv_dy
    eps_xy = 0.5 * (du_dy + dv_dx)
    
    tr_eps = eps_xx + eps_yy
    sigma_xx = lam * tr_eps + 2 * mu * eps_xx
    sigma_yy = lam * tr_eps + 2 * mu * eps_yy
    sigma_xy = 2 * mu * eps_xy
    
    dsigma_xx_dx = torch.autograd.grad(sigma_xx.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    dsigma_xy_dy = torch.autograd.grad(sigma_xy.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    dsigma_xy_dx = torch.autograd.grad(sigma_xy.sum(), xy_domain, create_graph=True)[0][:, 0:1]
    dsigma_yy_dy = torch.autograd.grad(sigma_yy.sum(), xy_domain, create_graph=True)[0][:, 1:2]
    
    residual_x = dsigma_xx_dx + dsigma_xy_dy
    residual_y = dsigma_xy_dx + dsigma_yy_dy
    
    loss_pde = torch.mean(residual_x**2) + torch.mean(residual_y**2)
    
    xy_right = boundary_right.clone().requires_grad_(True)
    u_theta_right = model(xy_right)
    u_right = apply_hard_bc(xy_right, u_theta_right)
    
    du_dx_r = torch.autograd.grad(u_right[:, 0].sum(), xy_right, create_graph=True)[0][:, 0]
    du_dy_r = torch.autograd.grad(u_right[:, 0].sum(), xy_right, create_graph=True)[0][:, 1]
    dv_dx_r = torch.autograd.grad(u_right[:, 1].sum(), xy_right, create_graph=True)[0][:, 0]
    dv_dy_r = torch.autograd.grad(u_right[:, 1].sum(), xy_right, create_graph=True)[0][:, 1]
    
    eps_xx_r = du_dx_r
    eps_yy_r = dv_dy_r
    eps_xy_r = 0.5 * (du_dy_r + dv_dx_r)
    
    tr_eps_r = eps_xx_r + eps_yy_r
    sigma_xx_r = lam * tr_eps_r + 2 * mu * eps_xx_r
    sigma_xy_r = 2 * mu * eps_xy_r
    
    traction_x_error = sigma_xx_r - P
    traction_y_error = sigma_xy_r - 0
    
    loss_bc_right = torch.mean(traction_x_error**2) + torch.mean(traction_y_error**2)
    
    xy_left = boundary_left.clone().requires_grad_(True)
    u_theta_left = model(xy_left)
    u_left = apply_hard_bc(xy_left, u_theta_left)
    
    du_dx_l = torch.autograd.grad(u_left[:, 0].sum(), xy_left, create_graph=True)[0][:, 0]
    du_dy_l = torch.autograd.grad(u_left[:, 0].sum(), xy_left, create_graph=True)[0][:, 1]
    dv_dx_l = torch.autograd.grad(u_left[:, 1].sum(), xy_left, create_graph=True)[0][:, 0]
    dv_dy_l = torch.autograd.grad(u_left[:, 1].sum(), xy_left, create_graph=True)[0][:, 1]
    
    eps_xx_l = du_dx_l
    eps_yy_l = dv_dy_l
    eps_xy_l = 0.5 * (du_dy_l + dv_dx_l)
    
    tr_eps_l = eps_xx_l + eps_yy_l
    sigma_xx_l = lam * tr_eps_l + 2 * mu * eps_xx_l
    sigma_xy_l = 2 * mu * eps_xy_l
    
    traction_x_error = sigma_xx_l - P
    traction_y_error = sigma_xy_l - 0
    
    loss_bc_left = torch.mean(traction_x_error**2) + torch.mean(traction_y_error**2)

    xy_top = boundary_top.clone().requires_grad_(True)
    u_theta_top = model(xy_top)
    u_top = apply_hard_bc(xy_top, u_theta_top)
    
    du_dx_t = torch.autograd.grad(u_top[:, 0].sum(), xy_top, create_graph=True)[0][:, 0]
    du_dy_t = torch.autograd.grad(u_top[:, 0].sum(), xy_top, create_graph=True)[0][:, 1]
    dv_dx_t = torch.autograd.grad(u_top[:, 1].sum(), xy_top, create_graph=True)[0][:, 0]
    dv_dy_t = torch.autograd.grad(u_top[:, 1].sum(), xy_top, create_graph=True)[0][:, 1]
    
    eps_xx_t = du_dx_t
    eps_yy_t = dv_dy_t
    eps_xy_t = 0.5 * (du_dy_t + dv_dx_t)
    
    tr_eps_t = eps_xx_t + eps_yy_t
    sigma_yy_t = lam * tr_eps_t + 2 * mu * eps_yy_t
    sigma_xy_t = 2 * mu * eps_xy_t
    
    traction_x_error = sigma_xy_t - 0
    traction_y_error = sigma_yy_t - P_top
    
    loss_bc_top = torch.mean(traction_x_error**2) + torch.mean(traction_y_error**2)
    
    xy_bottom = boundary_bottom.clone().requires_grad_(True)
    u_theta_bottom = model(xy_bottom)
    u_bottom = apply_hard_bc(xy_bottom, u_theta_bottom)
    
    du_dx_b = torch.autograd.grad(u_bottom[:, 0].sum(), xy_bottom, create_graph=True)[0][:, 0]
    du_dy_b = torch.autograd.grad(u_bottom[:, 0].sum(), xy_bottom, create_graph=True)[0][:, 1]
    dv_dx_b = torch.autograd.grad(u_bottom[:, 1].sum(), xy_bottom, create_graph=True)[0][:, 0]
    dv_dy_b = torch.autograd.grad(u_bottom[:, 1].sum(), xy_bottom, create_graph=True)[0][:, 1]
    
    eps_xx_b = du_dx_b
    eps_yy_b = dv_dy_b
    eps_xy_b = 0.5 * (du_dy_b + dv_dx_b)
    
    tr_eps_b = eps_xx_b + eps_yy_b
    sigma_xy_b = 2 * mu * eps_xy_b
    
    traction_x_error = sigma_xy_b - 0
    
    loss_bc_bottom = torch.mean(traction_x_error**2)
    
    xy_hole = boundary_hole.clone().requires_grad_(True)
    u_theta_hole = model(xy_hole)
    u_hole = apply_hard_bc(xy_hole, u_theta_hole)
    
    du_dx_h = torch.autograd.grad(u_hole[:, 0].sum(), xy_hole, create_graph=True)[0][:, 0]
    du_dy_h = torch.autograd.grad(u_hole[:, 0].sum(), xy_hole, create_graph=True)[0][:, 1]
    dv_dx_h = torch.autograd.grad(u_hole[:, 1].sum(), xy_hole, create_graph=True)[0][:, 0]
    dv_dy_h = torch.autograd.grad(u_hole[:, 1].sum(), xy_hole, create_graph=True)[0][:, 1]
    
    eps_xx_h = du_dx_h
    eps_yy_h = dv_dy_h
    eps_xy_h = 0.5 * (du_dy_h + dv_dx_h)
    
    tr_eps_h = eps_xx_h + eps_yy_h
    sigma_xx_h = lam * tr_eps_h + 2 * mu * eps_xx_h
    sigma_yy_h = lam * tr_eps_h + 2 * mu * eps_yy_h
    sigma_xy_h = 2 * mu * eps_xy_h
    
    n_x = -xy_hole[:, 0] / R
    n_y = -xy_hole[:, 1] / R
    
    traction_x_hole = sigma_xx_h * n_x + sigma_xy_h * n_y
    traction_y_hole = sigma_xy_h * n_x + sigma_yy_h * n_y
    
    loss_bc_hole = torch.mean(traction_x_hole**2 + traction_y_hole**2)
    
    w_pde = 1.0
    w_bc = 10.0
    
    total_loss = w_pde * loss_pde + w_bc * (loss_bc_left + loss_bc_right + loss_bc_top + loss_bc_bottom + loss_bc_hole)
    
    return total_loss, loss_pde, loss_bc_left, loss_bc_right, loss_bc_top, loss_bc_bottom, loss_bc_hole

def train_lbfgs(P, P_top, n_domain=6000, n_boundary=500, lbfgs_max_iter=15000, output_dir='results'):
    lam = E * nu / ((1 + nu) * (1 - 2*nu))
    mu = E / (2 * (1 + nu))
    
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data(P, P_top)
    
    check_points = [2000, 5000, 10000, 15000]
    
    model = PINN_Network().to(DEVICE)
    domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, boundary_hole = generate_points(
        n_domain, n_boundary, n_boundary*2
    )
    
    loss_history = []
    
    start_time = time.time()
    
    print("=" * 60)
    print(f"L-BFGS优化训练 (P={P}, P_top={P_top})")
    print("=" * 60)
    
    lbfgs_optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_max_iter,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-7,
        tolerance_change=1e-12
    )
    
    iteration = [0]
    
    def closure():
        lbfgs_optimizer.zero_grad()
        loss, loss_pde, loss_bc_l, loss_bc_r, loss_bc_t, loss_bc_b, loss_bc_h = compute_loss_strong(
            model, domain_pts, boundary_left, boundary_right, boundary_top, boundary_bottom, 
            boundary_hole, lam, mu, P, P_top
        )
        loss.backward()
        
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        if iteration[0] in check_points:
            current_time = time.time() - start_time
            print(f"\n训练时长: {current_time:.2f} 秒")
            compute_error_at_iteration(model, iteration[0], ref_x, ref_y, ref_u1, ref_u2, 
                                     lam, mu, output_dir, "L-BFGS ")
        
        if iteration[0] % 100 == 0:
            print(f"L-BFGS Iter {iteration[0]:5d}: Loss={loss_value:.6f}, "
                  f"PDE={loss_pde:.6f}, BC_right={loss_bc_r:.6f}")
        
        iteration[0] += 1
        return loss
    
    lbfgs_optimizer.step(closure)
    
    training_time = time.time() - start_time
    print(f"\n总训练时间: {training_time:.2f} 秒")
    print(f"最终损失: {loss_history[-1]:.6f}")
    print(f"总迭代数: {iteration[0]}")
    
    return model, loss_history, training_time

def visualize_final_results(model, P, P_top, output_dir):
    ref_x, ref_y, ref_u1, ref_u2 = load_reference_data(P, P_top)
    xy_tensor = torch.tensor(np.stack([ref_x, ref_y], axis=1), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        u_theta = model(xy_tensor)
        u_pred = apply_hard_bc(xy_tensor, u_theta).cpu().numpy()
    
    U = u_pred[:, 0]
    V = u_pred[:, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].tricontourf(ref_x, ref_y, U, levels=70, cmap='RdBu_r')
    axes[0].set_title('  ')
    axes[0].set_aspect('equal')
    axes[0].set_xticks([-L, 0, L])
    axes[0].set_yticks([-L, 0, L])
    axes[0].tick_params(axis='both', pad=12)
    circle1 = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[0].add_patch(circle1)
    cb1 = plt.colorbar(im1, ax=axes[0], fraction=0.062, pad=0.08, aspect=14)
    cb1.locator = ticker.MaxNLocator(nbins=3)
    cb1.update_ticks()
    
    im2 = axes[1].tricontourf(ref_x, ref_y, V, levels=70, cmap='RdBu_r')
    axes[1].set_title('  ')
    axes[1].set_aspect('equal')
    axes[1].set_xticks([-L, 0, L])
    axes[1].set_yticks([-L, 0, L])
    axes[1].tick_params(axis='both', pad=12)
    circle2 = plt.Circle((0, 0), R, fill=True, facecolor='white', edgecolor='black', linewidth=0)
    axes[1].add_patch(circle2)
    cb2 = plt.colorbar(im2, ax=axes[1], fraction=0.062, pad=0.08, aspect=14)
    cb2.locator = ticker.MaxNLocator(nbins=3)
    cb2.update_ticks()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_displacement.svg'), format='svg', bbox_inches='tight')
    plt.close()

def plot_loss_history(loss_history, P, P_top, output_dir):
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(loss_history)), loss_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('  ')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_history.svg'), format='svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("L-BFGS优化PINNs - 参数敏感性分析")
    print("=" * 60)
    print(f"问题参数：")
    print(f"板尺寸: {L} × {L} mm")
    print(f"圆孔半径: {R} mm")
    print(f"杨氏模量: {E} MPa")
    print(f"泊松比: {nu}")
    print("=" * 60)
    
    P_values = [0, -1, -2, -3, -4, -5]
    P_top_values = [0, -1, -2, -3, -4, -5]
    
    main_results_dir = 'sensitivity_results'
    if not os.path.exists(main_results_dir):
        os.makedirs(main_results_dir)
    
    summary_file = os.path.join(main_results_dir, 'summary.txt')
    
    total_cases = len(P_values) * len(P_top_values)
    case_num = 0
    
    results_data = []
    
    with open(summary_file, 'w', encoding='utf-8') as summary:
        summary.write("参数敏感性分析汇总\n")
        summary.write("=" * 80 + "\n")
        summary.write(f"总试验数: {total_cases}\n")
        summary.write(f"P取值: {P_values}\n")
        summary.write(f"P_top取值: {P_top_values}\n")
        summary.write("=" * 80 + "\n\n")
        
        for P in P_values:
            for P_top in P_top_values:
                case_num += 1
                print(f"\n{'='*80}")
                print(f"试验 {case_num}/{total_cases}: P={P}, P_top={P_top}")
                print(f"{'='*80}\n")
                
                case_dir = os.path.join(main_results_dir, f'P_{P}_Ptop_{P_top}')
                if not os.path.exists(case_dir):
                    os.makedirs(case_dir)
                
                log_file = os.path.join(case_dir, 'training_log.txt')
                
                with open(log_file, 'w', encoding='utf-8') as f:
                    with redirect_stdout(f):
                        model, loss_history, training_time = train_lbfgs(
                            P=P, P_top=P_top, 
                            n_domain=6000, 
                            n_boundary=500,
                            lbfgs_max_iter=20000,
                            output_dir=case_dir
                        )
                        
                        print("\n生成最终误差图...")
                        ref_x, ref_y, ref_u1, ref_u2 = load_reference_data(P, P_top)
                        if ref_x is not None:
                            lam = E * nu / ((1 + nu) * (1 - 2*nu))
                            mu = E / (2 * (1 + nu))
                            compute_error_at_iteration(model, len(loss_history), ref_x, ref_y, 
                                                     ref_u1, ref_u2, lam, mu, case_dir, "Final ")
                        
                        if ref_x is not None:
                            xy_tensor = torch.tensor(np.stack([ref_x, ref_y], axis=1), 
                                                   dtype=torch.float32, device=DEVICE)
                            with torch.no_grad():
                                u_theta = model(xy_tensor)
                                u_pred = apply_hard_bc(xy_tensor, u_theta).cpu().numpy()
                            
                            mse_u = np.mean((u_pred[:, 0] - ref_u1)**2)
                            mse_v = np.mean((u_pred[:, 1] - ref_u2)**2)
                            mape_u = np.mean(np.abs((ref_u1 - u_pred[:, 0]) / (np.abs(ref_u1) + 1e-8))) * 100
                            mape_v = np.mean(np.abs((ref_u2 - u_pred[:, 1]) / (np.abs(ref_u2) + 1e-8))) * 100
                            r_u = np.corrcoef(ref_u1, u_pred[:, 0])[0, 1]
                            r_v = np.corrcoef(ref_u2, u_pred[:, 1])[0, 1]
                            
                            print(f"\n最终误差指标：")
                            print(f"水平位移(U) - MSE: {mse_u:.8e}, MAPE: {mape_u:.4f}%, R: {r_u:.6f}")
                            print(f"竖向位移(V) - MSE: {mse_v:.8e}, MAPE: {mape_v:.4f}%, R: {r_v:.6f}")
                        else:
                            mse_u = mse_v = mape_u = mape_v = r_u = r_v = None
                        
                        results_data.append({
                            'P': P,
                            'P_top': P_top,
                            'MSE_U': mse_u,
                            'MSE_V': mse_v,
                            'MAPE_U': mape_u,
                            'MAPE_V': mape_v,
                            'R_U': r_u,
                            'R_V': r_v
                        })
                        
                        model_filename = f'P_{P}_Ptop_{P_top}.pth'
                        model_path = os.path.join(case_dir, model_filename)
                        torch.save(model, model_path)
                        print(f"\n模型已完整保存到: {model_path}")
                
                loss_df = pd.DataFrame({
                    'Iteration': range(len(loss_history)),
                    'Loss': loss_history
                })
                loss_df.to_csv(os.path.join(case_dir, 'loss_history.csv'), index=False)
                
                plot_loss_history(loss_history, P, P_top, case_dir)
                
                visualize_final_results(model, P, P_top, case_dir)
                
                summary.write(f"试验 {case_num}: P={P}, P_top={P_top}\n")
                summary.write(f"  训练时间: {training_time:.2f} 秒\n")
                summary.write(f"  最终损失: {loss_history[-1]:.6f}\n")
                summary.write(f"  总迭代数: {len(loss_history)}\n")
                if ref_x is not None:
                    summary.write(f"  MSE_U: {mse_u:.8e}, MSE_V: {mse_v:.8e}\n")
                    summary.write(f"  MAPE_U: {mape_u:.4f}%, MAPE_V: {mape_v:.4f}%\n")
                    summary.write(f"  R_U: {r_u:.6f}, R_V: {r_v:.6f}\n")
                summary.write(f"  结果保存在: {case_dir}\n")
                summary.write(f"  模型保存为: {model_filename}\n")
                summary.write("-" * 80 + "\n")
                summary.flush()
                
                print(f"\n试验 {case_num}/{total_cases} 完成！结果已保存到 {case_dir}")
    
    print(f"\n{'='*80}")
    print("所有参数敏感性分析完成！")
    print(f"汇总结果保存在: {summary_file}")
    
    print("\n正在生成误差指标汇总Excel文件...")
    results_df = pd.DataFrame(results_data)
    
    metrics = {
        'MSE_U': 'MSE_U.xlsx',
        'MSE_V': 'MSE_V.xlsx',
        'MAPE_U': 'MAPE_U.xlsx',
        'MAPE_V': 'MAPE_V.xlsx',
        'R_U': 'R_U.xlsx',
        'R_V': 'R_V.xlsx'
    }
    
    for metric, filename in metrics.items():
        pivot_table = results_df.pivot(index='P', columns='P_top', values=metric)
        pivot_table = pivot_table.sort_index(ascending=False).sort_index(axis=1, ascending=False)
        excel_path = os.path.join(main_results_dir, filename)
        pivot_table.to_excel(excel_path)
        print(f"  {metric} 已保存到: {excel_path}")
    
    print(f"{'='*80}")
