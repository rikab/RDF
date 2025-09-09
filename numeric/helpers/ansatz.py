import torch
import math

import numpy as np

N_integrator = 5000
T_MAX = 50
eps = 1e-6


def get_factorial_cache(max_M, max_N, mstar, device):
    factorial_cache_n = torch.tensor(
        [math.factorial(k) for k in range(max_N)], device=device
    ).float()
    factorial_cache_m = torch.tensor(
        [math.factorial(k) for k in range(mstar, max_M + mstar - 1)], device=device
    ).float()
    n_range = torch.arange(max_N, device=device) # t^0 ... t^{N-1}
    m_range = torch.arange(mstar, max_M + mstar - 1, device=device) # alpha^{mstar} ... # alpha^{mstar-1}
    return factorial_cache_m, factorial_cache_n, m_range, n_range


def helper_theta(x, x0, temperature=100):
    #return torch.where(x >= 0, 1.0, 0.0)
    return torch.sigmoid(temperature * (x - x0))

def helper_theta_ratio(x, x_top, x_bottom, temperature=100):
    return torch.exp(torch.log(helper_theta(x, x_top, temperature) + eps) - torch.log(helper_theta(x, x_bottom, temperature)+ eps))


def helper_ReLU(x, x0 = 0, temperature = 100):

    # Softplus function
    #return torch.nn.Softplus(beta=temperature)(x - x0)
    return torch.abs(x)


def f(t, alpha, g_coeffs, theta, mstar, factorial_cache_info):

    B = t.shape[0]
    max_M, max_N = g_coeffs.shape

    factorial_cache_m, factorial_cache_n, m_range, n_range = factorial_cache_info

    t_exp = t.unsqueeze(1).expand(B, max_N)

    theta_0_exp = theta[0].unsqueeze(0).expand(B, max_N)
    
    n_range_exp = n_range.unsqueeze(0).expand(B, max_N)
    
    factorial_cache_n_exp = factorial_cache_n.unsqueeze(0).expand(B, max_N)
    t_powers = (t_exp**n_range) / factorial_cache_n  # (B, max_N)
    g_coeffs_0_exp = g_coeffs[0].unsqueeze(0).expand(B, max_N)
    heaviside_theta_gstar = helper_theta(t_exp, theta_0_exp)

    
    g_star = helper_ReLU(
        torch.sum(g_coeffs_0_exp * t_powers * heaviside_theta_gstar, dim=-1) # NEW
   ) 


    t_exp = t_exp.unsqueeze(1).expand(B, max_M - 1, max_N)
    t_powers_exp = t_powers.unsqueeze(1).expand(B, max_M - 1, max_N)
    g_1_exp = g_coeffs[1:].unsqueeze(0).expand(B, max_M - 1, max_N)


    

    """
    if theta[0,0] > theta[1,0]: 
        heaviside_theta_ghigher = helper_theta(t_exp, theta_0_exp) 
    else:
        heaviside_theta_ghigher = helper_theta(t_exp, theta_1_exp) 
    """
    #heaviside_theta_ghigher = helper_theta_ratio(t_exp, theta_1_exp, theta_0_exp) 
    ##heaviside_theta_ghigher = helper_theta(t_exp, theta_1_exp) 

    # below code only works for order 2
    theta_large = max(theta[0,0], theta[1,0])
    theta_small = min(theta[0,0], theta[1,0])

    heaviside_theta_ghigher = torch.zeros_like(t)

    # left region -- zero
    # middle region: ratio
    heaviside_theta_ghigher += ((t > theta_small) & ( t <= theta_large))*helper_theta_ratio(t, theta_small, theta_large) 
    # right region
    heaviside_theta_ghigher += (t > theta_large)*1.0
    heaviside_theta_ghigher = heaviside_theta_ghigher.unsqueeze(1).expand(B, max_M - 1).unsqueeze(2).expand(B, max_M-1, max_N)
    

   

    factorial_cache_m_exp = (
        factorial_cache_m.unsqueeze(0)
        .expand(B, max_M-1)
        .unsqueeze(2)
        .expand(B, max_M-1, max_N)
    )

   

    g_coeffs_higher = g_1_exp / factorial_cache_m_exp

    g_higher_mat = torch.sum(
        g_coeffs_higher * t_powers_exp * heaviside_theta_ghigher, dim=-1
    )  # (N_integrator, max_M - 1, max_N) -> (N_integrator, max_M - 1)

    g_higher = torch.sum((alpha ** m_range) * g_higher_mat, dim=-1)

    return  alpha**mstar * g_star * torch.exp(-g_higher)


def cumulative_trapezoidal(alpha, g_coeffs, theta, mstar, t_grid, device, factorial_cache_info):
    f_vals = f(t_grid, alpha, g_coeffs, theta, mstar, factorial_cache_info)
    dt = t_grid[1] - t_grid[0]
    cum = torch.cumsum((f_vals[:-1] + f_vals[1:]) * 0.5 * dt, dim=0)
    cum = torch.cat([torch.zeros(1, device=device), cum])
    return cum


def q(t, alpha, g_coeffs, theta, mstar, device):

    factorial_cache_info = get_factorial_cache(g_coeffs.shape[0], g_coeffs.shape[1], mstar, device)

    # added for 1d theta
    theta = theta.expand(-1, g_coeffs.shape[1])

    t_dense = torch.linspace(
        0, T_MAX, N_integrator, device=device
    )

    
    F_dense = cumulative_trapezoidal(alpha, g_coeffs, theta, mstar, t_dense, device, factorial_cache_info)

    # Interpolate
    idx = torch.searchsorted(t_dense, t.clamp(max=t_dense[-1]), right=True) - 1
    idx = idx.clamp(min=0, max=t_dense.shape[0] - 2)
    t0, t1 = t_dense[idx], t_dense[idx + 1]
    F0, F1 = F_dense[idx], F_dense[idx + 1]
    exp_term = F0 + (F1 - F0) * (t - t0) / (t1 - t0 + eps)


    
    return f(t, alpha, g_coeffs, theta, mstar, factorial_cache_info) * torch.exp(-exp_term)


def get_taylor_expanded_ansatz(fn, alpha_0, alpha, order_to_match):

    base = fn(alpha_0)  # (args.n_bins-1,)

    if order_to_match >= 1:
        _, d1 = torch.autograd.functional.jvp(
                fn,
                (alpha_0,),
                (torch.ones_like(alpha_0),),
                create_graph=True,
            )  # (args.n_bins-1,)
        d2 = None

    if order_to_match == 2:
        d1_fn = lambda a: torch.autograd.functional.jvp(
            fn, (a,), (torch.ones_like(a),), create_graph=True
        )[1]
        _, d2 = torch.autograd.functional.jvp(
            d1_fn,
            (alpha_0,),
            (torch.ones_like(alpha_0),),
            create_graph=True,
        )  # (args.n_bins-1,


    
    # construct the taylor expansion for the loc_alphas
    batch_ansatz = base
    try:
        
        if order_to_match >= 1:
            batch_ansatz = batch_ansatz + alpha[:, None] * d1
        if order_to_match == 2:
            batch_ansatz = batch_ansatz + 0.5 * (alpha[:, None] ** 2) * d2
    except:
        if order_to_match >= 1:
            batch_ansatz = batch_ansatz + alpha * d1
        if order_to_match == 2:
            batch_ansatz = batch_ansatz + 0.5 * (alpha ** 2) * d2

    return batch_ansatz

    

    