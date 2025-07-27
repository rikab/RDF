import torch

N_integrator = 250
eps = 1e-12


def helper_theta(x, x0, temperature=20):
    #return torch.where(x >= 0, 1.0, 0.0)
    return torch.sigmoid(temperature * (x - x0))


def f(t, alpha, g_coeffs, theta, mstar, factorial_cache_info):

    B = t.shape[0]
    max_M, max_N = g_coeffs.shape

    factorial_cache_m, factorial_cache_n, m_range, n_range = factorial_cache_info


    t_exp = t.unsqueeze(1).expand(B, max_N)
    n_range_exp = n_range.unsqueeze(0).expand(B, max_N)
    factorial_cache_n_exp = factorial_cache_n.unsqueeze(0).expand(B, max_N)
    t_powers = (t_exp**n_range) / factorial_cache_n  # (B, max_N)
    theta_0_exp = theta[0].unsqueeze(0).expand(B, max_N)
    g_coeffs_0_exp = g_coeffs[0].unsqueeze(0).expand(B, max_N)
    heaviside_theta_gstar = helper_theta(t_exp, theta_0_exp)

    g_star = alpha**mstar * torch.abs(
        torch.sum(g_coeffs_0_exp * t_powers * heaviside_theta_gstar, dim=-1)
    )

    t_exp = t_exp.unsqueeze(1).expand(B, max_M - 1, max_N)
    t_powers_exp = t_powers.unsqueeze(1).expand(B, max_M - 1, max_N)

    g_1_exp = g_coeffs[1:].unsqueeze(0).expand(B, max_M - 1, max_N)
    theta_1_exp = theta[1:].unsqueeze(0).expand(B, max_M - 1, max_N)

    factorial_cache_m_exp = (
        factorial_cache_m.unsqueeze(0)
        .expand(B, max_M)
        .unsqueeze(2)
        .expand(B, max_M, max_N)[:, 1:, :]
    )
    heaviside_theta_ghigher = helper_theta(t_exp, theta_1_exp)

    g_coeffs_higher = g_1_exp / factorial_cache_m_exp

    g_higher_mat = torch.sum(
        g_coeffs_higher * t_powers_exp * heaviside_theta_ghigher, dim=-1
    )  # (N_integrator, max_M - 1, max_N) -> (N_integrator, max_M - 1)

    g_higher = torch.sum((alpha ** (mstar + m_range)) * g_higher_mat, dim=-1)
    return g_star * torch.exp(-g_higher)


def cumulative_trapezoidal(alpha, g_coeffs, theta, mstar, t_grid, device, factorial_cache_info):
    f_vals = f(t_grid, alpha, g_coeffs, theta, mstar, factorial_cache_info)
    dt = t_grid[1] - t_grid[0]
    cum = torch.cumsum((f_vals[:-1] + f_vals[1:]) * 0.5 * dt, dim=0)
    cum = torch.cat([torch.zeros(1, device=device), cum])
    return cum


def q(t, alpha, g_coeffs, theta, mstar, t_min, t_max, device, factorial_cache_info):

    # added for 1d theta
    theta = theta.expand(-1, g_coeffs.shape[1])

    t_dense = torch.linspace(
        t_min, t_max, N_integrator, device=device
    )
    F_dense = cumulative_trapezoidal(alpha, g_coeffs, theta, mstar, t_dense, device, factorial_cache_info)

    # Interpolate
    idx = torch.searchsorted(t_dense, t.clamp(max=t_dense[-1]), right=True) - 1
    idx = idx.clamp(min=0, max=t_dense.shape[0] - 2)
    t0, t1 = t_dense[idx], t_dense[idx + 1]
    F0, F1 = F_dense[idx], F_dense[idx + 1]
    exp_term = F0 + (F1 - F0) * (t - t0) / (t1 - t0 + eps)
    return f(t, alpha, g_coeffs, theta, mstar, factorial_cache_info) * torch.exp(-exp_term)


