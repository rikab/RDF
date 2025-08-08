import jax
import jax.numpy as jnp
import jax.nn as jnn

N_integrator = 250
eps = 1e-12


def helper_theta(x, x0, temperature=50):
    # return jnp.where(x >= 0, 1.0, 0.0)
    return jnn.sigmoid(temperature * (x - x0))


def f(t, alpha, g_coeffs, theta, mstar, factorial_cache_info):

    B = t.shape[0]
    max_M, max_N = g_coeffs.shape

    factorial_cache_m, factorial_cache_n, m_range, n_range = factorial_cache_info

    t_exp = jnp.broadcast_to(t[:, None], (B, max_N))
    t_powers = (t_exp ** n_range) / factorial_cache_n  # (B, max_N)

    theta_0_exp = jnp.broadcast_to(theta[0][None, :], (B, max_N))
    g_coeffs_0_exp = jnp.broadcast_to(g_coeffs[0][None, :], (B, max_N))
    heaviside_theta_gstar = helper_theta(t_exp, theta_0_exp)

    g_star = alpha**mstar * jnp.abs(
        jnp.sum(g_coeffs_0_exp * t_powers * heaviside_theta_gstar, axis=-1)
    )

    t_exp = jnp.broadcast_to(t_exp[:, None, :], (B, max_M - 1, max_N))
    t_powers_exp = jnp.broadcast_to(t_powers[:, None, :], (B, max_M - 1, max_N))

    g_1_exp = jnp.broadcast_to(g_coeffs[1:][None, :, :], (B, max_M - 1, max_N))
    theta_1_exp = jnp.broadcast_to(theta[1:][None, :, :], (B, max_M - 1, max_N))

    factorial_cache_m_exp = jnp.broadcast_to(
        factorial_cache_m[None, :, None], (B, max_M, max_N)
    )[:, 1:, :]
    heaviside_theta_ghigher = helper_theta(t_exp, theta_1_exp)

    g_coeffs_higher = g_1_exp / factorial_cache_m_exp

    g_higher_mat = jnp.sum(
        g_coeffs_higher * t_powers_exp * heaviside_theta_ghigher, axis=-1
    )  # (B, max_M - 1)

    g_higher = jnp.sum((alpha ** (mstar + m_range)) * g_higher_mat, axis=-1)
    return g_star * jnp.exp(-g_higher)


def cumulative_trapezoidal(alpha, g_coeffs, theta, mstar, t_grid, factorial_cache_info):
    f_vals = f(t_grid, alpha, g_coeffs, theta, mstar, factorial_cache_info)
    dt = t_grid[1] - t_grid[0]
    cum = jnp.cumsum((f_vals[:-1] + f_vals[1:]) * 0.5 * dt, axis=0)
    cum = jnp.concatenate([jnp.zeros((1,), dtype=cum.dtype), cum])
    return cum


def q(t, alpha, g_coeffs, theta, mstar, t_min, t_max, factorial_cache_info):

    # added for 1d theta
    theta = jnp.broadcast_to(theta, (theta.shape[0], g_coeffs.shape[1]))

    t_dense = jnp.linspace(t_min, t_max, N_integrator)
    F_dense = cumulative_trapezoidal(alpha, g_coeffs, theta, mstar, t_dense, factorial_cache_info)

    # Interpolate
    idx = jnp.searchsorted(t_dense, jnp.clip(t, a_max=t_dense[-1]), side="right") - 1
    idx = jnp.clip(idx, 0, t_dense.shape[0] - 2)
    t0, t1 = t_dense[idx], t_dense[idx + 1]
    F0, F1 = F_dense[idx], F_dense[idx + 1]
    exp_term = F0 + (F1 - F0) * (t - t0) / (t1 - t0 + eps)
    return f(t, alpha, g_coeffs, theta, mstar, factorial_cache_info) * jnp.exp(-exp_term)
