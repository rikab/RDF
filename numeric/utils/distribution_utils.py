import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import diffrax

from utils.function_utils import polynomial, ReLU, relu_polynomial


eps = 1e-16
T_MAX = 10
N_GRID = 1000

TINY = 1e-30
MAX_EXP = 60.0

# Clipped exp to avoid overflow
def _exp_clip(x):
    return jnp.exp(jnp.clip(x, -MAX_EXP, MAX_EXP))



# Precompute dense t grid 
t_dense = jnp.linspace(0.0, T_MAX, N_GRID, dtype=jnp.float32)
dt = (T_MAX - 0.0) / (N_GRID - 1)

def construct_cdf(function, t_func):

    def cdf(x, alpha, params):
        t = t_func(x)
        return jnp.nan_to_num(jnp.exp(-function(t, alpha, params)))
    return cdf



def construct_pdf(function, t_func):

    cdf = construct_cdf(function, t_func)
    derivative = jax.grad(cdf, argnums=0)

    def pdf(x, alpha, params):
        return jnp.nan_to_num(derivative(x, alpha, params) )

    return pdf



@jax.jit
def f(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):


    poly = polynomial(t, alpha, g_mn, thetas_coeffs, temps_coeffs)
    g_star_poly = relu_polynomial(t, alpha, -g_star, thetas, temps, temps_relu)
    return  g_star_poly * _exp_clip( - poly )


# _df_dt = jax.grad(f, argnums=0)

@jax.jit
def _build_integral_cache(alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):
    f_vals = jax.vmap(
        f, in_axes=(0, None, None, None, None, None, None, None, None)
    )(t_dense, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu)

    dt = (T_MAX - 0.0) / (N_GRID - 1)

    def body(i, F):
        # i==1: fall back to trapezoid (no i-2 available)
        F1 = F.at[1].set(0.5 * dt * (f_vals[0] + f_vals[1]))

        # For i >= 2:
        F_even = F.at[i].set(F[i-2] + (dt/3.0)*(f_vals[i-2] + 4.0*f_vals[i-1] + f_vals[i]))
        delta_odd = (dt/12.0) * (5.0*f_vals[i] + 8.0*f_vals[i-1] - f_vals[i-2])
        F_odd  = F.at[i].set(F[i-1] + delta_odd)

        F_out = jax.lax.cond(i == 1, lambda _: F1,
                 lambda _: jax.lax.cond((i & 1) == 0, lambda __: F_even, lambda __: F_odd, None),
                 operand=None)
        return F_out

    F_dense = jnp.zeros((N_GRID,), dtype=f_vals.dtype)
    F_dense = jax.lax.fori_loop(1, N_GRID, body, F_dense)
    return t_dense, f_vals, F_dense

@jax.jit
def _integral_from_cache(t, t_dense, f_vals, F_dense):
    dt = (T_MAX - 0.0) / (N_GRID - 1)
    t = jnp.clip(t, 0.0, T_MAX)
    k  = jnp.floor(t / dt).astype(jnp.int32)
    k  = jnp.clip(k, 0, N_GRID - 2)

    t0 = t_dense[k]
    w  = (t - t0) / (dt + eps)

    # neighbors
    fm2 = f_vals[jnp.clip(k-2, 0, N_GRID-1)]
    fm1 = f_vals[jnp.clip(k-1, 0, N_GRID-1)]
    f0  = f_vals[k]
    f1  = f_vals[k+1]
    fp2 = f_vals[jnp.clip(k+2, 0, N_GRID-1)]

    # three stencils
    partial_center = w*(f0) + w*(f1 - fm1)/4.0 + (w*w)*( -2.0*f0 + f1 + fm1 )/6.0
    partial_left   = w*(f0) + w*( -3.0*f0 + 4.0*f1 - fp2 )/4.0 + (w*w)*( f0 - 2.0*f1 + fp2 )/6.0
    partial_right  = w*(f0) + w*( fm1 - fm2 )/4.0 + (w*w)*( -2.0*f0 + fm1 + fm2 )/6.0

    use_left  = (k == 0)
    use_right = (k >= N_GRID - 3)
    partial = jax.lax.select(use_left,  partial_left,
              jax.lax.select(use_right, partial_right, partial_center))

    return F_dense[k] + dt * partial

@jax.jit
def make_integrate_f(alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):
    t_dense, f_vals, F_dense = _build_integral_cache(
        alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu
    )
    @jax.jit
    def integrate_f(t):
        return _integral_from_cache(t, t_dense, f_vals, F_dense)
    return integrate_f

@jax.jit
def q(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):
    t_dense, f_vals, F_dense = _build_integral_cache(
        alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu
    )
    integral = _integral_from_cache(t, t_dense, f_vals, F_dense)
    return f(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu) * jnp.exp(-integral)


def build_q_mstar(mstar):

    def q_mstar(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):

        M, N = g_star.shape

        new_M = M + mstar

        g_star_0s = jnp.zeros((mstar, N))
        new_g_star = jnp.vstack((g_star_0s, g_star))

        g_mn_0s =  jnp.zeros((mstar, N))
        new_g_mn = jnp.vstack((g_mn_0s, g_mn))

        theta_0s = jnp.zeros((mstar))
        new_thetas = jnp.concatenate((theta_0s, thetas))

        theta_c_0s = jnp.zeros((mstar))
        new_thetas_c = jnp.concatenate((theta_c_0s, thetas_coeffs))

        temp_0s = jnp.zeros(mstar)
        new_temps = jnp.concatenate((temp_0s, temps))

        temp_c_0s = jnp.zeros(mstar)
        new_temps_c = jnp.concatenate((temp_c_0s, temps_coeffs))

        temp_r_0s = jnp.zeros(mstar)
        new_temps_r = jnp.concatenate((temp_r_0s, temps_relu))

        return q(t, alpha, new_g_star, new_g_mn, new_thetas, new_thetas_c, new_temps, new_temps_c, new_temps_r)
    
    return q_mstar






# def log_q(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):

#     # f part
#     poly = polynomial(t, alpha, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu)
#     g_star_poly = polynomial(t, alpha, g_star, thetas)
#     term1 = jnp.log( -1 * g_star_poly) - poly

#     # integral part
#     integral = integrate_f(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu)
#     term2 = -1 * integral

#     return term1 + term2    