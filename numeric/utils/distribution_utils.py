import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import diffrax

from utils.function_utils import polynomial, ReLU, relu_polynomial


eps = 1e-16
T_MAX = 20
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

def _build_integral_cache(alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):

    f_vals = jax.vmap(f, in_axes=(0, None, None, None, None, None, None, None, None))(
        t_dense, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu
    )


    # per-interval exact Hermite area (fourth order)
    # h is constant on your uniform grid
    h = (T_MAX - 0.0) / (N_GRID - 1)

    df_c = (f_vals[2:] - f_vals[:-2]) / (2.0 * h)
    df0 = (f_vals[1] - f_vals[0]) / h
    dfN = (f_vals[-1] - f_vals[-2]) / h
    df_vals = jnp.concatenate([df0[None], df_c, dfN[None]])
    df_vals = jnp.nan_to_num(df_vals, neginf=0.0, posinf=0.0)


    # integral over full interval [0,h] (u=1) with the closed forms above
    H00_1 = 0.5 - 1.0 + 1.0          # = 0.5
    H10_1 = 0.25 - 2.0/3.0 + 0.5     # = 1/12
    H01_1 = -0.5 + 1.0               # = 0.5
    H11_1 = 0.25 - 1.0/3.0           # = -1/12

    areas = h * (
        H00_1 * f_vals[:-1]
        + H10_1 * (h * df_vals[:-1])
        + H01_1 * f_vals[1:]
        + H11_1 * (h * df_vals[1:])
    )

    F_dense = jnp.concatenate([jnp.array([0.0], dtype=f_vals.dtype), jnp.cumsum(areas)])
    return t_dense, f_vals, df_vals, F_dense

@jax.jit
def _integral_from_cache(t, t_dense, f_vals, df_vals, F_dense):
    dt = (T_MAX - 0.0) / (N_GRID - 1)
    t = jnp.clip(t, 0.0, T_MAX)
    k = jnp.clip(jnp.floor(t / dt).astype(jnp.int32), 0, N_GRID - 2)

    t0 = t_dense[k]
    t1 = t_dense[k + 1]
    f0 = f_vals[k]
    f1 = f_vals[k + 1]
    df0 = df_vals[k]
    df1 = df_vals[k + 1]
    F0 = F_dense[k]

    h = t1 - t0
    u = (t - t0) / (h + eps)

    # Hermite antiderivative basis at u
    H00 = 0.5*u**4 - u**3 + u
    H10 = 0.25*u**4 - (2.0/3.0)*u**3 + 0.5*u**2
    H01 = -0.5*u**4 + u**3
    H11 = 0.25*u**4 - (1.0/3.0)*u**3

    local = h * (H00 * f0 + H10 * (h * df0) + H01 * f1 + H11 * (h * df1))
    return F0 + local

@jax.jit
def make_integrate_f(alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):
    t_dense, f_vals, df_vals, F_dense = _build_integral_cache(
        alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu
    )
    @jax.jit
    def integrate_f(t):
        return _integral_from_cache(t, t_dense, f_vals, df_vals, F_dense)
    return integrate_f

@jax.jit
def q(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):
    t_dense, f_vals, df_vals, F_dense = _build_integral_cache(
        alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu
    )
    integral = _integral_from_cache(t, t_dense, f_vals, df_vals, F_dense)
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