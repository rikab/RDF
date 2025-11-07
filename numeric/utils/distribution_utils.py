import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import diffrax

from utils.function_utils import polynomial, ReLU, relu_polynomial


eps = 1e-16
T_MAX = 5
N_GRID = 500

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

# Helper function to evaluate integral from precomputed cache
@jax.jit
def _build_integral_cache(alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):


    f_vals = jax.vmap(f, in_axes=(0, None, None, None, None, None, None, None, None))(t_dense, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu)


    # trapezoid with uniform spacing
    F_dense = jnp.concatenate([jnp.array([0.0], dtype=f_vals.dtype), jnp.cumsum(0.5 * (f_vals[:-1] + f_vals[1:]) * dt)])
    
    return t_dense, f_vals, F_dense


@jax.jit
def _integral_from_cache(t, t_dense, f_vals, F_dense):

    dt = (T_MAX - 0.0) / (N_GRID - 1)
    t = jnp.clip(t, 0.0, T_MAX)
    k  = jnp.floor(t / dt).astype(jnp.int32)
    k  = jnp.clip(k, 0, N_GRID - 2)

    t0 = t_dense[k]
    t1 = t_dense[k + 1]
    f0 = f_vals[k]
    f1 = f_vals[k + 1]
    F0 = F_dense[k]

    w   = (t - t0) / (t1 - t0 + eps)
    f_t = (1.0 - w) * f0 + w * f1

    return F0 + 0.5 * (f0 + f_t) * (t - t0)



@jax.jit
def make_integrate_f(alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):

    # Doesn't depend on t so wont get vmapped
    t_dense, f_vals, F_dense = _build_integral_cache(alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu)

    @jax.jit
    def integrate_f(t):

        return _integral_from_cache(t, t_dense, f_vals, F_dense)

    return integrate_f



@jax.jit
def q(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):
    
    # Doesn't depend on t so wont get vmapped
    t_dense, f_vals, F_dense = _build_integral_cache(alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu)
    
    
    integral = _integral_from_cache(t, t_dense, f_vals, F_dense)

    # log_f = jnp.log( jnp.clip(f(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu), a_min=TINY) )

    return f(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu)  * jnp.exp(-integral)





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





def log_q(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):

     # f part
     poly = polynomial(t, alpha, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu)
     g_star_poly = polynomial(t, alpha, g_star, thetas)
     term1 = jnp.log( -1 * g_star_poly) - poly

     # integral part
     integral = integrate_f(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu)
     term2 = -1 * integral

     return term1 + term2    