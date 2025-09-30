import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import diffrax

from utils.function_utils import polynomial, ReLU, relu_polynomial


eps = 1e-16
T_MAX = 5
N_GRID = 750

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




def f(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):


    poly = polynomial(t, alpha, g_mn, thetas_coeffs, temps_coeffs)
    g_star_poly = relu_polynomial(t, alpha, -g_star, thetas, temps, temps_relu)
    return g_star_poly * jnp.exp( - poly )



def make_integrate_f(alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):

    t_dense = jnp.linspace(0.0, T_MAX, N_GRID)
    f_vals = jax.vmap(f, in_axes=(0, None, None, None, None, None, None, None, None))(t_dense, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu)
    dt = (T_MAX - 0.0) / (N_GRID - 1)

    # cumtrapz with uniform spacing:
    F_dense = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(0.5 * (f_vals[:-1] + f_vals[1:]) * dt)] )

    # scalar to scalar integrator that interpolates F at arbitrary t
    def integrate_f(t):


        # clamp t into [0, t_max]
        t = jnp.clip(t, 0.0, T_MAX)

        # locate bin k with s[k] <= t < s[k+1]
        k = jnp.searchsorted(t_dense, t, side="right") - 1
        k = jnp.clip(k, 0, N_GRID - 2)

        t0 = t_dense[k]
        t1 = t_dense[k+1]
        f0 = f_vals[k]
        f1 = f_vals[k+1]
        F0 = F_dense[k]

        # linear interpolation of f within [t0, t1]
        w = (t - t0) / (t1 - t0 + eps)
        f_t = (1.0 - w) * f0 + w * f1

        # trapezoid on the partial cell [s0, t]:
        return F0 + 0.5 * (f0 + f_t) * (t - t0)

    return jax.jit(integrate_f)


# def integrate_f(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):



#     ts = jnp.linspace(0, t, 1000)
#     f_vals = jax.vmap(f, in_axes = (0, None, None, None, None))(ts, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu)
#     return jnp.trapz(f_vals, ts)

    try: 
        epsabs = epsrel = 1e-5

        def dI_dt(t, y, args):
            return f(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu)
        

        term = diffrax.ODETerm(dI_dt,)
        solver = diffrax.Dopri5()

        y0 = jnp.array([0.0])
        t0 = 0.0
        T = t
        saveat = diffrax.SaveAt(ts = jnp.array([T]))
        stepsize_controller = diffrax.PIDController(rtol = epsrel, atol = epsabs)
        dt0 = 0.01

        sol = diffrax.diffeqsolve(term, solver, t0 = t0, t1 = T, y0 = y0, saveat = saveat, dt0 = dt0, stepsize_controller = stepsize_controller, max_steps = 10000)

        y = sol.ys[0]
        return y
    
    except:

        def dI_dt(t, I):
            return f(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu)

        I0 = 0.0
        ts = jnp.array([0.0, t])
        Is = odeint(dI_dt, I0, ts)

        term = Is[-1]
        return term

    # y, info = quadgk(dI_dt, [0, t], epsabs=epsabs, epsrel=epsrel)

    return term


def q(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu):

    integrate_f = make_integrate_f(alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu)

    return (f(t, alpha, g_star, g_mn, thetas, thetas_coeffs, temps, temps_coeffs, temps_relu) * jnp.exp(-integrate_f(t)))





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