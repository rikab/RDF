import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import diffrax

from utils.function_utils import polynomial


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




def f(t, alpha, g_star, g_mn, thetas):


    poly = polynomial(t, alpha, g_mn, thetas)
    g_star_poly = polynomial(t, alpha, g_star, thetas)
    return -1 * g_star_poly * jnp.exp( - poly)



@jax.jit
def integrate_f(t, alpha, g_star, g_mn, thetas):

    try: 
        epsabs = epsrel = 1e-5

        def dI_dt(t, y, args):
            return f(t, alpha, g_star, g_mn, thetas)
        

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
            return f(t, alpha, g_star, g_mn, thetas)

        I0 = 0.0
        ts = jnp.array([0.0, t])
        Is = odeint(dI_dt, I0, ts)

        term = Is[-1]
        return term

    # y, info = quadgk(dI_dt, [0, t], epsabs=epsabs, epsrel=epsrel)

    return term



def q(t, alpha, g_star, g_mn, thetas):

    return f(t, alpha, g_star, g_mn, thetas) * jnp.exp(-integrate_f(t, alpha, g_star, g_mn, thetas))


def log_q(t, alpha, g_star, g_mn, thetas):

    # f part
    poly = polynomial(t, alpha, g_mn, thetas)
    g_star_poly = polynomial(t, alpha, g_star, thetas)
    term1 = jnp.log( -1 * g_star_poly) - poly

    # integral part
    integral = integrate_f(t, alpha, g_star, g_mn, thetas)
    term2 = -1 * integral

    return term1 + term2    
