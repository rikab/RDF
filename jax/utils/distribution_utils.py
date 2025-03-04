import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint


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




def f(t, alpha, g_star, g_mn):


    poly = polynomial(t, alpha, g_mn)
    g_star_poly = polynomial(t, alpha, g_star)
    return -1 * g_star_poly * jnp.exp( - poly)



@jax.jit
def integrate_f(t, alpha, g_star, g_mn):

    # g_mn_alpha = collapse_in_alpha(alpha, g_mn)

    # # Get the Gaussian (quadratic) part
    # g0 = g_mn_alpha[0]
    # g1 = g_mn_alpha[1]
    # g2 = g_mn_alpha[2]

    # prefactor = jnp.exp(-)

    def dI_dt(I, t):
        return f(t, alpha, g_star, g_mn)
    
    I0 = 0.0
    ts = jnp.array([0.0, t])
    Is = odeint(dI_dt, I0, ts)

    return Is[-1]



def q(t, alpha, g_star, g_mn):

    return f(t, alpha, g_star, g_mn) * jnp.exp(-integrate_f(t, alpha, g_star, g_mn))
