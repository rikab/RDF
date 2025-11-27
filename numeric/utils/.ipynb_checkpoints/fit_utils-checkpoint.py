import jax
import jax.numpy as jnp
from functools import partial



@partial(jax.jit, static_argnames=['m0', 'mstar'])
def randomize_params(params, scale=0.1, key=None, m0 = 1, mstar=1):


    k1,k2,k3,k4,k5,k6,k7 = jax.random.split(key, 7)

    shape_m, shape_n = params["g_star"].shape

    
    g_star = params["g_star"].at[-m0:, :].add(scale  * jax.random.normal(k1, (m0, shape_n)))
    g_coeffs = params["g_coeffs"].at[-m0-mstar:-mstar, :].add(scale  * jax.random.normal(k2, (m0, shape_n)))

    thetas = params["thetas"].at[-m0:].add(scale * jax.random.normal(k3, (m0,)))
    thetas = params["thetas"].at[-m0:].set(jnp.minimum(thetas[-1], thetas[-2]))
    thetas = params["thetas"].at[-m0:].set(jnp.maximum(thetas[-m0:], -1))
    thetas_coeffs = params["thetas_coeffs"].at[-m0-mstar:-mstar].add(scale * jax.random.normal(k4, (m0,)))
    thetas_coeffs = params["thetas_coeffs"].at[-m0-mstar:-mstar].set(jnp.minimum(thetas_coeffs[-1-mstar], thetas_coeffs[-1-mstar-1]))
    thetas_coeffs = params["thetas_coeffs"].at[-m0-mstar:-mstar].set(jnp.maximum(thetas_coeffs[-m0-mstar:-mstar], -1))

    temps = (params["temps"].at[-m0:].add(scale * jax.random.normal(k5, (m0,))))
    temps_coeffs = (params["temps_coeffs"].at[-m0-mstar:-mstar].add(scale * jax.random.normal(k6, (m0,))))
    temps_positive = (params["temps_positive"].at[-m0:].add(scale  * jax.random.normal(k7, (m0,))))


    return {
      **params, "g_star": g_star, "g_coeffs": g_coeffs, "thetas": thetas,
      "thetas_coeffs": thetas_coeffs, "temps": temps,
      "temps_coeffs": temps_coeffs, "temps_positive": temps_positive
    }



# Function to freeze lower orders so that only the highest order is trained

# @jax.jit
def freeze_lower_orders(params, original_params, m0 = 1, mstar = 1):
    # copy refs
    g_star, g_coeffs = params["g_star"], params["g_coeffs"]
    thetas, thetas_coeffs = params["thetas"], params["thetas_coeffs"]
    temps, temps_coeffs = params["temps"], params["temps_coeffs"]
    temps_positive = params["temps_positive"]

    # indices

    m_end = -m0
    c_end = -m0 - mstar

    # stop grad through all orders <= (m-1) by copying originals and stopping grad
    g_star = g_star.at[:m_end].set(jax.lax.stop_gradient(original_params["g_star"][:m_end]))
    g_coeffs = g_coeffs.at[:c_end].set(jax.lax.stop_gradient(original_params["g_coeffs"][:c_end]))
    thetas = thetas.at[:m_end].set(jax.lax.stop_gradient(original_params["thetas"][:m_end]))
    thetas_coeffs = thetas_coeffs.at[:c_end].set(jax.lax.stop_gradient(original_params["thetas_coeffs"][:c_end]))
    temps = temps.at[:m_end].set(jax.lax.stop_gradient(original_params["temps"][:m_end]))
    temps_coeffs = temps_coeffs.at[:c_end].set(jax.lax.stop_gradient(original_params["temps_coeffs"][:c_end]))
    temps_positive = temps_positive.at[:m_end].set(jax.lax.stop_gradient(original_params["temps_positive"][:m_end]))


    # # Turn off theta gradients
    # thetas = thetas.at[-m0:].set(jax.lax.stop_gradient(thetas[-m0:]))
    # thetas_coeffs = thetas_coeffs.at[-m0-mstar:-mstar].set(jax.lax.stop_gradient(thetas_coeffs[-m0-mstar:-mstar]))

    out = dict(params)
    out["g_star"] = g_star
    out["g_coeffs"] = g_coeffs
    out["thetas"] = thetas
    out["thetas_coeffs"] = thetas_coeffs
    out["temps"] = temps
    out["temps_coeffs"] = temps_coeffs
    out["temps_positive"] = temps_positive
    
    return out


# Mask parameters:

def create_mask(params, m0=1, mstar=1):

    # Initialize mask
    mask = jax.tree_map(lambda x: jnp.ones_like(x, dtype = bool), params)

    # Set stuff to False
    rows = params["g_coeffs"].shape[0]
    mask["g_star"] = mask["g_star"].at[:-m0].set(False)
    mask["thetas"] = mask["thetas"].at[:-m0].set(False)
    mask["temps"] = mask["temps"].at[:-m0].set(False)
    mask["temps_positive"] = mask["temps_positive"].at[:-m0].set(False)

    mask["g_coeffs"] = mask["g_coeffs"].at[:-m0-mstar].set(False)
    mask["thetas_coeffs"] = mask["thetas_coeffs"].at[:-m0-mstar].set(False)
    mask["temps_coeffs"] = mask["temps_coeffs"].at[:-m0-mstar].set(False)

    mask["g_coeffs"] = mask["g_coeffs"].at[rows-mstar:].set(False)
    mask["thetas_coeffs"] = mask["thetas_coeffs"].at[rows-mstar:].set(False)
    mask["temps_coeffs"] = mask["temps_coeffs"].at[rows-mstar:].set(False)

    # # Turn off thetas learning
    # mask["thetas"] = mask["thetas"].at[-m0:].set(False)
    # mask["thetas_coeffs"] = mask["thetas_coeffs"].at[:-m0-mstar:].set(False)

    return mask
