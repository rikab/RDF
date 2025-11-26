import numpy as np
import jax.numpy as jnp




# Constants
NC = 3
NF = 5

CF = (NC**2 - 1) / (2 * NC) # Quark Casimir, 4/3
CA = NC # Gluon Casimir, 3
TF = 1/2 

MZ = 91.1876 # GeV
MW = 80.379 # GeV
MH = 125.18 # GeV
MT = 173.21 # GeV

alpha_s_MZ = 0.1181


beta0 = (11 * NC - 4 * TF * NF) / 3
beta1 = (34 * NC**2 - 20 * CA * TF * NF - 12 * CF * TF * NF ) / 3


def running_coupling(alpha_0, mu0, mu):
    """
    Running coupling constant at scale mu, given the coupling at scale mu0.
    """
    return alpha_0 / (1 + beta0 * alpha_0 * jnp.log(mu / mu0) / (2 * jnp.pi))