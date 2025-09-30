#!/usr/bin/env python
# coding: utf-8

# In[8]:


import argparse
import yaml
from tqdm import tqdm
import sys
import copy

import pickle

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax   

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


from helpers.data import get_pdf_toy_JAX, read_in_data_JAX


# # Initialization and Setup

# In[ ]:


# TODO: Replace this with Radha's YAML files
mstar = 1
m = 1
n = 7
starting_nuisance_n = 3
dist = "thrust"
tag = "g_star"

mult_factor = 2 if dist == "thrust" else 1


gaussian_prior_param = 10 # std of g/m!n!
alpha_init = 0.118
zero_error_scale = 1 # Fraction of minimum error to set error on 0, default is 1
lr = 0.00100
weight_decay = 0.001
epochs = 50000
batch_size = 320*1
seed = 42
freeze_previous_order = False   # NEW
bogo_init = True
random_batch = True
ignore_left = False # Ignore the leftmost point 

name = f"{dist}_m{m}_n{n}"
init_from = f"{dist}_m{m-1}_n{n}"

# Seed stuff
np.random.seed(seed)
jax_key = jax.random.PRNGKey(seed)

order_colors = {1 : "blue", 2 : "purple"}

# Colors
colors = ["red", "violet", "blue"][::-1]
darkcolors = ["darkred", "darkviolet", "darkblue"][::-1]


# Initialize params
params = {
    "alpha" : alpha_init,
    "g_star" : -jnp.zeros((m+1 - mstar, n+1)),
    "g_coeffs" : -jnp.zeros((m+1 - mstar, n+1)),
    "thetas" : jnp.zeros((m+1 - mstar,)) ,
    "thetas_coeffs" : jnp.zeros((m+1 - mstar,)) ,
    "temps" : 1 * jnp.ones((m+1 - mstar,)),
    "temps_coeffs" : 1 * jnp.ones((m+1 - mstar,)),
    "temps_positive" : 0.1 * jnp.ones((m+1 - mstar,)),
}

beta_limits = [1e-3, 25]
alpha_limits = [0.1, 0.14]


factorials = np.ones_like(params["g_star"])
for mi in range(params["g_star"].shape[0]):
    for ni in range(params["g_star"].shape[1]):
        factorials[mi, ni] = 1 / math.factorial(mi + mstar) / math.factorial(ni)




# In[10]:


# Init from previous file:

def initialize_params(name_, order):

    # Initialize params
    params_ = {
        "alpha" : alpha_init,
        "g_star" : -np.zeros((order+2 - mstar, n+1)),
        "g_coeffs" : -np.zeros((order+2 - mstar, n+1)),
        "thetas" : np.zeros((order+2 - mstar,)) ,
        "thetas_coeffs" : np.zeros((order+2 - mstar,)) ,
        "temps" : 1 * np.ones((order+2 - mstar,)),
        "temps_coeffs" : 1 * np.ones((order+2 - mstar,)),
        "temps_positive" : 0.1 * np.ones((order+2 - mstar,)),
    }

    try:
        with open(f"output_JAX/{name_}_params.pkl", "rb") as f:
            init_params = copy.deepcopy(pickle.load(f))

        g_coeffs_init = init_params["g_coeffs"]
        g_star_init = init_params["g_star"]
        thetas_init = init_params["thetas"]
        thetas_coeffs_init = init_params["thetas_coeffs"]
        temps_init = init_params["temps"]
        temps_coeffs_init = init_params["temps_coeffs"]
        temps_pos_init = init_params["temps_positive"]


        init_m, init_n = g_coeffs_init.shape

        params_["g_coeffs"][:init_m, :init_n] = copy.deepcopy(g_coeffs_init)
        params_["g_star"][:init_m, :init_n] = copy.deepcopy(g_star_init)
        params_["thetas"][:init_m] = copy.deepcopy(thetas_init)
        params_["thetas_coeffs"][:init_m] = copy.deepcopy(thetas_coeffs_init)
        params_["temps"][:init_m] = copy.deepcopy(temps_init) 
        params_["temps_coeffs"][:init_m] = copy.deepcopy(temps_coeffs_init)
        params_["temps_positive"][:init_m] = copy.deepcopy(temps_pos_init)

        for k in params_.keys():
            params_[k] = jnp.array(params_[k])

        return copy.deepcopy(params_)
    
    except:
        print(f"No file" + f"output_JAX/{name_}_params.pkl")
        


params1 = initialize_params(f"{dist}_m1_n{n}", 1)
params2 = initialize_params(f"{dist}_m2_n{n}", 2)
params_array = [params1, params2]

# if init_from:
#     params = init_params(init_from)

# # Save the original params for the purpose of freezing
# original_params = params.copy()


# In[11]:


def randomize_params(params, scale = 1/10):


    new_params = copy.deepcopy(params)
    shape_m, shape_n = new_params["g_star"].shape

    for k in new_params.keys():
        new_params[k] = np.array(new_params[k])

    new_params["g_star"][-1,:] = np.random.normal(loc = new_params["g_star"][-1,:], size = shape_n, scale = scale) 
    new_params["g_coeffs"][-1-mstar,:] = np.random.normal(loc = new_params["g_coeffs"][-1-mstar,:], size = shape_n, scale =  scale )  

    new_params["thetas"][-1] = np.random.normal(loc = new_params["thetas"][-1], scale =  scale)
    new_params["thetas_coeffs"][-1-mstar] = np.random.normal(loc = new_params["thetas_coeffs"][-1-mstar], scale =   scale)

    new_params["temps"][-1] = np.abs(np.random.normal(loc = new_params["temps"][-1], scale = scale))
    new_params["temps_coeffs"][-1-mstar] = np.abs(np.random.normal(loc = new_params["temps_coeffs"][-1-mstar], scale = scale)) 
    new_params["temps_positive"][-1] = np.abs(np.random.normal(loc = new_params["temps_positive"][-1], scale = scale))

    for k in new_params.keys():
        new_params[k] = jnp.array(new_params[k])

    return new_params



# # Compilation

# In[12]:


from utils.function_utils import polynomial, taylor_expand_in_alpha
from utils.distribution_utils import build_q_mstar, log_q, f
from rikabplotlib.plot_utils import newplot

q = build_q_mstar(mstar)

# compile
q_vmap = jax.vmap(q, in_axes=(0,None,None,None,None,None,None,None,None))


# Taylor Expansions
q0_vmap = jax.vmap(taylor_expand_in_alpha(q, 0), in_axes=(0,None,None,None,None,None,None,None,None))
q1_vmap = jax.vmap(taylor_expand_in_alpha(q, 1), in_axes=(0,None,None,None,None,None,None,None,None))
q2_vmap = jax.vmap(taylor_expand_in_alpha(q, 2), in_axes=(0,None,None,None,None,None,None,None,None))
q3_vmap = jax.vmap(taylor_expand_in_alpha(q, 3), in_axes=(0,None,None,None,None,None,None,None,None))
# ... Add more if necessary, can loop if we really have to

qm_vmap = [q0_vmap, q1_vmap, q2_vmap, q3_vmap]

# ##### Second vmap over alpha #####
q_vmap2 = jax.vmap(q_vmap, in_axes = (None, 0, None, None, None, None, None, None, None))
qm_vmap2 = []
for qi in qm_vmap:
    qi_vmap2 = jax.vmap(qi, in_axes = (None, 0, None, None, None, None, None, None, None))
    qm_vmap2.append(qi_vmap2)



# Select the Taylor Expansion Function
CHOSEN_Q_VMAP = qm_vmap[m]
CHOSEN_Q_VMAP2 = qm_vmap2[m]


# Helper functions
def Q_ANSATZ(ts, alpha, params):


    t_reg = 1.0
    ns = np.arange(0, n+1)
    t_pows = np.power(t_reg, ns)
    m_pow = params["g_star"].shape[0]-1

    normed_g_star =  params["g_star"]
    normed_g_coeffs = params["g_coeffs"]
    normed_g_star = normed_g_star.at[-1].set(normed_g_star[-1]/alpha**m_pow * t_pows)
    normed_g_coeffs = normed_g_coeffs.at[-1-mstar,].set(normed_g_coeffs[-1-mstar] / alpha**m_pow * t_pows)

    return q_vmap(ts, alpha, normed_g_star, normed_g_coeffs, params["thetas"], params["thetas_coeffs"], params["temps"], params["temps_coeffs"], params["temps_positive"])


def Q_ANSATZ_alpha(ts, params):
    
    t_reg = 1.0
    ns = np.arange(0, n+1)
    t_pows = np.power(t_reg, ns)
    m_pow = params["g_star"].shape[0]-1



    normed_g_star =  params["g_star"]
    normed_g_coeffs = params["g_coeffs"]
    normed_g_star = normed_g_star.at[-1].set(normed_g_star[-1]/params["alpha"]**m_pow * t_pows)
    normed_g_coeffs = normed_g_coeffs.at[-1-mstar,].set(normed_g_coeffs[-1-mstar] / params["alpha"]**m_pow * t_pows)


    return q_vmap(ts, params["alpha"], normed_g_star, normed_g_coeffs, params["thetas"], params["thetas_coeffs"], params["temps"], params["temps_coeffs"], params["temps_positive"])



# # Data Setup
# 
# Opal Data from https://www.hepdata.net/record/ins440721?version=1&table=Table%203
# 
# ALEPH Data from https://www.hepdata.net/record/ins636645?version=1&table=Table%2054

# In[13]:


LEP = {
    "THRUST": 1-np.array([
        0.585,0.595,0.605,0.615,0.625,0.635,0.645,0.655,0.665,0.675,
        0.685,0.695,0.705,0.715,0.725,0.735,0.745,0.755,0.765,0.775,
        0.785,0.795,0.805,0.815,0.825,0.835,0.845,0.855,0.865,0.875,
        0.885,0.895,0.905,0.915,0.925,0.935,0.945,0.955,0.965,0.975,
        0.985,0.995
    ]),
    "THRUST_HIGH": 1-np.array([
        0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,
        0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,
        0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,
        0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,
        0.98,0.99
    ]),
    "THRUST_LOW": 1-np.array([
        0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,
        0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,
        0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,
        0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,
        0.99,1.0
    ]),
    "DSIG": np.array([
        0.00115,0.00229,0.0035,0.00839,0.01283,0.02148,0.0433,0.06556,0.10284,0.12575,
        0.16396,0.18416,0.21424,0.24814,0.29427,0.34347,0.36536,0.43088,0.46923,0.54584,
        0.59942,0.66087,0.768,0.857,0.98079,1.1127,1.27035,1.43939,1.63938,1.93459,
        2.23002,2.63072,3.06214,3.70212,4.54709,5.64464,7.27722,9.62952,13.60396,18.56511,
        12.88646,1.31186
    ]),
    "stat_plus": np.array([
        0.00034,0.0005,0.00061,0.00096,0.00116,0.00152,0.00223,0.00267,0.00334,0.00366,
        0.00418,0.00449,0.00466,0.00511,0.00567,0.00609,0.00614,0.00691,0.00709,0.0079,
        0.00814,0.00856,0.00928,0.00974,0.0106,0.01125,0.01209,0.01339,0.01372,0.01504,
        0.01641,0.01811,0.01892,0.02073,0.02343,0.02568,0.02929,0.03385,0.04085,0.04923,
        0.03872,0.00986
    ]),
    "stat_minus": np.array([
        -0.00034,-0.0005,-0.00061,-0.00096,-0.00116,-0.00152,-0.00223,-0.00267,-0.00334,-0.00366,
        -0.00418,-0.00449,-0.00466,-0.00511,-0.00567,-0.00609,-0.00614,-0.00691,-0.00709,-0.0079,
        -0.00814,-0.00856,-0.00928,-0.00974,-0.0106,-0.01125,-0.01209,-0.01339,-0.01372,-0.01504,
        -0.01641,-0.01811,-0.01892,-0.02073,-0.02343,-0.02568,-0.02929,-0.03385,-0.04085,-0.04923,
        -0.03872,-0.00986
    ]),
    "sys1_plus": np.array([
        0.00052,0.00093,0.00087,0.00077,0.00082,0.00095,0.00496,0.00273,0.00353,0.00297,
        0.00474,0.00871,0.00455,0.01328,0.00573,0.00801,0.00707,0.00868,0.01309,0.00687,
        0.00809,0.02434,0.0152,0.01211,0.01281,0.00939,0.00874,0.00965,0.02383,0.02559,
        0.0291,0.01555,0.03499,0.0235,0.03555,0.03708,0.0454,0.07026,0.07561,0.08476,
        0.22928,0.19542
    ]),
    "sys1_minus": np.array([
        -0.00052,-0.00093,-0.00087,-0.00077,-0.00082,-0.00095,-0.00496,-0.00273,-0.00353,-0.00297,
        -0.00474,-0.00871,-0.00455,-0.01328,-0.00573,-0.00801,-0.00707,-0.00868,-0.01309,-0.00687,
        -0.00809,-0.02434,-0.0152,-0.01211,-0.01281,-0.00939,-0.00874,-0.00965,-0.02383,-0.02559,
        -0.0291,-0.01555,-0.03499,-0.0235,-0.03555,-0.03708,-0.0454,-0.07026,-0.07561,-0.08476,
        -0.22928,-0.19542
    ]),
    "sys2_plus": np.array([
        0.00033,0.00065,0.001,0.0017,0.0026,0.00436,0.00307,0.00464,0.00728,0.00984,
        0.01283,0.01441,0.01503,0.01741,0.02065,0.01874,0.01993,0.0235,0.01728,0.0201,
        0.02208,0.02607,0.0303,0.03381,0.04685,0.05316,0.06069,0.052,0.05922,0.06989,
        0.06846,0.08076,0.09401,0.07685,0.09439,0.11718,0.11117,0.1471,0.20781,0.68299,
        0.47408,0.04826
    ]),
    "sys2_minus": np.array([
        -0.00033,-0.00065,-0.001,-0.0017,-0.0026,-0.00436,-0.00307,-0.00464,-0.00728,-0.00984,
        -0.01283,-0.01441,-0.01503,-0.01741,-0.02065,-0.01874,-0.01993,-0.0235,-0.01728,-0.0201,
        -0.02208,-0.02607,-0.0303,-0.03381,-0.04685,-0.05316,-0.06069,-0.052,-0.05922,-0.06989,
        -0.06846,-0.08076,-0.09401,-0.07685,-0.09439,-0.11718,-0.11117,-0.1471,-0.20781,-0.68299,
        -0.47408,-0.04826
    ]),
}


OPAL = {
    "THRUST": 1-np.array([0.74,0.815,0.865,0.895,0.92,0.94,0.955,0.965,0.975,0.985,0.995]),
    "THRUST_HIGH": 1-np.array([0.7,0.78,0.85,0.88,0.91,0.93,0.95,0.96,0.97,0.98,0.99]),
    "THRUST_LOW": 1-np.array([0.78,0.85,0.88,0.91,0.93,0.95,0.96,0.97,0.98,0.99,1.0]),
    "DSIG": np.array([0.14,0.93,1.02,3.1,3.56,3.92,6.5,10.1,18.1,22.1,8.6]),
    "stat_plus": np.array([0.12,0.24,0.39,0.64,0.82,0.84,1.5,1.9,2.5,2.9,1.7]),
    "stat_minus": np.array([-0.12,-0.24,-0.39,-0.64,-0.82,-0.84,-1.5,-1.9,-2.5,-2.9,-1.7]),
    "sys_plus": np.array([0.47,0.47,0.61,0.59,0.81,1.29,1.4,1.6,2.6,2.6,2.6]),
    "sys_minus": np.array([-0.47,-0.47,-0.61,-0.59,-0.81,-1.29,-1.4,-1.6,-2.6,-2.6,-2.6]),
}


def add_total_error_columns(d,):

    running_err2 = d["stat_plus"]**2
    for key in d.keys():
        if "sys" in key and "plus" in key:
            running_err2 += d[key]**2

    d["total_err"] = np.sqrt(running_err2)

    return d

LEP = add_total_error_columns(LEP)
OPAL = add_total_error_columns(OPAL)

print(LEP["DSIG"])
print(LEP["total_err"])


# In[14]:


if dist == "thrust":

    xOPAL_bin_centers = OPAL["THRUST"]
    xLEP_bin_centers = LEP["THRUST"]
    tOPAL_bin_centers = np.log(1/(2 * xOPAL_bin_centers))
    tLEP_bin_centers = np.log(1/(2 * xLEP_bin_centers))
    
    T_MAX = 6
    min_value = np.exp(-T_MAX ) / 2


    xLEP_low = LEP["THRUST_LOW"]
    xLEP_high = LEP["THRUST_HIGH"]

 

    xLEP_low = np.clip(xLEP_low, min_value, None)
    xLEP_high = np.clip(xLEP_high, min_value, None)

    tLEP_low = np.log(1/(2 * xLEP_low))
    tLEP_high = np.log(1/(2 * xLEP_high))


    xlow =  OPAL["THRUST_LOW"]
    xhigh = OPAL["THRUST_HIGH"]
    xOPAL_bin_widths = (xhigh-xlow) 

    xlow =  LEP["THRUST_LOW"]
    xhigh = LEP["THRUST_HIGH"]
    xLEP_bin_widths = (xhigh-xlow) 

    yOPALs = OPAL["DSIG"]
    yLEPs = LEP["DSIG"]

    yOPAL_errs = OPAL["total_err"]
    yLEP_errs = LEP["total_err"]

    if ignore_left:
        yOPAL_errs[-1] *= 100
        yLEP_errs[-1] *= 100



    x_bin_centers = np.sort(np.concatenate([xOPAL_bin_centers, xLEP_bin_centers]))
    t_bin_centers =(np.log(1/(2 * x_bin_centers)))
    ys = np.concatenate([yOPALs, yLEPs])
    y_errs = np.concatenate([yOPAL_errs, yLEP_errs])



# Plot data


fig, ax = newplot("full")
plt.errorbar(xOPAL_bin_centers, yOPALs, yerr=yOPAL_errs, xerr = xOPAL_bin_widths, fmt='o', color='black', lw=1, capsize=2, label = "OPAL Data")
plt.errorbar(xLEP_bin_centers, yLEPs, yerr=yLEP_errs, xerr = xLEP_bin_widths, fmt='o', color='grey', lw=1, capsize=2, label = "LEP Data")



q_vals1 = Q_ANSATZ(t_bin_centers, alpha_init, params1) / x_bin_centers #* mult_factor
q_vals2 = Q_ANSATZ(t_bin_centers, alpha_init, params2) / x_bin_centers #* mult_factor




plt.plot(x_bin_centers, q_vals1, color = "blue", label = r"RDF $\mathcal{O}(\alpha_s^1)$")
plt.plot(x_bin_centers, q_vals2, color = "purple", label = r"RDF $\mathcal{O}(\alpha_s^2)$")



for i in range(100):
    q_vals1 = Q_ANSATZ(t_bin_centers, alpha_init, randomize_params(params1, scale = 1/10)) / x_bin_centers #* mult_factor

    plt.plot(x_bin_centers, q_vals1, color = "blue", alpha = 0.02)


for i in range(100):
    q_vals2 = Q_ANSATZ(t_bin_centers, alpha_init, randomize_params(params2, scale = 1/10)) / x_bin_centers #* mult_factor
    plt.plot(x_bin_centers, q_vals2, color = "purple", alpha = 0.02)


# plt.plot(x_bin_centers, q_taylor_vals)

print(np.trapz(Q_ANSATZ(t_bin_centers, alpha_init, params2), t_bin_centers))

plt.ylim(1e-5, 1e3)

plt.legend(ncol = 2)

plt.yscale("log")





# # Loss

# In[15]:


# Weighted MSE
@jax.jit
def loss_function(params, alpha, ys, yerrs):


    # y_preds_low = jnp.nan_to_num(Q_ANSATZ(tLEP_low, alpha, params)) / xLEP_low
    y_preds_center = jnp.nan_to_num(Q_ANSATZ(tLEP_bin_centers, alpha, params)) / xLEP_bin_centers
    # y_preds_high = jnp.nan_to_num(Q_ANSATZ(tLEP_high, alpha, params)) / xLEP_high

    # # Use Simpson's Rule
    # bin_width = xLEP_high - xLEP_low
    # y_preds = 1 / 6 * (y_preds_low + 4 * y_preds_center  + y_preds_high)

    y_preds = y_preds_center #bin_width / 6 * (y_preds_low + 4 * y_preds_center  + y_preds_high)

    y_errs_rescaled = yerrs #/ mean_errors

    return jnp.sum(jnp.nan_to_num((y_preds - ys)**2 /(y_errs_rescaled**2 ))) / 2
    

@jax.jit
def loss_function_alpha(params, ys, yerrs):



    y_preds = jnp.nan_to_num(Q_ANSATZ_alpha(tLEP_bin_centers, params)) / xLEP_bin_centers #* mult_factor
    y_errs_rescaled = yerrs #/ mean_errors
    loss = jnp.sum(jnp.nan_to_num((y_preds - ys)**2 /(y_errs_rescaled**2 ))) / 2



    g_coeffs = params["g_coeffs"]
    g_star = params["g_star"]

    loss += jnp.nan_to_num(jnp.sum((g_coeffs[-2] * factorials[-1])**2 + (g_star[-1] * factorials[-1])**2) /2 / gaussian_prior_param**2 )



    return loss



print("Initial Loss: ", loss_function(params2, alpha_init, yLEPs, yLEP_errs))


# # Scan over alphas

# In[16]:


alphas = np.linspace(0.00, 0.5, 2000)

l1s = []
l2s = []


for a in alphas:

    l1s.append(loss_function(params1, a, yLEPs, yLEP_errs))
    l2s.append(loss_function(params2, a, yLEPs, yLEP_errs))

l1s = np.array(l1s)
l2s = np.array(l2s)

fig, ax = newplot("full")
plt.plot(alphas, 2 * (l1s - np.min(l1s)), color = "blue")
plt.plot(alphas, 2 * (l2s - np.min(l2s)), color = "purple")
plt.xlabel(r"$\alpha_s$")
plt.ylabel(r"-2$L$")

plt.yscale("log")


fig, ax = newplot("full")
plt.errorbar(xOPAL_bin_centers, yOPALs, yerr=yOPAL_errs, xerr = xOPAL_bin_widths, fmt='o', color='black', lw=1, capsize=2, label = "OPAL Data")
plt.errorbar(xLEP_bin_centers, yLEPs, yerr=yLEP_errs, xerr = xLEP_bin_widths, fmt='o', color='grey', lw=1, capsize=2, label = "LEP Data")
best_alpha = alphas[np.argmin(l2s)]
print(best_alpha)


q_vals2 = Q_ANSATZ(tLEP_bin_centers, alpha_init, params2) / xLEP_bin_centers #* mult_factor
q_vals2best = Q_ANSATZ(tLEP_bin_centers, best_alpha, params2) / xLEP_bin_centers #* mult_factor

diffs = ((yLEPs - q_vals2) / yLEP_errs) **2
diffsbest = ((yLEPs - q_vals2best) / yLEP_errs)**2

plt.plot(xLEP_bin_centers, q_vals2, color = "purple", label = r"RDF $\mathcal{O}(\alpha_s^2)$")
plt.plot(xLEP_bin_centers, q_vals2best, color = "purple", label = r"RDF $\mathcal{O}(\alpha_s^2)$", ls = "--")

plt.yscale("log")

fig, ax = newplot("full")
plt.plot(xLEP_bin_centers, diffs, color = "purple")
plt.plot(xLEP_bin_centers, diffsbest, color = "purple", ls = "--")


# 

# In[ ]:


@jax.jit
def projector(params, original_params):


    # Get the signs right
    g_star = params["g_star"]
    g_coeffs = params["g_coeffs"]
    thetas = params["thetas"]
    thetas_coeffs = params["thetas_coeffs"]
    temps = params["temps"]
    temps_coeffs = params["temps_coeffs"]
    temps_pos = params["temps_positive"]


    # g_star = g_star.at[:, -1].set(-jnp.abs(g_star[:, -1]))
    # g_coeffs = g_coeffs.at[:, -1].set(-jnp.abs(g_coeffs[:, -1]))

    # Clip beta
    temps = jnp.clip(temps, beta_limits[0], beta_limits[1])
    temps_coeffs = jnp.clip(temps_coeffs, beta_limits[0], beta_limits[1])
    temps_pos = jnp.clip(temps_pos, beta_limits[0], beta_limits[1])

    # Clip alpha
    params["alpha"] = jnp.clip(params["alpha"], alpha_limits[0], alpha_limits[1])

    # temps = beta_limits[0] + jnp.ones_like(temps) * epoch * (beta_limits[1] - beta_limits[0])
    # temps_coeffs = beta_limits[0] + jnp.ones_like(temps_coeffs) * epoch * (beta_limits[1] - beta_limits[0])

    g_coeffs = g_coeffs.at[g_coeffs.shape[0]-mstar:,:].set(0)  # Current order doesn't count
    thetas_coeffs = thetas_coeffs.at[g_coeffs.shape[0]-mstar:,].set(-1)  # Current order doesn't count


    # # For the nuisance params, only allow higher order t's
    # g_star = g_star.at[-1, :starting_nuisance_n].set(0)
    # g_coeffs = g_coeffs.at[-1-mstar, :starting_nuisance_n].set(0)

    # For the nuisance params, only allow higher order t's
    # g_star = g_star.at[-1, :].set(0)
    g_coeffs = g_coeffs.at[-1-mstar, :].set(0)



    # Restore the original parameters 
    g_coeffs_init = original_params["g_coeffs"]
    g_star_init = original_params["g_star"]
    thetas_init = original_params["thetas"]
    thetas_coeffs_init = original_params["thetas_coeffs"]
    temps_init = original_params["temps"]
    temps_coeffs_init = original_params["temps_coeffs"]
    temps_pos_init = original_params["temps_positive"]


    g_star = g_star.at[:-1].set(g_star_init[:-1])
    g_coeffs = g_coeffs.at[:-1-mstar].set(g_coeffs_init[:-1-mstar])
    thetas = thetas.at[:-1].set(thetas_init[:-1])
    thetas_coeffs = thetas_coeffs.at[:-1-mstar].set(thetas_coeffs_init[:-1-mstar])
    temps = temps.at[:-1].set(temps_init[:-1])
    temps_coeffs = temps_coeffs.at[:-1-mstar].set(temps_coeffs_init[:-1-mstar])
    temps_pos = temps_pos.at[:-1].set(temps_pos_init[:-1])



    params["g_star"] = g_star
    params["g_coeffs"] = g_coeffs
    params["thetas"] = thetas
    params["thetas_coeffs"] = thetas_coeffs
    params["temps"] = temps
    params["temps_coeffs"] = temps_coeffs
    params["temps_positive"] = temps_pos

    return params


# In[18]:


# Loop

def train(alpha, init_params, epochs, lr, jax_key, verbose = True, verbose_epochs = 1000, bogo_init = True, early_stopping = 1000, bogo_scale = 1, bogo_epochs = 5000):


    params = copy.deepcopy(init_params)
    original_params = copy.deepcopy(init_params)


    best_params = copy.deepcopy(init_params)
    best_loss = loss_function(init_params, alpha, yLEPs, yLEP_errs)

    if bogo_init:

        counter = 0
        losses = []
        ps = []
        for i in range(bogo_epochs):

            s = (bogo_epochs  - i) / bogo_epochs
            p = projector(randomize_params(params, scale= bogo_scale * s / (np.sqrt(counter)**2 + 1)), best_params)
            ps.append(p)
            loss = loss_function(ps[-1], alpha, yLEPs, yLEP_errs)
            losses.append(loss)

            if loss < best_loss:
                best_params = copy.deepcopy(p)
                best_loss = loss
                counter += 1

            if verbose:
                print(f"Bogo Epoch {i}, best loss {best_loss : .3e}, current loss {loss : .3e}, counter {counter}")
            


        min_arg = np.argmin(losses)
        params = ps[min_arg]
        print(losses[min_arg])


    # Initialize Optimizer
    opt = optax.adamw(lr , weight_decay= weight_decay)
    opt_state = opt.init(params)

    # Initialize logs
    losses = []
    params_log = []
    params_log.append(params)
    if verbose:
        g_coeffs_log = [params["g_coeffs"]]
        g_star_log = [params["g_star"]]
        thetas_log = [params["thetas"]]
        temps_log = [params["temps"]]
        thetas_c_log = [params["thetas_coeffs"]]
        temps_c_log = [params["temps_coeffs"]]
        temps_p_log = [params["temps_positive"]]


    @jax.jit
    def train_step(epoch, params, opt_state, random_key):
        
        # Boilerplate, in case we need random numbers
        key, subkey = jax.random.split(random_key)


        y_batch = yLEPs
        yerrs_batch = yLEP_errs

        # Get the gradients
        loss, grad_loss = jax.value_and_grad(loss_function)(params, alpha, y_batch, yerrs_batch)
        # nanflag = False
        # for k in grad_loss.keys():

        #     # print(grad_loss[k].dtype)
        #     if jnp.any(jnp.isnan(grad_loss[k])):
        #         nanflag = True
        
        #         print(f"nan detected at epoch {epoch}")
        #         break
        # if nanflag:
        #     params["g_star"] = params["g_star"].at[-1].set(params["g_star"][-1]  * (1 + jax.random.normal(key, shape = params["g_star"][-1].shape) * 1e-1))
        #     params["g_coeffs"] = params["g_coeffs"].at[-1-mstar].set(params["g_star"][-1]  * (1 +jax.random.normal(key, shape = params["g_coeffs"][-1].shape )* 1e-1))
                    

        #     print("old loss = " + str(loss.item()))
            
        #     loss, grad_loss = jax.value_and_grad(loss_function)(params, alpha, y_batch, yerrs_batch)
        #     print("new loss = " + str(loss.item()))

        # #         print(f"{k} has nans at {epoch}")
        #     grad_loss[k] = jnp.nan_to_num(grad_loss[k], nan = 1e-3 * params[k] *  jax.random.normal(key) )

        # Jax Grad Descent stuff
        updates, opt_state = opt.update(grad_loss, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        

        # Save the random key for the seed for the next epoch
        return params, opt_state, loss, subkey

        
    epochs_tqdm = tqdm(range(epochs))
    for epoch in epochs_tqdm:



        if (epoch+1) % verbose_epochs == 0 and verbose:
            fig, ax = newplot("full")
            plt.errorbar(xOPAL_bin_centers, yOPALs, yerr=yOPAL_errs, xerr = xOPAL_bin_widths, fmt='o', color='black', lw=1, capsize=2, label = "OPAL Data")
            plt.errorbar(xLEP_bin_centers, yLEPs, yerr=yLEP_errs, xerr = xLEP_bin_widths, fmt='o', color='grey', lw=1, capsize=2, label = "LEP Data")

            q_vals = Q_ANSATZ_alpha(tLEP_bin_centers, params) / xLEP_bin_centers #* mult_factor


            # q_vals1 = Q_ANSATZ(t_bin_centers, alpha_init, params1) / x_bin_centers * mult_factor
            # q_vals2 = Q_ANSATZ(t_bin_centers, alpha_init, params2) / x_bin_centers * mult_factor

            order_strings = {1 : r"$\mathcal{O}(\alpha_s^{1})$", 2 : r"$\mathcal{O}(\alpha_s^{2})$"}
            plt.plot(xLEP_bin_centers, q_vals, color = order_colors[m], label = r"RDF " + order_strings[m])
            # plt.plot(x_bin_centers, q_vals2, color = "purple", label = r"RDF $\mathcal{O}(\alpha_s^2)$")



            for i in range(50):
                q_vals1 = Q_ANSATZ(tLEP_bin_centers, alpha_init, projector(randomize_params(params, scale = bogo_scale), original_params)) / xLEP_bin_centers 
                plt.plot(xLEP_bin_centers, q_vals1, color = order_colors[m], alpha = 0.1)

            a = alpha #params["alpha"]
            plt.ylim(1e-5, 1e3)
            plt.legend(title = f"Epoch {epoch}, a_s = { a : .4f}")




        #     # plt.xlim(0, 1)
            plt.yscale("log")
            plt.show()


            fig, ax = newplot("full", width = 8 * 5, subplot_array=(1, 5))

            losses_ = np.array(losses)
            g_star_log_ = np.array(g_star_log)
            g_coeffs_log_ = np.array(g_coeffs_log)
            thetas_log_ = np.array(thetas_log)
            temps_log_ = np.array(temps_log)
            thetas_c_log_ = np.array(thetas_c_log)
            temps_c_log_ = np.array(temps_c_log)
            temps_p_log_ = np.array(temps_p_log)


            color = iter(
                cm.hsv(np.linspace(0, 1, g_coeffs_log_.shape[1] * g_coeffs_log_.shape[2]))
            )


            ax[0].plot(losses_)
            ax[0].set_yscale("log")

            for mi in range(g_star_log_.shape[1]):


                for ni in range(g_star_log_.shape[2]):

                    c = next(color)

                    ax[1].plot(g_star_log_[:,mi,ni], color = c, label = f"g_star_{mi}{ni}")
                    ax[2].plot(g_coeffs_log_[:,mi,ni], color = c, label = f"g_coeffs_{mi}{ni}")


                ax[3].plot(thetas_log_[:,mi], label = f"theta_{mi}", color = c)
                ax[4].plot(temps_log_[:,mi], label = f"beta_{mi}", color = c)

                ax[3].plot(thetas_c_log_[:,mi], label = f"theta_coeffs_{mi}", color = c, ls = "--")
                ax[4].plot(temps_c_log_[:,mi], label = f"beta_coeffs_{mi}", color = c, ls = "--")

                ax[4].plot(temps_p_log_[:,mi], label = f"beta_positive_{mi}", ls = ":", color = c)

            ax[0].set_ylabel("loss")
            ax[0].set_xlabel("epochs")

            ax[1].legend()
            ax[1].set_ylabel("g_star")
            ax[1].set_xlabel("epochs")

            ax[2].legend()
            ax[2].set_ylabel("g_coeffs")
            ax[2].set_xlabel("epochs")

            ax[3].legend()
            ax[3].set_ylabel("theta")
            ax[3].set_xlabel("epochs")


            ax[4].legend()
            ax[4].set_ylabel("beta")
            ax[4].set_xlabel("epochs")
            plt.yscale("log")
            plt.show()

        # Train Step
        params, opt_state, loss, jax_key = train_step(epoch, params, opt_state, jax_key)


        # Project the params to make them physical
        params = projector(params, original_params)
        # print(params)

        losses.append(float(loss))
        params_log.append(jax.device_get(params))

        if verbose:
                g_star_log.append(params['g_star'])
                g_coeffs_log.append(params['g_coeffs'])
                thetas_log.append(params['thetas'])
                temps_log.append(params["temps"])
                thetas_c_log.append(params["thetas_coeffs"])
                temps_c_log.append(params["temps_coeffs"])
                temps_p_log.append(params["temps_positive"])


        epochs_tqdm.set_description(f"{losses[-1] : .4e}")
        # print(g_coeffs_log[-1])
        # print(g_star_log[-1])
        # print(thetas_log[-1])


        # Early Stopping
        best_loss = np.min(losses)
        arg_best_loss = np.argmin(losses)

        if epoch - arg_best_loss > early_stopping:
            break



    return np.min(losses), params_log[np.argmin(losses)], losses, params_log


# In[ ]:







# In[ ]:





# In[ ]:


mis = [1, 2]

for mi in mis:

    loss, new_params, losses, params_log = train(0.09, params_array[mi-1], 50000, 10*lr, jax_key, verbose=True, verbose_epochs=5000, bogo_init=True, early_stopping=50000, bogo_scale=1)


    alphas = np.linspace(0.09, 0.15, 81)

    ls = []
    ps = []
    as_ = []
    params = copy.deepcopy(new_params)

    for alpha in alphas:


        # Check which set of params to use: the previous alpha, or the original init
        loss_original = loss_function(new_params, alpha, yLEPs, yLEP_errs)
        loss_prev = loss_function(params, alpha, yLEPs, yLEP_errs)

        print(alpha, loss_original, loss_prev)

        if loss_prev < loss_original:
            params = params
        else:
            params = copy.deepcopy(new_params)


        loss, params, losses, params_log = train(alpha, params, 25000, lr, jax_key, verbose=False, verbose_epochs=2500, bogo_init=True, early_stopping=5000, bogo_scale=1/10)
        ls.append(loss)
        ps.append(params)
        as_.append(alpha)

        fig, ax = newplot("column")
        quad = np.polyfit(np.array(as_), np.array(ls), deg = 2)

        min_ys = np.min(2*ls)
        plt.errorbar(as_, 2*np.array(ls)- min_ys, color = order_colors[mi], fmt = "x")

        argmin = np.argmin(ls)
        best_alpha = as_[argmin]

        plt.axhline(1)
        plt.axvline(best_alpha)
        plt.show()



    with open(f"output_JAX/alphas_{mi}_{n}_{tag}.pkl", "wb") as f:
        pickle.dump(as_, f)

    with open(f"output_JAX/ls_{mi}_{n}_{tag}.pkl", "wb") as f:
        pickle.dump(ls, f)    


# In[23]:


ls = np.array(ls)
as_ = np.array(as_)

min_ys = 0*np.min(ls)
plt.plot(as_, 2*(ls) - 2*min_ys, color = order_colors[m])
argmin = np.argmin(ls)
best_alpha = as_[argmin]
low_alpha = as_[np.argmin(np.abs(2*ls - 2*min_ys-1))]
# high_alpha = as_[len(as_) - np.argmin(np.abs(2*ls[::-1]-2*min_ys-1))]

# plt.axhline(1 )#+ 2*min_ys)
plt.axvline(best_alpha, color = order_colors[m], ls = "--")
plt.axvline(low_alpha, color = order_colors[m], ls = "--", alpha = 0.25)
# plt.axvline(high_alpha, color = order_colors[m], ls = "--", alpha = 0.25)

print(low_alpha, best_alpha)

# plt.ylim(0, 10)


# In[ ]:





# In[27]:


fig, ax = newplot("full")

for mi in range(1, 3):

        print(mi)

    # try:

        with open(f"output_JAX/alphas_{mi}_{n}_{tag}.pkl", "rb") as f:
            alphas = np.array(pickle.load(f))

        with open(f"output_JAX/ls_{mi}_{n}_{tag}.pkl", "rb") as f:
            ls = np.array(pickle.load(f))


        ls2 = 2 * ls

        min_ls = 0 #np.min(ls2)
        ls2 = ls2 - min_ls
        min_alpha = alphas[np.argmin(ls2)]

        alphas_below = alphas[ls2 < 1 + np.min(ls2)]
        low_alpha = alphas_below[0]
        high_alpha = alphas_below[-1]

        plt.plot(alphas, ls2, color = order_colors[mi])

        plt.axvline(min_alpha, color = order_colors[mi], ls = "--")
        plt.axvline(low_alpha, color = order_colors[mi], ls = "--", alpha = 0.25)
        plt.axvline(high_alpha, color = order_colors[mi], ls = "--", alpha = 0.25)

        plt.axvline(0.118, color = "black")
    # except:
    #     pass


# In[ ]:


.7 * (len(tLEP_bin_centers))


# for (i, l)

# 

# 

# 

# 

# 

# 
