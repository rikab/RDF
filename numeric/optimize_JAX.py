import argparse
import yaml
from tqdm import tqdm
import sys

import pickle

import jax
import jax.numpy as jnp
import optax   

import math
import numpy as np
import matplotlib.pyplot as plt

from helpers.data import get_pdf_toy, read_in_data_JAX
from helpers.ansatz import eps
from helpers.ansatz_JAX import q as q_jax



try:
    plt.style.use(
        "/global/cfs/cdirs/m3246/rikab/dimuonAD/helpers/style_full_notex.mplstyle"
    )
except:
    pass

# --------------
# ### CONFIG ###
# --------------

# Load the configuration from YAML file
def parse_args_dynamic(defaults):
    parser = argparse.ArgumentParser(description="Dynamic YAML to argparse")

    for key, value in defaults.items():
        arg_type = type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=value)

    return parser.parse_args()


with open("args.yaml", "r") as ifile:
    config = yaml.safe_load(ifile)

args = parse_args_dynamic(config)
mstar = args.mstar

outfile_name = f"{args.distribution}_{args.order_to_match}_{args.name}"

np.random.seed(args.seed)
key = jax.random.PRNGKey(args.seed)


# --------------------------------------------------
# ### Set up bins and initialize coefficients ###
# --------------------------------------------------

if args.run_toy: # define bins on-the-fly
    if args.use_logbins:
        t_bins = jnp.logspace(
            np.log10(args.t_min), np.log10(args.t_max), args.n_bins
        )
        t_bin_centers = jnp.sqrt((t_bins[1:] * t_bins[:-1]))
    else:
        t_bins = jnp.linspace(args.t_min, args.t_max, args.n_bins)
        t_bin_centers = 0.5 * (t_bins[1:] + t_bins[:-1])
        t_min = args.t_min
        t_max = args.t_max
else:
    # load in data, extract d bins
    data_dict, t_bins, t_bin_centers = read_in_data_JAX(args.distribution, args.order_to_match)
    t_min = jnp.min(t_bins)
    t_max = jnp.max(t_bins)


params = {
    "g_coeffs" : np.zeros((args.m, args.n), dtype=np.float32),
    "theta" : np.zeros((args.m, 1), dtype=np.float32),
}



if args.init_random:
    for m in range(args.m):
        for n in range(args.n):
            params["g_coeffs"][m, n] = np.random.normal(loc=0.0, scale=1.0 / (
                math.factorial(m + mstar) * math.factorial(n)
            ))

elif args.init_at_answer:
    outfile_name += "_init_at_answer"
    if args.distribution == "exponential":
        params["g_coeffs"][0, 0] = 1
    elif args.distribution == "angularity":
        params["g_coeffs"][0, 1] = 1


elif args.init_close_to_answer:
    outfile_name += "_init_close_to_answer"
    if args.distribution == "exponential":
        params["g_coeffs"][0, 0] = 0.9
    elif args.distribution == "angularity":
        params["g_coeffs"][0, 1] = 0.9


else:
    print("Must choose initialization!")
    sys.exit()


if not args.learn_theta:
    for m in range(args.m):
        for n in range(1):
            params["theta"][m, n] = -10.0 # large enough to not interfere with the sigmoid

"""
g_coeffs_to_fit.data[0,0] = -0.0438
g_coeffs_to_fit.data[0,1] = 0.8412
g_coeffs_to_fit.data[0,2] =0.0016
g_coeffs_to_fit.data[1,0] = 0.5639
g_coeffs_to_fit.data[1,1] = -0.0867
g_coeffs_to_fit.data[1,2] = -0.0433
g_coeffs_to_fit.data[2,0] = 0.1949
g_coeffs_to_fit.data[2,1] = 0.0947
g_coeffs_to_fit.data[2,2] = -0.0290

#theta_to_fit.data[0,0] = 0.4563
#theta_to_fit.data[1,0] =0.0
#theta_to_fit.data[2,0] = 0.0
"""

params["g_coeffs"] = jnp.array([[ 3.0699e-02,  7.3768e-01,  5.7048e-03],
        [ 5.1186e-03, -7.8693e-04, -3.9344e-04],
        [ 1.7691e-03,  8.5972e-04, -2.6297e-04]])


params["theta"] = jnp.array([[0.5142],
        [0.0000],
        [0.0000]])



max_M, max_N = params["g_coeffs"].shape

factorial_cache_n = jnp.array([math.factorial(k) for k in range(max_N)],
                                dtype=jnp.float32)
factorial_cache_m = jnp.array([math.factorial(k)
                                 for k in range(mstar, max_M + mstar)],
                                dtype=jnp.float32)
n_range = jnp.arange(max_N, dtype=jnp.int32)
m_range = jnp.arange(1, max_M, dtype=jnp.int32)

factorial_cache_info = factorial_cache_m, factorial_cache_n, m_range, n_range

# ---------------------------
# ### Taylor Coefficients ###
# ---------------------------



def taylor_coeffs(params, alpha0):

    # Base function as a function of alpha0
    fn = lambda a: q_jax(t_bin_centers, a, params['g_coeffs'], params['theta'], mstar, t_min, t_max, factorial_cache_info)

    base = fn(alpha0)

    if args.order_to_match >= 1:
        _, d1 = jax.jvp(fn,  (alpha0,), (1.0,)) # first derivative
    else:
        d1 = None

    if args.order_to_match == 2:
        d1_fn = lambda a: jax.jvp(fn, (a,), (1.0,))[1]
        _, d2 = jax.jvp(d1_fn, (alpha0,), (1.0,))  # second derivative
    else:
        d2 = None

    return base, d1, d2


# ---------------------------
# ###### Training Loop ######
# ---------------------------

def train_step(epoch, params, key):

    key, subkey = jax.random.split(key)


    # Load in alpha batch and PDF values
    if args.run_toy:
        loc_alphas = jax.random.exponential(subkey, shape=(args.batch_size,), dtype=jnp.float32)
        loc_alphas = loc_alphas / 0.118
        loc_alphas = jnp.clip(loc_alphas, 0, 4 * jnp.pi)

        batch_data_pdf = get_pdf_toy(loc_alphas, args.distribution, t_bin_centers, args.order_to_match, device=jax.devices()[0])  # (B, args.n_bins-1)


    else:
        n_alphas = len(data_dict.keys())
        idx = jax.random.choice(subkey, n_alphas, shape=(args.batch_size,), replace=False)
        loc_alphas_keys = [list(data_dict.keys())[i] for i in idx]
        loc_alphas = jnp.array([a for a in loc_alphas_keys], dtype=jnp.float32)
        batch_data_pdf = jnp.concatenate([data_dict[a][0] for a in loc_alphas_keys], axis=1).T
        batch_errors_pdf = jnp.concatenate([data_dict[a][1] for a in loc_alphas_keys], axis=1).T


    # Get the taylor expansion ansatz for the batch
    base, d1, d2 = taylor_coeffs(params, 0.0)

    # Construct the taylor expansion for the loc_alphas
    batch_ansatz = base
    if args.order_to_match >= 1:
        batch_ansatz = batch_ansatz + loc_alphas[:, None] * d1
    if args.order_to_match >= 2:
        batch_ansatz = batch_ansatz + 0.5 * (loc_alphas[:, None] ** 2) * d2


    # Reshape
    batch_data_pdf = batch_data_pdf.reshape(-1)
    batch_ansatz = batch_ansatz.reshape(-1)

    # Calculate the loss
    if args.ratio_loss:

        ratio = batch_data_pdf / (batch_ansatz + eps)
        loss = jnp.mean(jnp.square(jnp.log(jnp.abs(ratio) + eps))) + jnp.pi**2 * jnp.mean(ratio < 0)

    else:

        if args.weighted_mse_loss:
            rescaled_pdf = batch_data_pdf[batch_errors_pdf > 0]
            rescaled_ansatz = batch_ansatz[batch_errors_pdf > 0]
            rescaled_errors = batch_errors_pdf[batch_errors_pdf > 0]
            loss = jnp.mean(jnp.square(rescaled_pdf - rescaled_ansatz) / jnp.square(rescaled_errors))
        else:     
            loss = jnp.mean(jnp.square(batch_data_pdf - batch_ansatz))


    # Return a new key
    return loss




def train(params, epochs, batch_size, lr, key):

    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    losses = np.zeros((epochs, 1))
    lrs = np.zeros((epochs, 1))
    g_coeffs_log = np.zeros((epochs + 1, *params['g_coeffs'].shape))
    g_coeffs_log[0] = params['g_coeffs']
    theta_log = np.zeros((epochs + 1, *params['theta'].shape))
    theta_log[0] = params['theta']


    for epoch in tqdm(range(epochs)):

        key, subkey = jax.random.split(key)
        loss, grad_loss = jax.value_and_grad(lambda params: train_step(epoch, params, subkey))(params)
        updates, opt_state = opt.update(grad_loss, opt_state)
        params = optax.apply_updates(params, updates)

        losses[epoch] = loss
        lrs[epoch] = lr
        g_coeffs_log[epoch + 1] = params['g_coeffs']
        theta_log[epoch + 1] = params['theta']

    return losses, lrs, g_coeffs_log, theta_log


# Run training
losses, lrs, g_coeffs_log, theta_log = train(
    params, args.epochs, args.batch_size, args.lr, key
)

# -------------------------------------------------------------------------------
# PLOT LOSSES
# -------------------------------------------------------------------------------

fig, ax = plt.subplots(1, 4, figsize=(30, 6))

if args.ratio_loss:
    ax[0].plot(losses, label="ratio loss")
else:
    ax[0].plot(losses, label="MSE loss")
ax[0].legend()
ax[0].set_yscale("log")
ax[0].set_xlabel("Epoch")

# -------------------------------------------------------------------------------
# PLOT COEFFICIENTS
# -------------------------------------------------------------------------------
from matplotlib.pyplot import cm

color = iter(
    cm.hsv(np.linspace(0, 1, g_coeffs_log.shape[1] * g_coeffs_log.shape[2]))
)

for m in range(g_coeffs_log.shape[1]):
    for n in range(g_coeffs_log.shape[2]):
        c = next(color)
        label = f"$g_{{{m}{n}}}$"
        ax[1].plot(g_coeffs_log[:, m, n], label=label, color=c)
ax[1].legend()
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Coefficient value")


color = iter(
    cm.hsv(np.linspace(0, 1, theta_log.shape[1] * theta_log.shape[2]))
)

# -------------------------------------------------------------------------------
# PLOT THETA
# -------------------------------------------------------------------------------
for m in range(theta_log.shape[1]):
    for n in range(theta_log.shape[2]):
        c = next(color)
        label = f"theta {m} {n}"
        ax[2].plot(theta_log[:, m, n], label=label, color=c)
ax[2].legend()
ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("Theta value")



# -------------------------------------------------------------------------------
# PLOT CURVES
# -------------------------------------------------------------------------------
tt = jnp.linspace(t_min, 10, 200)
colors = ["red", "purple", "blue"]

g_coeffs_to_fit = g_coeffs_log[-1]
theta_to_fit = theta_log[-1]

for i, alpha in enumerate([0.148, 0.101, 0.049]):
    alpha_tensor = jnp.array(alpha)

    # plot ansatz
    ax[3].plot(tt,q_jax(tt, alpha_tensor, g_coeffs_to_fit, theta_to_fit, mstar, t_min, t_max, factorial_cache_info),label="Ansatz",color=colors[i],)

    if args.run_toy:
        # plot all-orders solution
        ax[3].plot(t_bin_centers.detach().cpu().numpy(),t_pdf_toy( alpha_tensor, args.distribution, t_bin_centers, -1, device).detach() .cpu().numpy(), label="Target (exact)",color=colors[i],linestyle="dashed",)
        # plot-fixed order target
        ax[3].scatter(  t_bin_centers.detach().cpu().numpy(),get_pdf_toy( alpha_tensor, args.distribution, t_bin_centers, args.order_to_match,  device ).detach().cpu()  .numpy(),label=f"Target (order $\\alpha^{args.order_to_match}$)",color=colors[i],s=0.8,)

    else:
        # plot histogram
        loc_data, loc_err = data_dict[alpha]
        ax[3].errorbar(t_bin_centers, loc_data[:,0], yerr=loc_err[:,0], label="Target (data)", color=colors[i], linestyle="dashed",)



ax[3].legend()
ax[3].set_xlabel("$t$")
ax[3].set_ylabel("Density")
# ax[2].set_ylim(-0.01, 0.4)
plt.savefig(f"plots/{outfile_name}_results.png", bbox_inches="tight")


np.save(f"output/{outfile_name}_losses", losses)
np.save(f"output/{outfile_name}_g_coeffs", g_coeffs_log)
np.save(f"output/{outfile_name}_theta", theta_log)

    
with open(f"output/{outfile_name}_config", "wb") as ofile:
    pickle.dump(config, ofile)

print("Final g")
print(g_coeffs_to_fit)
print("Final theta")

print(theta_to_fit)
