
import argparse
import yaml
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.func import jacrev
import pickle

import os

from helpers.data import get_pdf_toy, read_in_data
from helpers.ansatz import q, eps, get_taylor_expanded_ansatz
from helpers.training import get_loss, train

# Set PyTorch default dtype to float64
torch.set_default_dtype(torch.double)

# plt.style.use(
#     "/global/cfs/cdirs/m3246/rikab/dimuonAD/helpers/style_full_notex.mplstyle"
# )



parser_yaml = argparse.ArgumentParser()
parser_yaml.add_argument("--config", type=str, required=True)
# Get known args and keep the rest to parse later
yargs, remaining_argv = parser_yaml.parse_known_args()

with open(f"configs/{yargs.config}.yaml", "r") as ifile:
    config = yaml.safe_load(ifile)
print(f"Loaded config: {yargs.config}.yaml")

def parse_args_dynamic(defaults, argv):
    parser = argparse.ArgumentParser(description="Dynamic YAML to argparse")
    for key, value in defaults.items():
        arg_type = type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=value)
    return parser.parse_args(argv)

args = parse_args_dynamic(config, remaining_argv)

mstar = args.mstar

outfile_name = f"{args.distribution}_{args.order_to_match}_{args.name}"


device = args.device
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.run_toy: # define bins on-the-fly
    if args.use_logbins:
        t_bins = torch.logspace(
            np.log10(args.t_min), np.log10(args.t_max), args.n_bins, device=device
        )
        t_bin_centers = torch.sqrt((t_bins[1:] * t_bins[:-1]))
    else:
        t_bins = torch.linspace(args.t_min, args.t_max, args.n_bins, device=device)
        t_bin_centers = 0.5 * (t_bins[1:] + t_bins[:-1])
        t_min = args.t_min
        t_max = args.t_max
    data_dict = None
else:
    # load in data, extract d bins
    data_dict, t_bins, t_bin_centers = read_in_data(args.distribution, args.order_to_match, device)
    t_min = torch.min(t_bin_centers)
    t_max = torch.max(t_bin_centers)
        


# ########## Deal with 0-errors ##########

for a in data_dict.keys():

    y_data, y_err = data_dict[a]

    # Minimum y_err
    min_y_err = torch.min(y_err[y_err > 0])
    y_err = torch.clamp(y_err, min=min_y_err.item())

    data_dict[a] = (y_data, y_err)





"""
elif args.init_at_answer:
    outfile_name += "_init_at_answer"
    if args.distribution == "exponential":
        g_coeffs_to_fit.data[0, 0] = 1
    elif args.distribution == "angularity":
        g_coeffs_to_fit.data[0, 1] = 1


elif args.init_close_to_answer:
    outfile_name += "_init_close_to_answer"
    if args.distribution == "exponential":
        g_coeffs_to_fit.data[0, 0] = 0.9
    elif args.distribution == "angularity":
        g_coeffs_to_fit.data[0, 1] = 0.9


else:
    print("Must choose initialization!")
    sys.exit()
"""


g_coeffs_to_fit = torch.nn.Parameter(
    torch.zeros((args.m, args.n), device=device)
)
#theta_to_fit = torch.nn.Parameter(torch.zeros((args.m, args.n), device=device))
theta_to_fit = torch.nn.Parameter(torch.zeros((args.m, 1), device=device))


if args.init_random:
    for m in range(args.m):
        for n in range(args.n):
            g_coeffs_to_fit.data[m, n] = np.random.normal(loc=0.0, scale=1.0 / (
                math.factorial(m + mstar) * math.factorial(n)
            ))

if args.init_g_matrix_path != "none":
    g_coeffs_init = np.load(f"output/{args.init_g_matrix_path}_g_coeffs.npy")[-1]
    for m in range(g_coeffs_init.shape[0]):
        for n in range(g_coeffs_init.shape[1]):
            g_coeffs_to_fit.data[m, n] = g_coeffs_init[m,n]
    print(f"Initializing g to {g_coeffs_to_fit} from {args.init_g_matrix_path}")
    
if args.init_theta_path != "none":
    theta_init = np.load(f"output/{args.init_theta_path}_theta.npy")[-1]
    for m in range(theta_init.shape[0]):
        for n in range(1):
            theta_to_fit.data[m, n] = theta_init[m,n]
    print(f"Initializing theta to {theta_to_fit} from {args.init_theta_path}")

          
if args.reroll_initialization:
    
    # Reroll the initialization a bunch of times to get a better starting point
    counter = 1
    for i in range(1000):
        
        loss = get_loss(args.order_to_match, args.distribution, args.batch_size, 
                         g_coeffs_to_fit, theta_to_fit, mstar, t_bin_centers, device,
                        args.run_toy, args.weighted_mse_loss, data_dict)
    
        if i == 0:
            best_loss = loss.item()
            best_g_coeffs =  g_coeffs_to_fit.detach().cpu().numpy()
            best_theta =  theta_to_fit.detach().cpu().numpy()
        else:
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_g_coeffs = g_coeffs_to_fit.detach().cpu().numpy().copy()
                best_theta = theta_to_fit.detach().cpu().numpy().copy() 
                counter += 1
    
        # reinitialize the parameters
        for m in range(args.m):
            for n in range(args.n):
                g_coeffs_to_fit.data[m, n] = torch.tensor(np.random.normal(0, scale= 1.0  / (
                     math.factorial(m + mstar) * math.factorial(n) * 1
                )), device=device, dtype=torch.float32, requires_grad=True)
    
                # force even n to be negative, odd n to be positive
                if n % 2 == 0:
                    g_coeffs_to_fit.data[m, n] = -torch.abs(g_coeffs_to_fit.data[m, n])
                else:
                    g_coeffs_to_fit.data[m, n] = torch.abs(g_coeffs_to_fit.data[m, n])
    
        theta_to_fit.data = torch.tensor(np.random.uniform(0.0, 0.5, size=(args.m, 1)), device=device, dtype=torch.float32, requires_grad=True)
    
        # Print the loss, best loss, and best coefficients
        print(f"Iteration {i+1}: Loss = {loss.item():.6f}, Best Loss = {best_loss:.6f}, Best Theta = {best_theta}, current Theta = {theta_to_fit.detach().cpu().numpy()}, counter = {counter}")
    
    g_coeffs_to_fit.data = torch.tensor(best_g_coeffs, device=device, dtype=torch.float64, requires_grad=True)
    theta_to_fit.data = torch.tensor(best_theta, device=device, dtype=torch.float64, requires_grad=True)

    print(f"Initializing g to {g_coeffs_to_fit} from reroll")
    print(f"Initializing theta to {theta_to_fit} from reroll")

if not args.learn_theta:
    for m in range(args.m):
        for n in range(1):
            theta_to_fit.data[m, n] = -10.0 # large enough to not interfere with the sigmoid
              


g_coeffs_to_fit = g_coeffs_to_fit.double()
theta_to_fit = theta_to_fit.double()    


# Run training
losses, lrs, g_coeffs_log, theta_log = train(g_coeffs_to_fit, theta_to_fit,
    args.order_to_match, args.distribution, mstar, t_bin_centers, data_dict, args.epochs, args.batch_size, args.lr, args.weight_decay, args.learn_theta, args.run_toy, args.weighted_mse_loss, device
)

# -------------------------------------------------------------------------------
# PLOT LOSSES
# -------------------------------------------------------------------------------

fig, ax = plt.subplots(1, 4, figsize=(30, 6))

#if args.ratio_loss:
#    ax[0].plot(losses, label="ratio loss")
#else:
ax[0].plot(losses, label="MSE loss")
ax[0].legend()
ax[0].set_yscale("log")
ax[0].set_xlabel("Epoch")

# -------------------------------------------------------------------------------
# PLOT COEFFICIENTS
# -------------------------------------------------------------------------------
from matplotlib.pyplot import cm

color = iter(
    cm.hsv(np.linspace(0, 1, 2*g_coeffs_log.shape[1] * g_coeffs_log.shape[2]))
)

for m in range(g_coeffs_log.shape[1]):
    for n in range(g_coeffs_log.shape[2]):
        c = next(color)
        label = f"$g_{{{m}{n}}}$"
        ax[1].plot(g_coeffs_log[:, m, n], label=label, color=c)
ax[1].legend()
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Coefficient value")



# -------------------------------------------------------------------------------
# PLOT THETA
# -------------------------------------------------------------------------------
color = iter(
    cm.hsv(np.linspace(0, 1, 2*theta_log.shape[1] * theta_log.shape[2]))
)


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
tt = torch.linspace(t_min, 10, 200, device=device)
colors = ["red", "purple", "blue"]


for i, alpha in enumerate([0.148, 0.101, 0.049]):
    alpha_tensor = torch.tensor(alpha, device=device)

    # plot ansatz
    ax[3].plot(tt.detach().cpu().numpy(),q(tt, alpha_tensor, g_coeffs_to_fit, theta_to_fit, mstar, device).detach().cpu().numpy(),label="Ansatz",color=colors[i],)


      # plot ansatz derivative
    alpha_zero = torch.tensor(1e-12, device=device, requires_grad=True)
    fn = lambda a: q(
        t_bin_centers, a,  torch.tensor(g_coeffs_log[-1], device=device),  torch.tensor(theta_log[-1], device=device), mstar, device
    )
    batch_ansatz = get_taylor_expanded_ansatz(fn, alpha_zero, alpha_tensor, args.order_to_match)


    ax[3].plot(t_bin_centers.detach().cpu().numpy(),batch_ansatz.detach().cpu().numpy(),label=f"Ansatz, order {args.order_to_match}",color=colors[i],linestyle="dotted")

    if args.run_toy:
        # plot all-orders solution
        ax[3].plot(t_bin_centers.detach().cpu().numpy(),get_pdf_toy( alpha_tensor, args.distribution, t_bin_centers, -1, device).detach() .cpu().numpy(), label="Target (exact)",color=colors[i],linestyle="dashed",)
        # plot-fixed order target
        ax[3].scatter(  t_bin_centers.detach().cpu().numpy(),get_pdf_toy( alpha_tensor, args.distribution, t_bin_centers, args.order_to_match,  device ).detach().cpu()  .numpy(),label=f"Target (order $\\alpha^{args.order_to_match}$)",color=colors[i],s=0.8,)

    else:
        # plot histogram
        loc_data, loc_err = data_dict[alpha]
        ax[3].errorbar(t_bin_centers.detach().cpu().numpy(), loc_data.detach().cpu().numpy().reshape(-1,), yerr = loc_err.detach().cpu().numpy().reshape(-1,),  label="Target (data)",  color=colors[i],linestyle="dashed", alpha = 0.25)



ax[3].legend()
ax[3].set_xlabel("$t$")
ax[3].set_ylabel("Density")
ax[3].set_ylim(-2, 2)
plt.savefig(f"plots/{outfile_name}_results.png", bbox_inches="tight")


np.save(f"output/{outfile_name}_losses", losses)
np.save(f"output/{outfile_name}_g_coeffs", g_coeffs_log)
np.save(f"output/{outfile_name}_theta", theta_log)

    
with open(f"output/{outfile_name}_config", "wb") as ofile:
    pickle.dump(config, ofile)

torch.set_printoptions(precision=16, sci_mode=False)

print("Final g")
print(g_coeffs_to_fit)
print("Final theta")

print(theta_to_fit)
