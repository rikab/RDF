import argparse
import yaml
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.func import jacrev
import pickle

from helpers.data import get_pdf_toy, read_in_data
from helpers.ansatz import q, eps

from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

# Set PyTorch default dtype to float64
torch.set_default_dtype(torch.float32)

plt.style.use(
    "/global/cfs/cdirs/m3246/rikab/dimuonAD/helpers/style_full_notex.mplstyle"
)


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


device = "cuda"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
else:
    # load in data, extract d bins
    data_dict, t_bins, t_bin_centers = read_in_data(args.distribution, args.order_to_match, device)
    t_min = torch.min(t_bins)
    t_max = torch.max(t_bins)
        

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


if not args.learn_theta:
    for m in range(args.m):
        for n in range(1):
            theta_to_fit.data[m, n] = -10.0 # large enough to not interfere with the sigmoid

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
g_coeffs_to_fit = g_coeffs_to_fit.float()
theta_to_fit = theta_to_fit.float()
          
max_M, max_N = g_coeffs_to_fit.shape
factorial_cache_n = torch.tensor(
    [math.factorial(k) for k in range(max_N)], device=device
).float()
factorial_cache_m = torch.tensor(
    [math.factorial(k) for k in range(mstar, max_M + mstar)], device=device
).float()
n_range = torch.arange(max_N, device=device)
m_range = torch.arange(1, max_M, device=device)

factorial_cache_info = factorial_cache_m, factorial_cache_n, m_range, n_range



def train(epochs, batch_size, lr):

    if args.learn_theta:
        optimizer = torch.optim.AdamW([g_coeffs_to_fit, theta_to_fit], lr=lr)
    else:
        optimizer = torch.optim.AdamW([g_coeffs_to_fit], lr=lr)

    scheduler = ExponentialLR(optimizer, gamma=1)

    MSE_criterion = torch.nn.MSELoss()

    losses = np.zeros((epochs, 1))
    lrs = np.zeros((epochs, 1))
    g_coeffs_log = np.zeros((epochs + 1, *g_coeffs_to_fit.shape))
    g_coeffs_log[0] = g_coeffs_to_fit.detach().cpu().numpy()

    theta_log = np.zeros((epochs + 1, *theta_to_fit.shape))
    theta_log[0] = theta_to_fit.detach().cpu().numpy()

    alpha_zero = torch.tensor([0.0], device=device, requires_grad=True)

    for epoch in tqdm(range(epochs)):


        optimizer.zero_grad()

        # sample the whole batch of loc_alphas
        if args.run_toy:
            loc_alphas = (torch.distributions.Exponential(1 / 0.118).sample((batch_size,)).to(device))  # (B,)
            loc_alphas = torch.clamp(loc_alphas, min=0, max=4 * np.pi)

            #loc_alphas = ( torch.distributions.uniform.Uniform(low=1e-8, high=1).sample((batch_size,)) .to(device))
            #loc_alphas = 1.0 - torch.sqrt(1.0 - loc_alphas)

            # get the data pdf for the sampled loc_alphas for the entire batch
            batch_data_pdf = get_pdf_toy(loc_alphas,args.distribution,t_bin_centers,args.order_to_match, device)  # (B, args.n_bins-1)
        
        else:
            loc_alphas_keys = np.random.choice(list(data_dict.keys()), size=batch_size, replace=False)
            loc_alphas = torch.tensor([a for a in loc_alphas_keys]).to(device).reshape(-1, )
            batch_data_pdf = torch.cat([data_dict[a][0] for a in loc_alphas_keys], axis=1).T
            batch_errors_pdf = torch.cat([data_dict[a][1] for a in loc_alphas_keys], axis=1).T

        # get taylor expansion ansatz for the batch
        alpha_zero = torch.tensor(0.0, device=device, requires_grad=True)
        fn = lambda a: q(
            t_bin_centers, a, g_coeffs_to_fit, theta_to_fit, mstar, t_min, t_max, device, factorial_cache_info
        )
        base = fn(alpha_zero)  # (args.n_bins-1,)

        if args.order_to_match >= 1:
            _, d1 = torch.autograd.functional.jvp(
                fn,
                (alpha_zero,),
                (torch.ones_like(alpha_zero),),
                create_graph=True,
            )  # (args.n_bins-1,)
        if args.order_to_match == 2:
            d1_fn = lambda a: torch.autograd.functional.jvp(
                fn, (a,), (torch.ones_like(a),), create_graph=True
            )[1]
            _, d2 = torch.autograd.functional.jvp(
                d1_fn,
                (alpha_zero,),
                (torch.ones_like(alpha_zero),),
                create_graph=True,
            )  # (args.n_bins-1,)

        # construct the taylor expansion for the loc_alphas
        batch_ansatz = base
        if args.order_to_match >= 1:
            batch_ansatz = batch_ansatz + loc_alphas[:, None] * d1
        if args.order_to_match == 2:
            batch_ansatz = batch_ansatz + 0.5 * (loc_alphas[:, None] ** 2) * d2

        # #  plot the ansatz and data pdf for the first batch
        # if epoch % 10 == 0 or epoch == epochs - 1:
        #     plt.figure()
        #     plt.plot(t_bin_centers.detach().cpu().numpy(), batch_data_pdf[0].detach().cpu().numpy(), label="data")
        #     plt.plot(t_bin_centers.detach().cpu().numpy(), batch_ansatz[0].detach().cpu().numpy(), label="ansatz, taylor expansion", ls="dotted")
        #     plt.plot(t_bin_centers.detach().cpu().numpy(), fn(loc_alphas[0]).detach().cpu().numpy(), label="ansatz, full", ls="dashed")
        #     plt.legend()
        #     plt.title(f"Epoch {epoch + 1}, alpha={loc_alphas[0].item()}")
        #     plt.savefig(f"plots/{outfile_name}_epoch_{epoch}.png")

        # compute the loss

        batch_data_pdf = batch_data_pdf.reshape(-1).float()
        batch_ansatz = batch_ansatz.reshape(-1).float()
        batch_errors_pdf = batch_errors_pdf.reshape(-1).float()

        if args.ratio_loss:

            """

            log_p = torch.log(torch.abs(batch_data_pdf)  + eps)
            log_q = torch.log(torch.abs(batch_ansatz) + eps)


            # Add i*pi to the log of the negative values
            log_p = log_p + 1j * np.pi * (batch_data_pdf < 0)
            log_q = log_q + 1j * np.pi * (batch_ansatz < 0)


            # Take the magnitude of the difference
            diff = torch.abs(log_p - log_q)**2
            loss = torch.mean(torch.nan_to_num(diff))
            """
            ratio = batch_data_pdf / (batch_ansatz + eps)

            loss = torch.pow(
                torch.log(torch.abs(ratio) + eps), 2
            ) + np.pi**2 * (ratio < 0)
            loss = torch.mean(loss)

        else:
            # weighted MSE
            rescaled_pdf = batch_data_pdf[batch_errors_pdf > 0]
            rescaled_ansatz = batch_ansatz[batch_errors_pdf > 0]
            rescaled_errors = batch_errors_pdf[batch_errors_pdf > 0]
            loss = torch.mean(torch.pow(rescaled_pdf-rescaled_ansatz, 2)/torch.pow(rescaled_errors, 2))
            
            #loss = MSE_criterion(batch_data_pdf,batch_ansatz)


        loss.backward(retain_graph=True)


        # this strictly makes things worse??
        # torch.nn.utils.clip_grad_norm_(g_coeffs_to_fit, max_norm = 2.0)

        optimizer.step()

        # technically we should call the val loss...
        scheduler.step()

        losses[epoch] = loss.detach().cpu().numpy()
        lrs[epoch] = scheduler.get_lr()[0]
        g_coeffs_log[epoch + 1] = g_coeffs_to_fit.detach().cpu().numpy()
        theta_log[epoch + 1] = theta_to_fit.detach().cpu().numpy()

    return losses, lrs, g_coeffs_log, theta_log


# Run training
losses, lrs, g_coeffs_log, theta_log = train(
    args.epochs, args.batch_size, args.lr
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
tt = torch.linspace(t_min, 10, 200, device=device)
colors = ["red", "purple", "blue"]


for i, alpha in enumerate([0.148, 0.101, 0.049]):
    alpha_tensor = torch.tensor(alpha, device=device)

    # plot ansatz
    ax[3].plot(tt.detach().cpu().numpy(),q(tt, alpha_tensor, g_coeffs_to_fit, theta_to_fit, mstar, t_min, t_max, device, factorial_cache_info).detach().cpu().numpy(),label="Ansatz",color=colors[i],)

    if args.run_toy:
        # plot all-orders solution
        ax[3].plot(t_bin_centers.detach().cpu().numpy(),t_pdf_toy( alpha_tensor, args.distribution, t_bin_centers, -1, device).detach() .cpu().numpy(), label="Target (exact)",color=colors[i],linestyle="dashed",)
        # plot-fixed order target
        ax[3].scatter(  t_bin_centers.detach().cpu().numpy(),get_pdf_toy( alpha_tensor, args.distribution, t_bin_centers, args.order_to_match,  device ).detach().cpu()  .numpy(),label=f"Target (order $\\alpha^{args.order_to_match}$)",color=colors[i],s=0.8,)

    else:
        # plot histogram
        loc_data, loc_err = data_dict[alpha]
        ax[3].errorbar(t_bin_centers.detach().cpu().numpy(), loc_data.detach().cpu().numpy().reshape(-1,), yerr = loc_err.detach().cpu().numpy().reshape(-1,),  label="Target (data)",  color=colors[i],linestyle="dashed",)



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
