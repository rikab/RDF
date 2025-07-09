import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.func import jacrev
import pickle

from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

plt.style.use('/global/cfs/cdirs/m3246/rikab/dimuonAD/helpers/style_full_notex.mplstyle')

parser = argparse.ArgumentParser()


parser.add_argument("-dist", "--distribution", default="exponential", type=str)
parser.add_argument("-o", "--order_to_match", default=1, type=int)
parser.add_argument("-name", "--name", default="test", type=str)

parser.add_argument("-init_random", "--init_random", action="store_true")
parser.add_argument("-init_at_answer", "--init_at_answer", action="store_true")
parser.add_argument("-init_close_to_answer", "--init_close_to_answer", action="store_true")

parser.add_argument("-e", "--epochs", default=1, type=int)
parser.add_argument("-bs", "--batch_size", default=512, type=int)
parser.add_argument("-lr", "--lr", default=1e-3, type=float)
parser.add_argument("-s", "--seed", default=42, type=int)
parser.add_argument("-m", "--m", default=4, type=int)
parser.add_argument("-n", "--n", default=4, type=int)


mstar = 1

args = parser.parse_args()

outfile_name = f"{args.distribution}_{args.order_to_match}_{args.name}"


device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)


g_coeffs_to_fit = torch.zeros((args.m, args.n), device=device)

if args.init_random:
    for m in range(args.m):
        for n in range(args.n):
            g_coeffs_to_fit[m,n] = 1.0 / (math.factorial(m+1+mstar)*math.factorial(n+1))
    outfile_name += "_init_random"

    
elif args.init_at_answer:
    outfile_name += "_init_at_answer"
    if args.distribution == "exponential":
        g_coeffs_to_fit[0, 0] = 1
    elif args.distribution == "angularity":
        g_coeffs_to_fit[0, 1] = 1
        
        
elif args.init_close_to_answer:
    outfile_name += "_init_close_to_answer"
    if args.distribution == "exponential":
        g_coeffs_to_fit[0, 0] = 0.9
    elif args.distribution == "angularity":
        g_coeffs_to_fit[0, 1] = 0.9
        

else:
    print("Must choose initialization!")
    sys.exit()

max_M, max_N = g_coeffs_to_fit.shape
factorial_cache_n = torch.tensor([math.factorial(k) for k in range(max_N)], device=device).float()
factorial_cache_m = torch.tensor([math.factorial(k) for k in range(max_M)], device=device).float()
n_range = torch.arange(max_N, device=device)
m_range = torch.arange(1, max_M, device=device)

N_integrator = 250

def f(t, alpha, g_coeffs, mstar):
    t_powers = (t[..., None] ** n_range) / factorial_cache_n
    g_star = alpha ** mstar * torch.abs(torch.sum(g_coeffs[0] * t_powers, dim=-1))
    t_powers_exp = t_powers.unsqueeze(-2)
    g_coeffs_higher = g_coeffs[1:] / factorial_cache_m[1:, None]
    g_higher_mat = torch.sum(g_coeffs_higher * t_powers_exp, dim=-1)
    g_higher = torch.sum((alpha ** (mstar + m_range)) * g_higher_mat, dim=-1)
    return g_star * torch.exp(-g_higher)

def cumulative_trapezoidal(alpha, g_coeffs, mstar, t_grid):
    f_vals = f(t_grid, alpha, g_coeffs, mstar)
    dt = t_grid[1] - t_grid[0]
    cum = torch.cumsum((f_vals[:-1] + f_vals[1:]) * 0.5 * dt, dim=0)
    cum = torch.cat([torch.zeros(1, device=device), cum])
    return cum


def q(t, alpha, g_coeffs, mstar):


    t_dense = torch.linspace(0.0, xlim, N_integrator, device=device)
    F_dense = cumulative_trapezoidal(alpha, g_coeffs, mstar, t_dense)

    # Interpolate
    epsilon_regularization = 1e-12
    idx = torch.searchsorted(t_dense, t.clamp(max=t_dense[-1]), right=True) - 1
    idx = idx.clamp(min=0, max=t_dense.shape[0] - 2)
    t0, t1 = t_dense[idx], t_dense[idx + 1]
    F0, F1 = F_dense[idx], F_dense[idx + 1]
    exp_term = F0 + (F1 - F0) * (t - t0) / (t1 - t0 + epsilon_regularization)
    return f(t, alpha, g_coeffs, mstar) * torch.exp(-exp_term)


    
nbins = 80

if args.distribution == "exponential":
    xlim = 20
else:
    xlim = 15

t_bins = torch.linspace(0, xlim, nbins, device=device)
t_bin_centers = 0.5 * (t_bins[1:] + t_bins[:-1])

def get_pdf(alpha, *, example="exponential", order=1):
    alpha = torch.as_tensor(alpha, device=device)[..., None]
    if example == "exponential":
        if order == -1:
            y = alpha * torch.exp(-alpha * t_bin_centers)
        elif order == 1:
            y = alpha.expand_as(alpha * t_bin_centers) * 0 + alpha 
        elif order == 2:
            y = alpha * (1 - alpha * t_bin_centers)
    elif example == "angularity":
        if order == -1:
            y = alpha * t_bin_centers * torch.exp(-alpha * t_bin_centers**2 / 2)
        elif order == 1:
            y = alpha * t_bin_centers
        elif order == 2:
            y = alpha * t_bin_centers * (1 - alpha * t_bin_centers**2 / 2)
    else:
        raise ValueError("bad example/order")
    #y[y < 0] = 0
    return y.squeeze(0)


ofile = open(f"data/{outfile_name}_g_coeffs.txt", "w")
ofile.write(f"Epochs: {args.epochs}\nLearning rate: {args.lr}\nBatch size: {args.batch_size}\n\n")
ofile.write("initial g:\n")
np.savetxt(ofile, g_coeffs_to_fit.detach().cpu().numpy())

def train(epochs, batch_size, lr):


    g_coeffs_to_fit.requires_grad_()
    optimizer = torch.optim.AdamW([g_coeffs_to_fit], lr=lr)
    scheduler = ExponentialLR(optimizer, gamma = 0.9999)

    MSE_criterion = torch.nn.MSELoss()

    losses = np.zeros((epochs, 1))
    lrs = np.zeros((epochs, 1))
    g_coeffs_log = np.zeros((epochs + 1, *g_coeffs_to_fit.shape))
    g_coeffs_log[0] = g_coeffs_to_fit.detach().cpu().numpy()

    alpha_zero = torch.tensor([0.0], device=device, requires_grad=True)


    for epoch in tqdm(range(epochs)):

        optimizer.zero_grad() 

    
        # sample the whole batch of loc_alphas
        loc_alphas = torch.distributions.Exponential(1 / 0.118).sample((batch_size,)).to(device)                      # (B,)

        # get the data pdf for the sampled loc_alphas for the entire batch
        batch_data_pdf = get_pdf(loc_alphas, example=args.distribution, order=args.order_to_match)        # (B, nbins-1)

        # get taylor expansion ansatz for the batch
        alpha_zero = torch.tensor(0.0, device=device, requires_grad=True)
        fn = lambda a: q(t_bin_centers, a, g_coeffs_to_fit, mstar)
        base = fn(alpha_zero)                                # (nbins-1,)

        if args.order_to_match >= 1:
            _, d1 = torch.autograd.functional.jvp(fn,  (alpha_zero,), (torch.ones_like(alpha_zero),), create_graph=True)                         # (nbins-1,)
        if args.order_to_match == 2:
            d1_fn = lambda a: torch.autograd.functional.jvp(fn, (a,), (torch.ones_like(a),), create_graph=True)[1]
            _, d2 = torch.autograd.functional.jvp(d1_fn, (alpha_zero,), (torch.ones_like(alpha_zero),), create_graph=True)                         # (nbins-1,)

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
        loss = MSE_criterion(batch_data_pdf.reshape(-1), batch_ansatz.reshape(-1))
        loss.backward()

        # this strictly makes things worse??
        #torch.nn.utils.clip_grad_norm_(g_coeffs_to_fit, max_norm = 2.0)

        optimizer.step()

        # technically we should call the val loss...
        scheduler.step()

        losses[epoch] = loss.detach().cpu().numpy()
        lrs[epoch] = scheduler.get_lr()[0]
        g_coeffs_log[epoch + 1] = g_coeffs_to_fit.detach().cpu().numpy()

        #print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6e}")
        

        #

    return losses, lrs, g_coeffs_log

# Run training
losses, lrs, g_coeffs_log = train(args.epochs, args.batch_size, args.lr)




# Plot loss
fig, ax = plt.subplots(1, 4, figsize = (24, 6))


ax[0].plot(losses, label="MSE loss")
ax[0].legend()
ax[0].set_yscale("log")
ax[0].set_xlabel("Epoch")



ax[1].plot(lrs, label="MSE loss")
ax[1].legend()
ax[1].set_yscale("log")
ax[1].set_xlabel("LR")


from matplotlib.pyplot import cm
color = iter(cm.hsv(np.linspace(0, 1, g_coeffs_log.shape[1]*g_coeffs_log.shape[2])))

for m in range(g_coeffs_log.shape[1]):
    for n in range(g_coeffs_log.shape[2]):
        c = next(color)
        label = f"$g_{{{m}{n}}}$"
        ax[2].plot(g_coeffs_log[:, m, n], label=label, color=c)
ax[2].legend()
ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("Coefficient value")



tt = torch.linspace(0, xlim, 200, device=device)
colors = ["red", "purple", "blue"]


for i, alpha in enumerate([0.15, 0.1, 0.05]):
    alpha_tensor = torch.tensor(alpha, device=device)
    ax[3].plot( t_bin_centers.detach().cpu().numpy(),  get_pdf(alpha_tensor, example=args.distribution, order=-1).detach().cpu().numpy(), label="Target (exact)",  color=colors[i],  linestyle="dashed" )
    ax[3].scatter(  t_bin_centers.detach().cpu().numpy(), get_pdf(alpha_tensor, example=args.distribution, order=args.order_to_match).detach().cpu().numpy(), label=f"Target (order $\\alpha^{args.order_to_match}$)", color=colors[i], s=0.8)
    ax[3].plot(tt.detach().cpu().numpy(), q(tt, alpha_tensor, g_coeffs_to_fit, mstar).detach().cpu().numpy(), label="Ansatz", color=colors[i])

ax[3].legend()
ax[3].set_xlabel("$t$")
ax[3].set_ylabel("Density")
ax[3].set_ylim(-0.01, 0.4)
plt.savefig(f"plots/{outfile_name}_results.png", bbox_inches = 'tight')

ofile.write("final g:\n")
np.savetxt(ofile, g_coeffs_to_fit.detach().cpu().numpy())
ofile.close()


save_dict = {}
save_dict["loss"] = losses
save_dict["lrs"] = lrs
save_dict["g_coeffs"] = g_coeffs_log
with open(f"data/{outfile_name}", "wb") as ofile:
    pickle.dump(save_dict, ofile)
