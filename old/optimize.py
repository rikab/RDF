import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.func import jacrev

plt.style.use('/global/cfs/cdirs/m3246/rikab/dimuonAD/helpers/style_full_notex.mplstyle')

parser = argparse.ArgumentParser()


parser.add_argument("-dist", "--distribution", default="exponential", type=str)
parser.add_argument("-o", "--order_to_match", default=1, type=int)
parser.add_argument("-name", "--name", default="test", type=str)
parser.add_argument("-init_random", "--init_random", action="store_true")
parser.add_argument("-e", "--epochs", default=1, type=int)
parser.add_argument("-bs", "--batch_size", default=100, type=int)
parser.add_argument("-lr", "--lr", default=1e-3, type=float)
parser.add_argument("-s", "--seed", default=42, type=int)
parser.add_argument("-m", "--m", default=3, type=int)
parser.add_argument("-n", "--n", default=3, type=int)
parser.add_argument("-use_rikab_loss", "--use_rikab_loss", action="store_true")



args = parser.parse_args()

outfile_name = f"{args.distribution}_{args.order_to_match}_{args.name}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)


if args.init_random:
    g_coeffs_to_fit = 0.1 * torch.rand(size=(args.m, args.n), device=device).float()
else:
    g_coeffs_to_fit = torch.zeros((args.m, args.n), device=device)
    if args.distribution == "exponential":
        g_coeffs_to_fit[0, 0] = 1
    elif args.distribution == "angularity":
        g_coeffs_to_fit[0, 1] = 1

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

cumtrapz_cache = {}

def q(t, alpha, g_coeffs, mstar):
    alpha_key = float(alpha)
    if alpha_key not in cumtrapz_cache:
        t_dense = torch.linspace(0.0, xlim, N_integrator, device=device)
        cumtrapz_cache[alpha_key] = (
            t_dense,
            cumulative_trapezoidal(alpha, g_coeffs, mstar, t_dense),
        )

    t_dense, F_dense = cumtrapz_cache[alpha_key]
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
            y = alpha.expand_as(t_bin_centers)
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

mstar = 1
ofile = open(f"plots/{outfile_name}_g_coeffs.txt", "w")
ofile.write(f"Epochs: {args.epochs}\nLearning rate: {args.lr}\nBatch size: {args.batch_size}\n\n")
ofile.write("initial g:\n")
np.savetxt(ofile, g_coeffs_to_fit.detach().cpu().numpy())

def train(epochs, batch_size, lr):
    
    g_coeffs_to_fit.requires_grad_()
    optimizer = torch.optim.AdamW([g_coeffs_to_fit], lr=lr)
    MSE_criterion = torch.nn.MSELoss()
    losses = np.zeros((epochs, 1))
    g_coeffs_log = np.zeros((epochs + 1, *g_coeffs_to_fit.shape))
    g_coeffs_log[0] = g_coeffs_to_fit.detach().cpu().numpy()
    alpha_zero = torch.zeros(1, device=device, requires_grad=True)

    
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        # clear cache because g_coeffs just changed last step
        cumtrapz_cache.clear()

        # RADHA LOSS
        if not args.use_rikab_loss:

            batch_data_pdf = torch.zeros(batch_size * (nbins - 1), device=device)
            batch_ansatz = torch.zeros(batch_size * (nbins - 1), device=device)
    
            for bs in range(batch_size):
                loc_alpha = torch.distributions.Exponential(torch.tensor(1 / 0.118, device=device)).sample()
                with torch.no_grad():
                    loc_data_pdf = get_pdf(loc_alpha, example=args.distribution, order=args.order_to_match)
                batch_data_pdf[bs * (nbins - 1):(bs + 1) * (nbins - 1)] = loc_data_pdf
    
                for i, t in enumerate(t_bin_centers):
                    loc_ansatz = q(t, alpha_zero, g_coeffs_to_fit, mstar)
                    dx = loc_ansatz
                    for order in range(1, args.order_to_match + 1):
                        dx = torch.autograd.grad(dx, alpha_zero, create_graph=True, retain_graph=True)[0]
                        loc_ansatz += (loc_alpha ** order / math.factorial(order)) * dx
                    batch_ansatz[bs * (nbins - 1) + i] = loc_ansatz

                """
                if bs == 0:
                    plt.figure()
                    plt.plot(t_bin_centers.detach().cpu().numpy(), loc_data_pdf.detach().cpu().numpy(), label = "data")
                    plt.plot(t_bin_centers.detach().cpu().numpy(), batch_ansatz[:nbins-1].detach().cpu().numpy(), label = "ansatz, taylor expansion")
                    plt.plot(t_bin_centers.detach().cpu().numpy(), [q(t, loc_alpha, g_coeffs_to_fit, mstar).detach().cpu().numpy() for tt in t_bin_centers], label = "ansatz, full")
                    plt.legend()
                    plt.title(loc_alpha)
                    plt.savefig(f"plots/{epoch}.png")
                """

        elif args.use_rikab_loss:

   
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
        
        
            
        loss = MSE_criterion(batch_data_pdf, batch_ansatz)
        loss.backward()
        optimizer.step()

        losses[epoch] = loss.detach().cpu().numpy()
        g_coeffs_log[epoch + 1] = g_coeffs_to_fit.detach().cpu().numpy()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6e}")

    return losses, g_coeffs_log

# Run training
losses, g_coeffs_log = train(args.epochs, args.batch_size, args.lr)

# Plot loss
plt.figure()
plt.plot(losses, label="MSE loss")
plt.legend()
plt.yscale("log")
plt.xlabel("Epoch")
plt.savefig(f"plots/{outfile_name}_loss.png")

from matplotlib.pyplot import cm
color = iter(cm.hsv(np.linspace(0, 1, g_coeffs_log.shape[1]*g_coeffs_log.shape[2])))

plt.figure()
for m in range(g_coeffs_log.shape[1]):
    for n in range(g_coeffs_log.shape[2]):
        c = next(color)
        label = f"$g_{{{m}{n}}}$"
        plt.plot(g_coeffs_log[:, m, n], label=label, color=c)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Coefficient value")
plt.savefig(f"plots/{outfile_name}_coefficients.png")

tt = torch.linspace(0, xlim, 200, device=device)
colors = ["red", "purple", "blue"]


plt.figure()
for i, alpha in enumerate([0.15, 0.1, 0.05]):
    alpha_tensor = torch.tensor(alpha, device=device)
    plt.plot( t_bin_centers.detach().cpu().numpy(),  get_pdf(alpha_tensor, example=args.distribution, order=-1).detach().cpu().numpy(), label="Target (exact)",  color=colors[i],  linestyle="dashed" )
    plt.scatter(  t_bin_centers.detach().cpu().numpy(), get_pdf(alpha_tensor, example=args.distribution, order=args.order_to_match).detach().cpu().numpy(), label=f"Target (order $\\alpha^{args.order_to_match}$)", color=colors[i], s=0.8)
    plt.plot(tt.detach().cpu().numpy(), q(tt, alpha_tensor, g_coeffs_to_fit, mstar).detach().cpu().numpy(), label="Ansatz", color=colors[i])

plt.legend()
plt.xlabel("$t$")
plt.ylabel("Density")
plt.savefig(f"plots/{outfile_name}_results.png")

ofile.write("final g:\n")
np.savetxt(ofile, g_coeffs_to_fit.detach().cpu().numpy())
ofile.close()
