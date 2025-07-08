import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.func import jacrev



parser = argparse.ArgumentParser()

parser.add_argument("-dist", "--distribution", default="exponential", type=str)
parser.add_argument("-o", "--order_to_match", default=1, type=int)
parser.add_argument("-name", "--name", default="test", type=str)
parser.add_argument("-init_random", "--init_random", action="store_true")
parser.add_argument("-e", "--epochs", default=1, type=int)
parser.add_argument("-bs", "--batch_size", default=512, type=int)
parser.add_argument("-lr", "--lr", default=1e-3, type=float)
args = parser.parse_args()

outfile_name = f"{args.distribution}_{args.order_to_match}_{args.name}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Initalize coefficients to fit
if args.init_random:
    g_coeffs_to_fit = 0.1 * torch.rand(size=(4, 3)).float()
else:
    g_coeffs_to_fit = torch.tensor(np.array([[1, 0], [0, 0], [0, 0]]), device=device).float()


# """ 
# initial g:
# 0.015901 0.016642
# 0.086209 0.002081
# 0.049107 0.028169
# """

# g_coeffs_to_fit = torch.tensor([[0.015901, 0.016642],
#                                  [0.086209, 0.002081],
#                                  [0.049107, 0.028169]], device=device).float()

# Precompute factorials and ranges for vectorized operations
max_M, max_N = g_coeffs_to_fit.shape
factorial_cache_n = torch.tensor([math.factorial(k) for k in range(max_N)], device=device).float()  # shape (N,)
factorial_cache_m = torch.tensor([math.factorial(k) for k in range(max_M)], device=device).float()  # shape (M,)
n_range = torch.arange(max_N, device=device)              # 0..N-1
m_range = torch.arange(1, max_M, device=device)           # 1..M-1



N_integrator = 250

#  ansatz helper functions
def f(t, alpha, g_coeffs, mstar):
 

    # t^n / n!
    t_powers = (t[..., None] ** n_range) / factorial_cache_n           # (..., N)

    # g_star (row 0)
    g_star = alpha ** mstar * torch.abs(torch.sum(g_coeffs[0] * t_powers, dim=-1))

    # g_higher (rows 1,..., M-1)
    t_powers_exp = t_powers.unsqueeze(-2)                              # (..., 1, N)
    g_coeffs_higher = g_coeffs[1:] / factorial_cache_m[1:, None]       # (M-1, N)
    g_higher_mat = torch.sum(g_coeffs_higher * t_powers_exp, dim=-1)   # (..., M-1)
    g_higher = torch.sum((alpha ** (mstar + m_range)) * g_higher_mat, dim=-1)

    return g_star * torch.exp(-g_higher)


# cumulative‐trapz
# @RADHA, replaced integral with cumulative trapezoidal rule, since we need every t value later anyways
def cumulative_trapezoidal(alpha, g_coeffs, mstar, t_grid):

    f_vals = f(t_grid, alpha, g_coeffs, mstar) 
    dt = t_grid[1] - t_grid[0]
    cum = torch.cumsum((f_vals[:-1] + f_vals[1:]) * 0.5 * dt, dim=0)
    cum = torch.cat([torch.zeros(1, device=device), cum])          # prepend F(0)=0
    return cum


# lookup cache; cleared each epoch to stay in sync with updated g coeffs
cumtrapz_cache = {}

# SCALAR FUNCTION
def q(t, alpha, g_coeffs, mstar):


    alpha_key = float(alpha)

    # Check if we have already computed the cumulative trapezoidal integral for this alpha
    if alpha_key not in cumtrapz_cache: 
        t_dense = torch.linspace(0.0, xlim, N_integrator, device=device)
        cumtrapz_cache[alpha_key] = (  t_dense,cumulative_trapezoidal(alpha, g_coeffs, mstar, t_dense), )
        # format is (t_dense, F_dense) pairs, with alpha_key as the key


    t_dense, F_dense = cumtrapz_cache[alpha_key]

    # linear interpolation of F(t)

    epsilon_regularization = 1e-12  # to avoid division by zero
    idx = torch.searchsorted(t_dense, t.clamp(max=t_dense[-1]), right=True) - 1
    idx = idx.clamp(min=0, max=t_dense.shape[0] - 2)
    t0, t1 = t_dense[idx], t_dense[idx + 1]
    F0, F1 = F_dense[idx], F_dense[idx + 1]
    exp_term = F0 + (F1 - F0) * (t - t0) / (t1 - t0 + epsilon_regularization)

    return f(t, alpha, g_coeffs, mstar) * torch.exp(-exp_term)



nbins = 80
if args.distribution == "exponential":
    xlim = 100
elif args.distribution == "angularity":
    xlim = 15

t_bins = torch.linspace(0, xlim, nbins)
t_bin_centers = 0.5 * (t_bins[1:] + t_bins[:-1])

def get_pdf(alpha, *, example="exponential", order=1):


    alpha = torch.as_tensor(alpha, device=t_bin_centers.device)[..., None]  

    if example == "exponential":
        if order == -1:
            y = alpha * torch.exp(-alpha * t_bin_centers)          # (B, nbins-1)
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
    return y.squeeze(0) 





# logging
mstar = 1
ofile = open(f"plots/{outfile_name}_g_coeffs.txt", "w")
ofile.write(f"Epochs: {args.epochs}\nLearning rate: {args.lr}\nBatch size: {args.batch_size}\n\n")
ofile.write("initial g:\n")
np.savetxt(ofile, g_coeffs_to_fit.detach().cpu().numpy(), fmt='%f')


# train function
def train(epochs, batch_size, lr):


    g_coeffs_to_fit.requires_grad_()
    optimizer = torch.optim.AdamW([g_coeffs_to_fit], lr=lr)

    MSE_criterion = torch.nn.MSELoss()

    losses = np.zeros((epochs, 1))
    g_coeffs_log = np.zeros((epochs + 1, *g_coeffs_to_fit.shape))
    g_coeffs_log[0] = g_coeffs_to_fit.detach().cpu().numpy()

    alpha_zero = torch.tensor([0.0], device=device, requires_grad=True)


    for epoch in tqdm(range(epochs)):

        optimizer.zero_grad() 


        # clear cache because g_coeffs just changed last step
        cumtrapz_cache.clear()

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

        # compute the loss
        loss = MSE_criterion(batch_data_pdf.reshape(-1), batch_ansatz.reshape(-1))
        loss.backward()
        optimizer.step()

        losses[epoch] = loss.detach().cpu().numpy()
        g_coeffs_log[epoch + 1] = g_coeffs_to_fit.detach().cpu().numpy()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6e}")
        

    return losses, g_coeffs_log


# run training
losses, g_coeffs_log = train(args.epochs, args.batch_size, args.lr)

# plots
plt.figure()
plt.plot(losses, label="MSE loss")
plt.legend()
plt.yscale("log")
plt.xlabel("Epoch")
plt.savefig(f"plots/{outfile_name}_loss.png")

plt.figure()
for m in range(g_coeffs_log.shape[1]):
    for n in range(g_coeffs_log.shape[2]):
        plt.plot(g_coeffs_log[:, m, n], label=f"g_{m}{n}")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Coefficient value")
plt.savefig(f"plots/{outfile_name}_coefficients.png")

for alpha in [0.15, 0.1, 0.05]:
    tt = torch.linspace(0, xlim, 200)
    plt.figure()
    plt.plot(t_bin_centers.detach().cpu().numpy(), get_pdf(alpha, example=args.distribution, order=args.order_to_match), label="data, FO")
    plt.plot(t_bin_centers.detach().cpu().numpy(), get_pdf(alpha, example=args.distribution, order=-1), label="data, full")
    plt.plot(tt, [q(ttt, alpha, g_coeffs_to_fit, mstar).detach().cpu().numpy() for ttt in tt], label="ansatz")
    plt.legend()
    plt.xlabel("t")
    plt.title(f"alpha={alpha}")
    plt.savefig(f"plots/{outfile_name}_results_alpha_{alpha}.png")

ofile.write("final g:\n")
np.savetxt(ofile, g_coeffs_to_fit.detach().cpu().numpy(), fmt='%f')
ofile.close()
