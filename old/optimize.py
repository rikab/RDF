import argparse
import yaml
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.func import jacrev
import pickle

from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

plt.style.use('/global/cfs/cdirs/m3246/rikab/dimuonAD/helpers/style_full_notex.mplstyle')



def parse_args_dynamic(defaults):
    parser = argparse.ArgumentParser(description="Dynamic YAML to argparse")

    for key, value in defaults.items():
        arg_type = type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=value)

    return parser.parse_args()


mstar = 1


with open("args.yaml", "r") as ifile:
    config = yaml.safe_load(ifile)

args = parse_args_dynamic(config)


outfile_name = f"{args.distribution}_{args.order_to_match}_{args.name}"


device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)


    



if args.distribution in ["exponential", "angularity"]:
    run_toy = True

elif args.distribution in ["thrust", "c_parameter"]:
    run_toy = False

else:
    print("Must choose a valid distribution")
    exit()



if args.use_logbins:
    t_bins = torch.logspace(np.log10(args.t_min), np.log10(args.t_max), args.n_bins, device=device)
    t_bin_centers = torch.sqrt( (t_bins[1:] * t_bins[:-1]))
else:
    t_bins = torch.linspace(args.t_min, args.t_max, args.n_bins, device=device)
    t_bin_centers = 0.5 * (t_bins[1:] + t_bins[:-1])


g_coeffs_to_fit = torch.nn.Parameter(torch.zeros((args.m, args.n), device=device))
theta_to_fit = torch.nn.Parameter(torch.zeros((args.m, args.n), device=device))


if args.init_random:
    for m in range(args.m):
        for n in range(args.n):
            g_coeffs_to_fit.data[m,n] = 1.0 / (math.factorial(m+1+mstar)*math.factorial(n+1))

    
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

def helper_theta(x, temperature = 20):
    #return torch.where(x >= 0, 1.0, 0.0)
    return torch.sigmoid(temperature * x)


def f(t, alpha, g_coeffs, theta, mstar):
    t_powers = (t[..., None] ** n_range) / factorial_cache_n
    g_star = alpha ** mstar * torch.abs(torch.sum(g_coeffs[0] * t_powers * helper_theta(theta[0]), dim=-1))
    t_powers_exp = t_powers.unsqueeze(-2)
    g_coeffs_higher = g_coeffs[1:] * helper_theta(theta[1:])  / factorial_cache_m[1:, None]
    g_higher_mat = torch.sum(g_coeffs_higher * t_powers_exp, dim=-1)
    g_higher = torch.sum((alpha ** (mstar + m_range)) * g_higher_mat, dim=-1)
    return g_star * torch.exp(-g_higher)

def cumulative_trapezoidal(alpha, g_coeffs, theta, mstar, t_grid):
    f_vals = f(t_grid, alpha, g_coeffs, theta, mstar)
    dt = t_grid[1] - t_grid[0]
    cum = torch.cumsum((f_vals[:-1] + f_vals[1:]) * 0.5 * dt, dim=0)
    cum = torch.cat([torch.zeros(1, device=device), cum])
    return cum


def q(t, alpha, g_coeffs, theta, mstar):


    t_dense = torch.linspace(args.t_min, args.t_max, N_integrator, device=device)
    F_dense = cumulative_trapezoidal(alpha, g_coeffs, theta, mstar, t_dense)

    # Interpolate
    epsilon_regularization = 1e-12
    idx = torch.searchsorted(t_dense, t.clamp(max=t_dense[-1]), right=True) - 1
    idx = idx.clamp(min=0, max=t_dense.shape[0] - 2)
    t0, t1 = t_dense[idx], t_dense[idx + 1]
    F0, F1 = F_dense[idx], F_dense[idx + 1]
    exp_term = F0 + (F1 - F0) * (t - t0) / (t1 - t0 + epsilon_regularization)
    return f(t, alpha, g_coeffs, theta, mstar) * torch.exp(-exp_term)


def get_pdf_toys(alpha, *, example="exponential", order=1):
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




def read_in_data(file_indices, example, t_bins):
    from sklearn.preprocessing import MinMaxScaler


    data_dict = {}
    bin_width = t_bins[1] - t_bins[0]

    for i in file_indices:
        with open(f"event_records_LO_{i}.pkl", "rb") as ifile:
            loc_data_dict = pickle.load(ifile)
            for alpha in loc_data_dict.keys():
                loc_data = loc_data_dict[alpha][example]
                if example == "thrust":
                    loc_data = [2.0*(1.0 - x) for x in loc_data]

                #scaler = MinMaxScaler()
                #loc_data = scaler.fit_transform(np.array(loc_data).reshape(-1,1))

                # convert to t-space
                loc_data = [np.log(1.0 / (x + 1e-12)) for x in loc_data]
                
                y, _ = np.histogram(loc_data, weights = loc_data_dict[alpha]["weight"], bins = t_bins, density = False)
                y /= bin_width
                y = torch.tensor(y, device = device).reshape(-1, 1)
                data_dict[alpha] = y.squeeze(0).float()

    return data_dict

    
ofile = open(f"data/{outfile_name}_g_coeffs.txt", "w")
ofile.write(f"Epochs: {args.epochs}\nLearning rate: {args.lr}\nBatch size: {args.batch_size}\n\n")
ofile.write("initial g:\n")
np.savetxt(ofile, g_coeffs_to_fit.detach().cpu().numpy())

if not run_toy: # only needs to be done once
    data_dict = read_in_data([0, 1], args.distribution, t_bins)

def train(epochs, batch_size, lr):



    if args.learn_theta:
        optimizer = torch.optim.AdamW([g_coeffs_to_fit, theta_to_fit], lr=lr)
    else:
        optimizer = torch.optim.AdamW([g_coeffs_to_fit], lr=lr)
        
    scheduler = ExponentialLR(optimizer, gamma = 1)

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
        if run_toy:
            if False:
                loc_alphas = torch.distributions.Exponential(1 / 0.118).sample((batch_size,)).to(device)                      # (B,)
                loc_alphas = torch.clamp(loc_alphas, min = 0, max = 4*np.pi)
            else:
                loc_alphas = torch.distributions.uniform.Uniform(low = 1e-8, high = 1).sample((batch_size,)).to(device)    
                loc_alphas = 1.0 - torch.sqrt(1.0 - loc_alphas)

            # get the data pdf for the sampled loc_alphas for the entire batch
            batch_data_pdf = get_pdf_toy(loc_alphas, example=args.distribution, order=args.order_to_match)        # (B, args.n_bins-1)
            
        else:
            loc_alphas_keys = np.random.choice(list(data_dict.keys()), size = batch_size, replace = False)
            loc_alphas = torch.tensor([float(a.split("_")[1])*0.001 for a in loc_alphas_keys]).to(device).reshape(-1,)
            batch_data_pdf = torch.cat([data_dict[a] for a in loc_alphas_keys], axis = 1).T

        # get taylor expansion ansatz for the batch
        alpha_zero = torch.tensor(0.0, device=device, requires_grad=True)
        fn = lambda a: q(t_bin_centers, a, g_coeffs_to_fit, theta_to_fit, mstar)
        base = fn(alpha_zero)                                # (args.n_bins-1,)

        if args.order_to_match >= 1:
            _, d1 = torch.autograd.functional.jvp(fn,  (alpha_zero,), (torch.ones_like(alpha_zero),), create_graph=True)                         # (args.n_bins-1,)
        if args.order_to_match == 2:
            d1_fn = lambda a: torch.autograd.functional.jvp(fn, (a,), (torch.ones_like(a),), create_graph=True)[1]
            _, d2 = torch.autograd.functional.jvp(d1_fn, (alpha_zero,), (torch.ones_like(alpha_zero),), create_graph=True)                         # (args.n_bins-1,)

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

        batch_data_pdf = batch_data_pdf.reshape(-1)
        batch_ansatz = batch_ansatz.reshape(-1)

        
        if args.ratio_loss:
            eps = 1e-12
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

            loss = torch.pow(torch.log(torch.abs(ratio) + eps), 2) +  np.pi**2 * (ratio  < 0)
            loss = torch.mean(loss)

        else:
            loss = MSE_criterion(batch_data_pdf, batch_ansatz)

            
        loss.backward()

        # this strictly makes things worse??
        #torch.nn.utils.clip_grad_norm_(g_coeffs_to_fit, max_norm = 2.0)

        optimizer.step()

        # technically we should call the val loss...
        scheduler.step()

        losses[epoch] = loss.detach().cpu().numpy()
        lrs[epoch] = scheduler.get_lr()[0]
        g_coeffs_log[epoch + 1] = g_coeffs_to_fit.detach().cpu().numpy()
        theta_log[epoch + 1] = theta_to_fit.detach().cpu().numpy()

        #print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6e}")
        

        #

    return losses, lrs, g_coeffs_log, theta_log

# Run training
losses, lrs, g_coeffs_log, theta_log = train(args.epochs, args.batch_size, args.lr)




# Plot loss
fig, ax = plt.subplots(1, 3, figsize = (24, 6))

if args.ratio_loss:
    ax[0].plot(losses, label="ratio loss")
else:
    ax[0].plot(losses, label="MSE loss")
ax[0].legend()
ax[0].set_yscale("log")
ax[0].set_xlabel("Epoch")



from matplotlib.pyplot import cm
color = iter(cm.hsv(np.linspace(0, 1, g_coeffs_log.shape[1]*g_coeffs_log.shape[2])))

for m in range(g_coeffs_log.shape[1]):
    for n in range(g_coeffs_log.shape[2]):
        c = next(color)
        label = f"$g_{{{m}{n}}}$"
        ax[1].plot(g_coeffs_log[:, m, n], label=label, color=c)
ax[1].legend()
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Coefficient value")



tt = torch.linspace(args.t_min, args.t_max, 200, device=device)
colors = ["red", "purple", "blue"]



for i, alpha in enumerate([0.15, 0.1, 0.05]):
    alpha_tensor = torch.tensor(alpha, device=device)
    ax[2].plot(tt.detach().cpu().numpy(), q(tt, alpha_tensor, g_coeffs_to_fit, theta_to_fit, mstar).detach().cpu().numpy(), label="Ansatz", color=colors[i])
    if run_toy:
        ax[2].plot( t_bin_centers.detach().cpu().numpy(),  get_pdf_toy(alpha_tensor, example=args.distribution, order=-1).detach().cpu().numpy(), label="Target (exact)",  color=colors[i],  linestyle="dashed" )
        ax[2].scatter(  t_bin_centers.detach().cpu().numpy(), get_pdf_toy(alpha_tensor, example=args.distribution, order=args.order_to_match).detach().cpu().numpy(), label=f"Target (order $\\alpha^{args.order_to_match}$)", color=colors[i], s=0.8)
    else:
        alpha_string = "alpha_"+str(int(1000*alpha)).zfill(4)
        ax[2].plot( t_bin_centers.detach().cpu().numpy(),  data_dict[alpha_string].detach().cpu().numpy(), label="Target (exact)",  color=colors[i],  linestyle="dashed" )
    

    
   

ax[2].legend()
ax[2].set_xlabel("$t$")
ax[2].set_ylabel("Density")
#ax[2].set_ylim(-0.01, 0.4)
plt.savefig(f"plots/{outfile_name}_results.png", bbox_inches = 'tight')

ofile.write("final g:\n")
np.savetxt(ofile, g_coeffs_to_fit.detach().cpu().numpy())
ofile.close()


save_dict = {}
save_dict["loss"] = losses
save_dict["lrs"] = lrs
save_dict["g_coeffs"] = g_coeffs_log
save_dict["theta"] = theta_log

with open(f"data/{outfile_name}", "wb") as ofile:
    pickle.dump(save_dict, ofile)

print(theta_to_fit)
