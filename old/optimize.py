import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import scipy.integrate as integrate

import torch
import torchquad

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-dist", "--distribution", default="exponential", type=str)
parser.add_argument("-o", "--order_to_match", default=1, type = int)
parser.add_argument("-name", "--name", default="test", type = str)
parser.add_argument("-init_random", "--init_random", action="store_true")


parser.add_argument("-e", "--epochs", default=1, type = int)
parser.add_argument("-bs", "--batch_size", default=100, type = int)
parser.add_argument("-lr", "--lr", default=1e-3, type = float)
args = parser.parse_args()


outfile_name = f"{args.distribution}_{args.order_to_match}_{args.name}"

device = "cpu"

if args.init_random:
    g_coeffs_to_fit = 0.1*torch.rand(size=(4,3)).float()
else:
    g_coeffs_to_fit = torch.tensor(np.array([[1, 0], [0,0],[0,0]]), device=device).float()




# may want to revisit these choices
integrator = torchquad.Trapezoid()
N_integrator = 500


# ansatz helper functions

# g_coeffs: M x N matrix such that g = sum_{m, n} g_mn t^n alpha ^m
    # 1st row is g* term

def f(t, alpha, g_coeffs, mstar):
    M,N = g_coeffs.shape
    g_star = alpha**mstar * torch.abs(sum([g_coeffs[0,n]*torch.pow(t,n).to(device)/torch.tensor(np.math.factorial(n), device=device) for n in range(N) ]))
    g_higher = sum([alpha**(mstar + m) * g_coeffs[m,n]*torch.pow(t,n).to(device)/torch.tensor((np.math.factorial(n)*np.math.factorial(m)), device=device) for n in range(N) for m in range(1,M) ])
    return g_star*torch.exp(-g_higher)


# SCALAR FUNCTION
def q(t, alpha, g_coeffs, mstar):

    f_of_t = lambda tt: f(tt, alpha, g_coeffs, mstar)
    
    intfunc = lambda tt: integrator.integrate(f_of_t, dim=1, N=N_integrator, integration_domain=[[0, tt]])

    # vmap does not like the torch integrator but it's running decently quickly now...
    #vec_int = torch.func.vmap(intfunc)
    #exp_term = vec_int(t)

    exp_term = intfunc(t)

    return f_of_t(t)*torch.exp(-exp_term).to(device)



# choose data distribution


nbins = 80
if args.distribution == "exponential": 
    xlim = 100
elif args.distribution == "angularity":
    xlim = 15

t_bins = torch.linspace(0, xlim, nbins)
t_bin_centers = 0.5*(t_bins[1:] + t_bins[:-1]) 

def get_pdf(alpha, example="exponential", order=1):

    if example == "exponential":
        if order == - 1:
            y = torch.tensor(alpha*np.exp(-alpha*t_bin_centers))
        elif order == 1:
            y = alpha*torch.ones_like(t_bin_centers)
        elif order == 2: # 
            y = torch.tensor(alpha*(1-alpha*t_bin_centers))

    elif example == "angularity":
        if order == - 1:
            y = torch.tensor(alpha*t_bin_centers*np.exp(-alpha*t_bin_centers**2/2.0))
        elif order == 1: # 1st order
            y = torch.tensor(alpha*t_bin_centers)
        elif order == 2:
            y = torch.tensor(alpha*t_bin_centers*(1 - alpha*t_bin_centers**2/2.0))

  
    return y



# Initialize the g_coefficients

mstar = 1

ofile = open(f"plots/{outfile_name}_g_coeffs.txt", "w")

ofile.write(f"Epochs: {args.epochs}\n")
ofile.write(f"Learning rate: {args.lr}\n")
ofile.write(f"Batch size: {args.batch_size}\n\n")


ofile.write("initial g:\n")
np.savetxt(ofile, g_coeffs_to_fit.detach().cpu().numpy(), fmt='%f')
    
            
# TODO: results may be sensitive to the binning choice...



def train(epochs, batch_size, lr):


    g_coeffs_to_fit.requires_grad_()
    optimizer = torch.optim.AdamW([g_coeffs_to_fit], lr=lr)

    MSE_criterion = torch.nn.MSELoss()

    losses = np.zeros(shape = (epochs, 1))
    g_coeffs_log = np.zeros(shape = (epochs+1, g_coeffs_to_fit.shape[0],  g_coeffs_to_fit.shape[1]))

    g_coeffs_log[0] = g_coeffs_to_fit.detach().cpu().numpy()
    
    for epoch in tqdm(range(epochs)):

        batch_data_pdf = torch.zeros(size=(batch_size*(nbins-1),))
        batch_ansatz = torch.zeros(size=(batch_size*(nbins-1),))


        optimizer.zero_grad()

        for bs in range(batch_size):
    
            # generate a random alpha for the epoch
            loc_alpha = torch.distributions.Exponential(1 / 0.118).sample()
                
            # generate training data
            loc_data_pdf = get_pdf(loc_alpha, example=args.distribution, order=args.order_to_match)

            # add data to batch array
            batch_data_pdf[bs*(nbins-1):(bs+1)*(nbins-1)] = loc_data_pdf

            # calculate taylor components of the ansatz
            # need to do this for a scalar function toget the higher derivatives to work
            # TODO if mstar > m, then we only need to start at the d^m derivative since all lower ones will be zero
            alpha_zero = torch.tensor([0.0], device=device)
            alpha_zero.requires_grad_()

            for  i,t in enumerate(t_bin_centers):

                # 0th order
                loc_ansatz = q(t, alpha_zero, g_coeffs_to_fit, mstar)

                dx = loc_ansatz
                for order in range(1, args.order_to_match + 1):

                    dx = torch.autograd.grad(dx, alpha_zero, create_graph=True, retain_graph=True, allow_unused=True)[0]
                    alpha_pow = torch.pow(loc_alpha, order) / torch.tensor(np.math.factorial(order))

                    loc_ansatz += alpha_pow*dx

                # TODO: rescale by alpha? should all items in the batch have the same weight?
                batch_ansatz[bs*(nbins-1) + i] = loc_ansatz

            


    
            """
            plt.figure()
            plt.plot(t_bin_centers.detach().cpu().numpy(), loc_data_pdf)
            plt.plot(t_bin_centers.detach().cpu().numpy(), loc_ansatz.detach().cpu().numpy())
            plt.title(loc_alpha)
            plt.show()
            """


        loss = MSE_criterion(batch_data_pdf.float(), batch_ansatz.float())

           
        loss.backward()
        optimizer.step()

        # log 
        losses[epoch] = loss.detach().cpu().numpy()
        g_coeffs_log[epoch+1] = g_coeffs_to_fit.detach().cpu().numpy()


    return losses, g_coeffs_log

losses, g_coeffs_log = train(args.epochs, args.batch_size, args.lr)



plt.figure()

plt.plot(losses, label = "MSE loss")
plt.legend()
plt.yscale("log")

plt.xlabel("Epoch")
plt.savefig(f"plots/{outfile_name}_loss.png")

plt.figure()
for m in range(g_coeffs_log.shape[1]):
    for n in range(g_coeffs_log.shape[2]):

        plt.plot(g_coeffs_log[:,m,n], label = f"g_{m}{n}")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Coefficient value")
plt.savefig(f"plots/{outfile_name}_coefficients.png")




for alpha in [0.15, 0.1, 0.05]:
    
    
    tt = torch.linspace(0, xlim, 200)
    
    plt.figure()
    plt.plot(t_bin_centers.detach().cpu().numpy(), get_pdf(alpha, example=args.distribution, order=args.order_to_match), label = "data, FO")
    plt.plot(t_bin_centers.detach().cpu().numpy(), get_pdf(alpha, example=args.distribution, order=-1), label = "data, full")
    plt.plot(tt , [q(ttt, alpha, g_coeffs_to_fit, mstar).detach().cpu().numpy() for ttt in tt], label = "ansatz")
    plt.legend()
    plt.xlabel("t")
    plt.title(f"alpha={alpha}")
    plt.savefig(f"plots/{outfile_name}_results_alpha_{alpha}.png")


ofile.write("final g:\n")
np.savetxt(ofile, g_coeffs_to_fit.detach().cpu().numpy(), fmt='%f')

ofile.close()