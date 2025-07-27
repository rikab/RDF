import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tqdm import tqdm

import torch
import zuko
import argparse
import yaml

from helpers.distributions import *
from helpers.flow import *

parser = argparse.ArgumentParser()


parser.add_argument("-id", "--run_id", default="test")
parser.add_argument("-odir", "--outdir", default="/global/u1/r/rmastand/NNEFT/old/test/")
parser.add_argument("-mode", "--mode", default="UNIFORM_SAMPLES")
parser.add_argument("-loss", "--loss", default="logMSE", help="choose from linearMSE, logMSE, ratioMSE")
parser.add_argument("-C", "--C", default="C_theta", help="choose from C_theta, C_alpha_1, C_alpha_2, C_alpha_log_1, C_alpha_log_2, C_unscaled, C_theory")
parser.add_argument("-debug", "--debug", action="store_true")
parser.add_argument("-pretrain_mse", "--pretrain_mse", action="store_true")


# hyperparameters
parser.add_argument("-epochs", "--epochs", default=200, type=int)
parser.add_argument("-nx", "--batch_num_x", default=128, type=int)
parser.add_argument("-nc", "--batch_num_c", default=8, type=int)
parser.add_argument("-lr", "--lr", default=1e-4)
parser.add_argument("-seed", "--seed", default=1, type=int)


# network architecture
parser.add_argument("-aux", "--auxiliary_params", default=1, type=int)
parser.add_argument("-nt", "--num_transforms", default=3, type=int)
parser.add_argument("-hf", "--hidden_features", default="32,32")

args = parser.parse_args()

run_params = {"hyperparams":{}, "architecture":{}, "physics":{}}


# Physics Parameters
E0 = 500
R = 0.4
beta = 1
target_p = LO_angularity
run_params["physics"]["E0"] = E0
run_params["physics"]["R"] = R
run_params["physics"]["beta"] = beta
run_params["physics"]["target_p"] = str(target_p)


MODE = args.mode
LOSS = args.loss
C = args.C
DEBUG = args.debug
run_params["MODE"] = MODE
run_params["LOSS"] = LOSS
run_params["C"] = C
run_params["pretrain_mse"] = args.pretrain_mse


    
    
epochs = args.epochs
batch_num_x = args.batch_num_x
batch_num_c = args.batch_num_c
batch_size = batch_num_x*batch_num_c
lr = float(args.lr)

run_params["hyperparams"]["epochs"] = epochs
run_params["hyperparams"]["batch_num_x"] = batch_num_x
run_params["hyperparams"]["batch_num_c"] = batch_num_c
run_params["hyperparams"]["lr"] = lr
run_params["hyperparams"]["seed"] = int(args.seed)


auxiliary_params = args.auxiliary_params # Number of auxiliary parameters to use in the model, 0 has different behavior!
num_transforms = args.num_transforms # I think this is the number of layers?
hidden_features = [int(x) for x in args.hidden_features.split(",")]
x_range = (0, 1) # Range of x values to train on, should be at least as large as the support of the data
run_params["architecture"]["num_aux"] = auxiliary_params
run_params["architecture"]["num_transforms"] = num_transforms
run_params["architecture"]["hidden_features"] = hidden_features

torch.manual_seed(args.seed)
np.random.seed(args.seed)

"""
def f(a, b):
    return a*b

a = torch.tensor([1.0], requires_grad = True)
b = torch.tensor([torch.inf], requires_grad = True)
func = f(a,b)

df_da = torch.ones(a.shape)
df_da = torch.autograd.grad(inputs=a, outputs=func, grad_outputs=df_da, allow_unused=True, retain_graph=True)[0]
print(df_da)

 """         


# initialize the flow

flow = zuko.flows.NSF(features = auxiliary_params + 1, context=1, transforms= num_transforms, hidden_features=hidden_features)

optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

pytorch_total_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
print(f"Numb. trainable params: {pytorch_total_params}")
run_params["architecture"]["num_trainable_params"] = pytorch_total_params


with open(f"{args.outdir}/{args.run_id}.yml", "w") as outfile:
    yaml.dump(run_params, outfile, default_flow_style=False)
    
    
"""
TRAINING LOOP
"""

t = tqdm(range(epochs))
losses = []
training_cs = []
logps = []
logqs = []
logJs = []
xss = []

epsilon = 1e-7


for epoch in t:
    
    if args.pretrain_mse:
        train_mse = True
        if epoch < int(epochs / 2):
            train_residual = False
        else:
            train_residual = True

    else:
        train_mse = False 
        train_residual = True
        
    loss = 0
            
            

    # Used to be called "REVERSE" / "No samples"
    if MODE == "UNIFORM_SAMPLES":

        reroll = True

        while reroll:
            xs = torch.rand(batch_size, auxiliary_params + 1)  * (x_range[1] - x_range[0]) + x_range[0]
            # Auxillary variable -- uniform. alternatively set to const
            if auxiliary_params > 0:
                xs[:,1:] = torch.rand(batch_size, auxiliary_params)
                # xs[:,1] = torch.linspace(0.001, 0.999, batch_size)

            # Check if any x are 0 or 1 to float point error, reroll if so
            if torch.any(xs <= epsilon) or torch.any(xs >= 1 - epsilon):
                reroll = True
            else:
                reroll = False
                
        # now that we have the x's, define the residual functions
                
        def residual_full(q, alpha, func="exp"): 
            p = target_p(xs[:,0], alpha, E0, R)
            if func == "exp":
                # q = p*exp(r)
                return torch.log((p+epsilon)/(q + epsilon) )
            elif func == "ratio":
                # q = p*(1+r)
                return 1.0 - (p/q)
            elif func == "sum":
                # q = p + r
                return q - p
            
        def q_given_alpha(xs, alpha):
            logJ = torch.nansum(log_abs_det_jacobian_sigmoid(inverse_sigmoid(xs)), axis = 1)
            logq = flow(alpha).log_prob(inverse_sigmoid(xs)) +  logJ
            return torch.exp(logq)
            #return LL_angularity(xs[:,0], alpha, E0, R)
        
        def residual_func(input_alpha):
            loc_q = q_given_alpha(xs, input_alpha) 
            return residual_full(loc_q, input_alpha)
        

        # CALCULATE RESIDUAL LOSS
        if train_residual:

            # order 0
            q = q_given_alpha(xs, torch.tensor([0.0])) 
            order_0_loss = (residual_full(q, torch.tensor([0.0]))**2).reshape(-1,1)

            # order 1
            dResidual_dAlpha = torch.autograd.functional.jacobian(residual_func, torch.tensor([0.0]), create_graph=True)
            order_1_loss = dResidual_dAlpha**2

            loss_residual = torch.mean(order_0_loss + order_1_loss) 
            loss += loss_residual
        
        # CALCULATE MSE LOSS
        if train_mse:
            alpha_rand = torch.tensor([0.1])
            print(C_prescale(xs[:,0],alpha_rand, E0, R, args.C))
            print(target_p(xs[:,0], alpha_rand, E0, R))
            print(q_given_alpha(xs, alpha_rand))
            loss_mse = torch.nanmean((torch.log(target_p(xs[:,0], alpha_rand, E0, R)) - torch.log(q_given_alpha(xs, alpha_rand)))**2 )
            
            loss += loss_mse


            


        if DEBUG:
            logps.append(logp)
            logqs.append(logq)
            logJs.append(logJ)
            xss.append(xs)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(flow.parameters(), 1)


    # Gradient Descent
    optimizer.step()
    optimizer.zero_grad()

    losses.append(loss.item())
    t.set_description(f"Loss: {loss.item()}")
    

# save out the model
torch.save(flow.state_dict(), f"{args.outdir}/{args.run_id}_model")



fig, ax = plt.subplots(1,1)
losses = np.array(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
ax.scatter(np.arange(len(losses)), losses)
plt.yscale("log")
plt.title(args.run_id)
plt.savefig(f"{args.outdir}/{args.run_id}_loss")


"""
PLOTTING
"""

alpha_s_plot = 0.118

# Plot distribution of xs at a few values of c

fig, ax = plt.subplots(1,1)
cs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999][::-1]

DRAW_NF_HIST = False
DRAW_NF_LINES = True
aux_draws = 5

fig, ax = plt.subplots(1,1)
for i in range(len(cs)):

    # Draw hist
    c = torch.tensor((cs[i],))

    if DRAW_NF_HIST:
        samples = sigmoid(flow(c).sample((100000,)))
        x_samples = samples[:,0].detach().numpy()
        aux_samples = samples[:,1:].detach().numpy()

        jet_color = plt.get_cmap("jet")(  i / 10)
        if i == 0:
            label = "Generated Samples"
        else:
            label = None
        ax.hist(x_samples, bins=100, density=True, color = jet_color, alpha = 0.5, label = label, histtype="step")

    if DRAW_NF_LINES:

        for aux_draw in range(aux_draws):
            xs = torch.rand(10000, auxiliary_params + 1) * (x_range[1] - x_range[0]) + x_range[0]
            
            # Sort the first column
            xs = xs[torch.argsort(xs[:,0])]
            xs[:,1:] = torch.rand(1) * torch.ones((10000, auxiliary_params))
            

            logJ = torch.sum(log_abs_det_jacobian_sigmoid(inverse_sigmoid(xs)), axis = 1)
            ys = (flow(c).log_prob(inverse_sigmoid(xs)) + logJ).exp()

            xs = xs[:,0].detach().numpy().flatten()
            ys = np.nan_to_num(ys.detach().numpy().flatten())

            jet_color = plt.get_cmap("jet")( i / 10)
            if i == 0 and aux_draw == 0:
                label = "Learned Flow"
            else:
                label = None

            ax.plot(xs, ys, color = jet_color, lw = 1, alpha = 0.5, label = label)

            ax.vlines(cs[i], 0, target_p(c, alpha_s_plot, E0, R).detach().numpy(), color = jet_color, linestyle = "--", alpha = 0.25, lw = 0.5)

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("jet_r"), norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
plt.colorbar(sm, label="Cutoff")
cutoff = xs > 0.5

plt.plot(xs, target_p(torch.tensor(xs), alpha_s_plot, E0, R).detach().numpy(), color = "Black", label = "Target (Uncut)")
plt.plot(xs, LL_angularity(torch.tensor(xs), alpha_s_plot, E0, R), color = "black", linestyle = "--", label = "LL' Angularity", alpha = 0.25)
plt.plot(xs, LL_angularity(torch.tensor(xs), alpha_s_plot, E0, R), color = "black", linestyle = "--", label = "LL-exact Angularity")

plt.legend()
plt.yscale("log")
plt.ylim(1e-3, 1e3)
plt.title(args.run_id)
plt.title(args.run_id)
plt.savefig(f"{args.outdir}/{args.run_id}_curves")



# Hist the aux variable
fig, ax = plt.subplots(1,1)
c = torch.tensor((0.5,))
samples = sigmoid(flow(c).sample((100000,)))
x_samples = samples[:,0].detach().numpy()
aux_samples = samples[:,1:].detach().numpy()

if auxiliary_params > 0:
    fig, ax = plt.subplots(1,1)
    for i in range(auxiliary_params):
        plt.hist(aux_samples[:,i], bins=100, density=True, alpha = 0.5, label=f"Aux {i}")
        prob = flow(c).log_prob(samples).exp()
        # plt.scatter(aux_samples[:,i], prob.detach().numpy(), color = "red")
    plt.legend()
plt.title(args.run_id)
plt.savefig(f"{args.outdir}/{args.run_id}_aux_hist")

fig, ax = plt.subplots(1,1)

# Correlation between auxiliary variable 1 and x

fig, ax = plt.subplots(1,1)
bar = ax.hist2d(x_samples, aux_samples[:,0], bins=100, density=True, norm=mpl.colors.LogNorm(), cmap="Reds")
plt.colorbar(bar[3])
plt.xlabel("x")
plt.ylabel("Aux 0")
plt.title(args.run_id)
plt.savefig(f"{args.outdir}/{args.run_id}_aux_x_corr")
