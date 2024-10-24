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
parser.add_argument("-loss", "--loss", default="logMSE", help="choose from FORWARD, REVERSE, logMSE, ratioMSE")
parser.add_argument("-debug", "--debug", action="store_true")

# hyperparameters
parser.add_argument("-epochs", "--epochs", default=200)
parser.add_argument("-nx", "--batch_num_x", default=128)
parser.add_argument("-nc", "--batch_num_c", default=8)
parser.add_argument("-lr", "--lr", default=1e-3)
parser.add_argument("-seed", "--seed", default=1)


# network architecture
parser.add_argument("-aux", "--auxiliary_params", default=1)
parser.add_argument("-nt", "--num_transforms", default=5)
parser.add_argument("-hf", "--hidden_features", default="32,32")

args = parser.parse_args()

run_params = {"hyperparams":{}, "architecture":{}, "physics":{}}


# Physics Parameters
E0 = 500
R = 0.4
target_p = LO_angularity
run_params["physics"]["E0"] = E0
run_params["physics"]["R"] = R
run_params["physics"]["target_p"] = str(target_p)


MODE = args.mode
LOSS = args.loss
DEBUG = args.debug
run_params["MODE"] = MODE
run_params["LOSS"] = LOSS

epochs = args.epochs
batch_num_x = args.batch_num_x
batch_num_c = args.batch_num_c
batch_size = batch_num_x*batch_num_c
lr = args.lr

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

    # Generate some random cutoffs c
    c = torch.rand(batch_num_c) * (x_range[1] - x_range[0]) + x_range[0]
    # generate a c tensor the same length as the x tensor
    c_expanded = c.repeat_interleave(batch_num_x)

    if MODE == "GENERATED_SAMPLES":

        xs = sigmoid(flow(c).sample((batch_size,)))
        logJ = torch.sum(log_abs_det_jacobian_sigmoid(inverse_sigmoid(xs)), axis = 1)
        #logq = flow(c).log_prob(inverse_sigmoid(xs)) + logJ
        logp = torch.nan_to_num(torch.log(target_p(xs, E0, R))[:,0])
        
        logq_list = []
        for ind, loc_c in enumerate(c):  
            logq_c = flow(loc_c.reshape(1,1)).log_prob(inverse_sigmoid(xs[ind*batch_num_x:(ind+1)*batch_num_x]))
            logq_list.append(logq_c)
        logq = torch.concatenate(logq_list)
        logq += logJ

        if LOSS == "FORWARD":
            raise NotImplementedError("Impossible to do a forward KL loss with generated samples!")

        # Can only do reverse
        loss = torch.nanmean(Theta(xs[:,0] - c_expanded) * (logq - logp))
    

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

        # Note for log_prob to work, we need to pass a tensor of shape (batch_size, >=1)
        logJ = torch.nansum(log_abs_det_jacobian_sigmoid(inverse_sigmoid(xs)), axis = 1)
        logp = torch.nan_to_num(torch.log(target_p(xs, E0, R))[:,0])
        #logq_c = torch.nan_to_num(((flow(c).log_prob(inverse_sigmoid(xs)) + logJ)))
        
        # tile over the different choices of c
        logq_list = []
        for ind, loc_c in enumerate(c):  
            logq_c = torch.nan_to_num(flow(loc_c.reshape(1,1)).log_prob(inverse_sigmoid(xs[ind*batch_num_x:(ind+1)*batch_num_x])))
            logq_list.append(logq_c)
        logq = torch.concatenate(logq_list)
        logq += logJ
                
        # technically need to adjust logp if aux is anything other than U(0,1)

        # Bou
        allowed_error = counting_parameter(xs[:,0], E0, C = 1) ** 1
        # loss = torch.nanmean(Theta(xs[:,0] - c)  * (logp - logq)**2 ) 

        # For forward, do plog(p/q), for reverse do qlog(q/p)

        norm_p = torch.nanmean(Theta(xs[:,0] - c_expanded) * torch.exp(logp)) 
        norm_q = torch.nanmean(Theta(xs[:,0] - c_expanded) * torch.exp(logq)) 

        logp_prime = logp - torch.nan_to_num(1*torch.log(norm_p))
        logq_prime = logq - torch.nan_to_num(1*torch.log(norm_q))


        if LOSS == "FORWARD":
            loss = torch.nanmean(Theta(xs[:,0] - c_expanded) * torch.exp(logp_prime) * (logp_prime - logq_prime ))

        if LOSS == "REVERSE":
            loss = torch.nanmean(Theta(xs[:,0] - c_expanded) * torch.exp(logq_prime) * (logq_prime - logp_prime ))

        if LOSS == "logMSE":
            C = Theta(xs[:,0] - c_expanded)
            # C = allowed_error / torch.nanmean(allowed_error) 
            #loss = torch.mean(C * (logp - logq)**2 / c_expanded)
            loss = torch.mean(C * (logp - logq)**2 )

        if LOSS == "ratioMSE":
            C = Theta(xs[:,0] - c_expanded)
            C = allowed_error 
            loss = torch.nanmean(C * (1 - torch.exp(logq) / torch.exp(logp))**2 )

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


    training_cs.append(c)
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
plt.savefig(f"{args.outdir}/{args.run_id}_loss")


"""
PLOTTING
"""

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

            ax.vlines(cs[i], 0, target_p(c, E0, R), color = jet_color, linestyle = "--", alpha = 0.25, lw = 0.5)

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("jet_r"), norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
plt.colorbar(sm, label="Cutoff")
cutoff = xs > 0.5

plt.plot(xs, target_p(torch.tensor(xs), E0, R), color = "Black", label = "Target (Uncut)")
plt.plot(xs, LL_angularity(torch.tensor(xs), E0, R), color = "black", linestyle = "--", label = "LL' Angularity", alpha = 0.25)
plt.plot(xs, LL_angularity(torch.tensor(xs), E0, R), color = "black", linestyle = "--", label = "LL-exact Angularity")

plt.legend()
plt.yscale("log")
plt.ylim(1e-3, 1e3)
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
plt.savefig(f"{args.outdir}/{args.run_id}_aux_hist")

fig, ax = plt.subplots(1,1)

# Correlation between auxiliary variable 1 and x

fig, ax = plt.subplots(1,1)
bar = ax.hist2d(x_samples, aux_samples[:,0], bins=100, density=True, norm=mpl.colors.LogNorm(), cmap="Reds")
plt.colorbar(bar[3])
plt.xlabel("x")
plt.ylabel("Aux 0")

plt.savefig(f"{args.outdir}/{args.run_id}_aux_x_corr")
