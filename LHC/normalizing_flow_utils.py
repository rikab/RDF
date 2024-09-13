import base64
import io
import pickle
import numpy as np
import math
import torch
print(f'TORCH VERSION: {torch.__version__}')
import packaging.version
if packaging.version.parse(torch.__version__) < packaging.version.parse('1.5.0'):
    raise RuntimeError('Torch versions lower than 1.5.0 not supported')
import matplotlib.pyplot as plt

from tqdm import tqdm


if torch.cuda.is_available():
    torch_device = 'cuda'
    float_dtype = np.float32 # single
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch_device = 'cpu'
    float_dtype = np.float64 # double
    torch.set_default_tensor_type(torch.DoubleTensor)
    print(f"TORCH DEVICE: {torch_device}")


# Function to do torch tensor to numpy conversion
def grab(var):
    """ Function to do torch tensor to numpy conversion

    Args:
        var (torch.Tensor): torch tensor to be converted to numpy

    Returns:
        np.array: numpy array
    """
    return var.detach().cpu().numpy()


# ########################################
# ########## Prior Distribution ########## 
# ########################################

class SimpleNormal:
    def __init__(self, loc, var):
        self.dist = torch.distributions.normal.Normal(
        torch.flatten(loc), torch.flatten(var))
        self.shape = loc.shape
    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return logp
    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size,))
        return x.reshape(batch_size, *self.shape)
    

# ######################################
# ########## Normalizing Flow ##########
# ######################################

import torch
import torch.nn as nn
from torch.distributions.normal import Normal 

class Flow1d(nn.Module):
    def __init__(self, n_components):
        super(Flow1d, self).__init__()
        self.mus = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(torch.ones(n_components), requires_grad=True)

    def forward(self, x):
        x = x.view(-1,1)
        weights = self.weight_logits.softmax(dim=0).view(1,-1)
        distribution = Normal(self.mus, self.log_sigmas.exp())
        z = (distribution.cdf(x) * weights).sum(dim=1)
        log_dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1).log()
        return z, log_dz_by_dx

class LogitTransform(nn.Module):
    def __init__(self, alpha):
        super(LogitTransform, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        x_new = (self.alpha/2 + (1-self.alpha)*x).to(torch_device)
        z = torch.log(x_new) - torch.log(1-x_new)
        log_dz_by_dx = torch.log(torch.FloatTensor([1-self.alpha]).to(torch_device)) - torch.log(x_new) - torch.log(1-x_new)
        return z, log_dz_by_dx
        

class FlowComposable1d(nn.Module):
    def __init__(self, flow_models_list):
        super(FlowComposable1d, self).__init__()
        self.flow_models_list = nn.ModuleList(flow_models_list).to(torch_device)

    def forward(self, x):
        z, sum_log_dz_by_dx = x, 0
        for flow in self.flow_models_list:
            z, log_dz_by_dx = flow(z)
            sum_log_dz_by_dx += log_dz_by_dx
        return z, sum_log_dz_by_dx
    



def initialize_model(num_points, lr):

    # Initialize the prior
    prior = SimpleNormal(torch.zeros(num_points), torch.ones(num_points))

    # Initialize the model
    # Model archutecture not optimized at all
    flow_models_list = [Flow1d(2), LogitTransform(0.1), Flow1d(2), LogitTransform(0.1), Flow1d(2)]
    flow = FlowComposable1d(flow_models_list)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    return flow, optimizer, prior


# ########## Training Loop ##########

def train_model(target_function, loss_function, num_epochs, num_points, batch_size, lr):
    """ Function to train the normalizing flow model to approximate the target distribution 

    Args:
        target_function (function): The target distribution to be approximated
        loss_function (function): The loss function to be used, as a function f(logp, logq)
        num_epochs (int): Number of epochs to train the model
        num_points (int): Number of points to be used in the flow
        batch_size (int): Batch size to be used in the training
        lr (float): Learning rate to be used in the training

    Returns:
        FlowComposable1d: Trained model
        list: List of training losses
    """

    train_losses = []
    flow, optimizer, prior = initialize_model(num_points, lr)


    for epoch in tqdm(range(num_epochs)):
    
    
        optimizer.zero_grad()
        
        # APPLY FLOW TO PRIOR
        # draw samples from prior
        z = prior.sample_n(batch_size).to(torch_device) # shape x: (batch_size, num_points)
        logq = prior.log_prob(z) # shape logq: (batch_size, num_points)
        x, logJ = flow(z)
        logJ = logJ.reshape(batch_size, num_points)

        x = x.reshape(batch_size, num_points) 
        mask = 1
        
        # need to reshape the outputs match those from the previous step
        logq = logq - logJ 

        # CALCULATE THE TARGET
        log_target = torch.log(target_function(x))
        logp = log_target

        # Mean over the points to get the total logp and logq
        regulator = 1
        total_logq = torch.nanmean(mask * logq, dim=1) #/ (mask.sum() + regulator)
        total_logJ = torch.nanmean(mask * logJ, dim=1) #/ (mask.sum() + regulator)
        total_logp = torch.nanmean(mask * logp, dim=1) #/ (mask.sum() + regulator)

        # We DON'T want to sum, we want to mean in the loss, since the effective batch size can change

            
        # CALCULATE THE LOSS
        loss = loss_function(total_logp, total_logq)
        
        loss.backward()
        optimizer.step()
            
        train_losses.append(grab(loss))


    return flow, train_losses





# #####################################
#  ########## LOSS FUNCTIONS ##########
# #####################################

def calc_dkl(logp, logq):
    return (torch.nan_to_num(logq) - torch.nan_to_num(logp)).sum() # reverse KL, assuming samples from q

def calc_MSE(logp, logq):

    p = torch.exp(logp)
    q = torch.exp(logq)

    return 0.5 * torch.mean((p - q)**2)

def calc_logMSE(logp, logq):

    return 0.5 * torch.mean((logp - logq)**2)



