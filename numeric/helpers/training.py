import numpy as np
import torch
from helpers.ansatz import q, get_taylor_expanded_ansatz
from helpers.data import get_pdf_toy

from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from tqdm import tqdm


MSE_criterion = torch.nn.MSELoss()

def get_loss(order_to_match, distribution, batch_size, 
             g_coeffs_to_fit, theta_to_fit, mstar, t_bin_centers, device,
             run_toy, mse_weighted_loss, data_dict, ratio_loss=False):

    alpha_zero = torch.tensor([1e-12], device=device, requires_grad=True)

    if run_toy:
        loc_alphas = (torch.distributions.Exponential(1 / 0.118).sample((batch_size,)).to(device))  # (B,)
        loc_alphas = torch.clamp(loc_alphas, min=0, max=4 * np.pi)

        #loc_alphas = ( torch.distributions.uniform.Uniform(low=1e-8, high=1).sample((batch_size,)) .to(device))
        #loc_alphas = 1.0 - torch.sqrt(1.0 - loc_alphas)

        # get the data pdf for the sampled loc_alphas for the entire batch
        batch_data_pdf = get_pdf_toy(loc_alphas, distribution,t_bin_centers, order_to_match, device)  # (B, args.n_bins-1)

    else:
        loc_alphas_keys = np.random.choice(list(data_dict.keys()), size=batch_size, replace=False)
        loc_alphas = torch.tensor([a for a in loc_alphas_keys]).to(device).reshape(-1, )
        batch_data_pdf = torch.cat([data_dict[a][0] for a in loc_alphas_keys], axis=1).T
        batch_errors_pdf = torch.cat([data_dict[a][1] for a in loc_alphas_keys], axis=1).T

    fn = lambda a: q(
            t_bin_centers, a, g_coeffs_to_fit, theta_to_fit, mstar, device
        )
    
    batch_ansatz = get_taylor_expanded_ansatz(fn, alpha_zero, loc_alphas, order_to_match)
    
    
    batch_data_pdf = batch_data_pdf.reshape(-1).double()
    batch_ansatz = batch_ansatz.reshape(-1).double()


    if mse_weighted_loss:
                batch_errors_pdf = batch_errors_pdf.reshape(-1).double()

                # Get minimum error
                min_error = torch.min(batch_errors_pdf[batch_errors_pdf > 0])

                # clip from below
                batch_errors_pdf = torch.clamp(batch_errors_pdf, min=min_error)                

                rescaled_pdf = batch_data_pdf
                rescaled_ansatz = batch_ansatz
                rescaled_errors = batch_errors_pdf / torch.mean(batch_errors_pdf)

                loss = torch.mean(torch.pow(rescaled_pdf-rescaled_ansatz, 2)/torch.pow(rescaled_errors, 2)) / 2
    elif ratio_loss:
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
        loss = MSE_criterion(batch_data_pdf,batch_ansatz)


    return loss


def train(g_coeffs_to_fit, theta_to_fit, order_to_match, distribution, mstar, t_bin_centers, data_dict, epochs, batch_size, lr, wd, learn_theta, run_toy, weighted_mse_loss, device):

    if learn_theta:
        optimizer = torch.optim.AdamW([g_coeffs_to_fit, theta_to_fit], lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW([g_coeffs_to_fit], lr=lr, weight_decay=wd)

    scheduler = ExponentialLR(optimizer, gamma=1)

    

    losses = np.zeros((epochs, 1))
    lrs = np.zeros((epochs, 1))
    g_coeffs_log = np.zeros((epochs + 1, *g_coeffs_to_fit.shape))
    g_coeffs_log[0] = g_coeffs_to_fit.detach().cpu().numpy()

    theta_log = np.zeros((epochs + 1, *theta_to_fit.shape))
    theta_log[0] = theta_to_fit.detach().cpu().numpy()

    alpha_zero = torch.tensor([1e-12], device=device, requires_grad=True)

    for epoch in tqdm(range(epochs)):

        optimizer.zero_grad()

        loss = get_loss(order_to_match, distribution, batch_size, 
                     g_coeffs_to_fit, theta_to_fit, mstar, t_bin_centers, device,
                    run_toy, weighted_mse_loss, data_dict)

        loss.backward()

        # this strictly makes things worse??
        # torch.nn.utils.clip_grad_norm_(g_coeffs_to_fit, max_norm = 2.0)

        optimizer.step()

        scheduler.step()

        losses[epoch] = loss.detach().cpu().numpy()
        lrs[epoch] = scheduler.get_lr()[0]
        g_coeffs_log[epoch + 1] = g_coeffs_to_fit.detach().cpu().numpy()
        theta_log[epoch + 1] = theta_to_fit.detach().cpu().numpy()

    return losses, lrs, g_coeffs_log, theta_log