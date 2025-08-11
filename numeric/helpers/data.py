import numpy as np
import torch
import pickle
import jax.numpy as jnp


N_C = 3
N_F = 5

C_F = (N_C**2 - 1) / (2 * N_C) # Quark Casimir, 4/3
C_A = N_C # Gluon Casimir, 3
T_F = 1/2 

zeta_3 = 1.20205690315959

MZ = 91.1876 # GeV
MW = 80.379 # GeV
MH = 125.18 # GeV
MT = 173.21 # GeV


alpha_s_MZ = 0.1181


beta0 = (11 * N_C - 4 * T_F * N_F) / 3
beta1 = (34 * N_C**2 - 20 * C_A * T_F * N_F - 12 * C_F * T_F * N_F ) / 3


def running_coupling(alpha_0, mu0, mu):
    """
    Running coupling constant at scale mu, given the coupling at scale mu0.
    """
    return alpha_0 / (1 + beta0 * alpha_0 * torch.log(mu / mu0) / (2 * np.pi))


def get_pdf_toy(alpha, example, tt, order, device):
    alpha = torch.as_tensor(alpha, device=device)[..., None]
    if example == "exponential":
        if order == -1:
            y = alpha * torch.exp(-alpha * tt)
        elif order == 1:
            y = alpha.expand_as(alpha * tt) * 0 + alpha
        elif order == 2:
            y = alpha * (1 - alpha * tt)
    elif example == "angularity":
        if order == -1:
            y = alpha * tt * torch.exp(-alpha * tt**2 / 2)
        elif order == 1:
            y = alpha * tt
        elif order == 2:
            y = alpha * tt * (1 - alpha * tt**2 / 2)
    elif example == "harder_exp":
        if order == 1:
            y = alpha.expand_as(alpha * tt) * 0 + alpha
        elif order == 2:
            y = alpha * (1 - alpha * tt)
        elif order == -1:
            y = (alpha + 2*alpha**2*tt) * torch.exp(-alpha*tt - alpha**2 *tt**2)

        
    elif example == "LO_thrust":

        if order == 1:

            term_1 = (torch.exp(-tt) * (3 - 6 * torch.exp(tt) + 8 * torch.exp(2 * tt)) * torch.log(2 * (-1 + torch.exp(tt)))) / (-1 + 2 * torch.exp(tt))
            term_2 =  (9 * torch.exp(-2 * tt)) / 4 - 3 * torch.exp(-tt)
            y = (alpha * 2 / 3 / np.pi) * (term_1 - term_2)
            y = torch.nan_to_num(y)
            y[(y < 0) & (tt < 5)] = 0 # approximate the theta function on the LHS
          

        elif order == 2:

            # 2nd order with lots of approximations

            def DA(t):
                return(-3 + 4 * t + np.log(16)) * C_F

            def DB(t):
            
                term1 =(1/4) * (9 - 8 * np.pi**2 - 52 * t + 16 * zeta_3 - 52 * np.log(2) +
                16 * np.pi**2 * (t + np.log(2)) +
                72 * (t + np.log(2))**2 -
                32 * (t + np.log(2))**3) * C_F**2
                term2 = (1/9) * (45 - 22 * (t + np.log(2)) - 36 * (t + np.log(2))**2) * C_F * N_F * T_F

                term3 = (1/36) * (-513 + 216 * zeta_3 + 338 * (t + np.log(2)) -
                  24 * np.pi**2 * (t + np.log(2)) +
                  396 * (t + np.log(2))**2) * C_A * C_F
                
                return term1 + term2 + term3

            tt_exp = tt.unsqueeze(0).expand(alpha.shape[0], tt.shape[0])
            alpha_exp = alpha.expand(alpha.shape[0], tt.shape[0])

            x_exp = torch.exp(-tt_exp)
            scale = 91.1876 * x_exp
            alpha_s = alpha_exp#running_coupling(alpha_exp, MZ, scale)

            DA_exp = DA(tt_exp)
            DB_exp = DB(tt_exp)

            alpha_bar = alpha_s / (2 * np.pi)
            y = alpha_bar * DA_exp + alpha_bar**2 * DB_exp       
           
            y[(y < 0) & (tt < 5)] = 0 # approximate the theta function on the LHS

        elif order == -1:

            t_adjusted = tt - torch.log(torch.tensor([2.0], device = device))

            y = (
                alpha
                * (4 / 3)
                * (1.0 / (2 * np.pi))
                * (4 * t_adjusted - 3)
                * torch.exp(
                    -alpha
                    * (4 / 3)
                    * (1.0 / (2 * np.pi))
                    * (2 * torch.pow(t_adjusted, 2) - 3 * t_adjusted)
                )
            )
            y[y < 0] = 0

    else:
        raise ValueError("bad example/order")
    # y[y < 0] = 0
    return y.squeeze(0)


def read_in_data(distribution, order, device, space="t"):

    if order == 1:
        order_key = "LO"
    elif order == 2:
        order_key = "NLO"

    if space == "t":
        if distribution == "thrust":
            path_to_data = "data/thrust_data.pkl"
        elif distribution == "c_param":
            path_to_data = "data/c_param_data.pkl"
    elif space == "x":
        if distribution == "thrust":
            path_to_data = "data/LINEAR_thrust_data.pkl"
        elif distribution == "c_param":
            path_to_data = "data/LINEAR_c_param_data.pkl"
        
    data_dict = {}

    with open(path_to_data, "rb") as ifile:
        loc_data_dict = pickle.load(ifile)
        for alpha in loc_data_dict.keys():

            y_data = loc_data_dict[alpha][f"values_{order_key}"]
            y_data = torch.tensor(y_data, device=device).reshape(-1, 1)

            y_err = loc_data_dict[alpha][f"mcerr_{order_key}"]
            y_err = torch.tensor(y_err, device=device).reshape(-1, 1)

            data_dict[float(alpha)*1e-3] = y_data, y_err

            bin_centers = torch.tensor(loc_data_dict[alpha]["bin_centers"], device=device).reshape(-1, )
            bin_edges = np.concatenate([loc_data_dict[alpha]["bin_lows"], loc_data_dict[alpha]["bin_highs"][-1].reshape(-1,)])
            bin_edges = torch.tensor(bin_edges, device=device).reshape(-1, )


    return data_dict, bin_edges, bin_centers


def read_in_data_JAX(distribution, order):

    if order == 1:
        order_key = "LO"
    elif order == 2:
        order_key = "NLO"

    if distribution == "thrust":
        path_to_data = "data/thrust_data.pkl"
    elif distribution == "c_param":
        path_to_data = "data/c_param_data.pkl"
        
    data_dict = {}

    with open(path_to_data, "rb") as ifile:
        loc_data_dict = pickle.load(ifile)
        for alpha in loc_data_dict.keys():

            y_data = loc_data_dict[alpha][f"values_{order_key}"]
            y_data = jnp.array(y_data).reshape(-1, 1)

            y_err = loc_data_dict[alpha][f"mcerr_{order_key}"]
            y_err = jnp.array(y_err).reshape(-1, 1)

            data_dict[float(alpha)*1e-3] = y_data, y_err

            bin_centers = jnp.array(loc_data_dict[alpha]["bin_centers"]).reshape(-1, )
            bin_edges = jnp.concatenate([loc_data_dict[alpha]["bin_lows"], loc_data_dict[alpha]["bin_highs"][-1].reshape(-1,)])
            bin_edges = jnp.array(bin_edges).reshape(-1, )


    return data_dict, bin_edges, bin_centers