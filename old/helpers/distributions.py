import numpy as np
import torch

"""
"
"
GENERIC
"
"
"""


lambda_qcd = 0.2
beta_0 = 11 - 2/3 * 3

def Gaussian(x):

    mean = 0
    std = 1

    # Gaussian 
    p = 1/np.sqrt(2 * np.pi  * std**2) * np.exp(- (x - mean)**2 / 2 / std**2)
    return p

def Uniform(x): 

    return Theta(x) * Theta(1 - x)


def Theta(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

def alpha_s(scale):
    return 4 * np.pi / (beta_0 * torch.log(scale**2 / lambda_qcd**2))

def dalpha_dscale(scale):
    return -4 * np.pi / beta_0  / (torch.log(scale**2 / lambda_qcd**2) ** 2) * 2 / scale





"""
"
"
ANGULARITIES
"
"
"""

def LO_angularity(lambda_, E0, R, beta = 1):

    scale = E0 * torch.pow(lambda_, 1 / (beta))

    alpha_s_scale = alpha_s(scale)
    C_F = 4/3
    C_A = 3

    return -torch.nan_to_num(((2 * alpha_s_scale * C_F / (1 * np.pi * beta * R)) * torch.log(lambda_ ) / lambda_ * Theta(1 - lambda_)) )* Theta(lambda_)* Theta(1-lambda_)



def df_dx(lambda_, E0, R, beta = 1):

    lambda_qcd = 0.2
    beta_0 = 11 - 2/3 * 3

    scale = E0 * torch.pow(lambda_, 1 / (beta))
    alpha_s_scale = alpha_s(scale)

    C_F = 4/3


    dLO_dalpha =  -C_F / (beta * R *  np.pi) * torch.pow(torch.log(lambda_), 2)
    dalpha_dscale = -4 * np.pi / beta_0  / (torch.log(scale**2 / lambda_qcd**2) ** 2) * 2 / scale
    dscale_dlambda = E0

    return dLO_dalpha * dalpha_dscale * dscale_dlambda

   

def LL_angularity(lambda_, E0, R, beta = 1):

    scale = E0 * torch.pow(lambda_, 1 / (beta))

    alpha_s_scale = alpha_s(scale)
    C_F = 4/3
    C_A = 3

    p = torch.nan_to_num(LO_angularity(lambda_, E0, R, beta) * torch.exp(-1 * alpha_s_scale * C_F / (beta * R *  np.pi) * torch.pow(torch.log(lambda_), 2)) )

    return (p * Theta(lambda_) * Theta(1 - lambda_))



def LL_exact_angularity(lambda_, E0, R, beta = 1):

    scale = E0 * torch.pow(lambda_, 1 / (beta))

    alpha_s_scale = alpha_s(scale)
    C_F = 4/3
    C_A = 3

    p = torch.nan_to_num((LO_angularity(lambda_, E0, R, beta) + df_dx(lambda_, E0, R, beta)) * torch.exp(-1 * alpha_s_scale * C_F / (beta * R *  np.pi) * torch.pow(torch.log(lambda_), 2)) )

    return  (p * Theta(lambda_) * Theta(1 - lambda_)) * 1


def counting_parameter(x, E0, C = 1):
    return C * alpha_s(E0 * x) * torch.log(1/x) / x 


"""
"
"
C(x,c)
"
"
"""
def C_theta(x,c,E0,R):
    return Theta(x-c)

def C_alpha_1(x,c,E0,R):
    soft_cutoff = beta_0*torch.log((x*E0)**2/ lambda_qcd**2)
    return 1.0 / (alpha_s(x*E0)**2 + soft_cutoff*alpha_s(x*E0)**3)

def C_alpha_2(x,c,E0,R):
    soft_cutoff = beta_0*torch.log((x*E0)**2/ lambda_qcd**2)
    return 1.0 / (alpha_s(x*E0) + soft_cutoff*alpha_s(x*E0)**2)**2

def C_alpha_log_1(x,c,E0,R):
    soft_cutoff = beta_0*x*(torch.log(1.0/x))*torch.log((x*E0)**2/ lambda_qcd**2)
    return 1.0 / ((alpha_s(x*E0)*torch.log(1.0/x)/x)**2 + soft_cutoff*(alpha_s(x*E0)*torch.log(1.0/x)/x)**3)

def C_alpha_log_2(x,c,E0,R):
    soft_cutoff = beta_0*x*(torch.log(1.0/x))*torch.log((x*E0)**2/ lambda_qcd**2)
    return 1.0 / ((alpha_s(x*E0)*torch.log(1.0/x)/x) + soft_cutoff*(alpha_s(x*E0)*torch.log(1.0/x)/x)**2)**2

def C_unscaled(x,c,E0,R):
    return c/alpha_s(x*E0)

def C_theory(x,c,E0,R):
    R_x = LL_exact_angularity(x, E0, R)/LO_angularity(x, E0, R)
    return LO_angularity(x, E0, R)*R_x/(2*torch.log(R_x))

