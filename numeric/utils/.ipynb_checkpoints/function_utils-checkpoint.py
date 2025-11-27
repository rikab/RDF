import numpy as np
import jax.numpy as jnp
import jax
import math
from jax.scipy.signal import convolve2d
from functools import partial

# #################################
# ########## POLYNOMIALS ##########
# #################################




@partial(jax.jit, static_argnames=("length",))
def build_powers(base, length):
    
    base = jnp.asarray(base, dtype=jnp.float32)
    i = jnp.arange(length, dtype=base.dtype)
    step = jnp.where(i == 0, jnp.ones((), base.dtype), base / i)  # [1, b/1, b/2, ...]
    return jnp.cumprod(step)

# @jax.jit
def Theta(t, temps):

    # temp = 400
    # print(temps)
    return jax.nn.sigmoid(t * 100 / (temps + 1e-6))

# @jax.jit
def ReLU(x, temps):


    return 2*Theta(x, temps)*x -x

    # return jnp.abs(x)
    temp = 100
    # return jnp.log(1 + jnp.exp(temp * x)) / temp
    return jax.nn.relu(x) + 1 * jax.nn.relu(-x) #- 0.001*x
    # return jax.nn.softplus(temp * x)/temp #* jnp.log(10)
    return x
    # return x * ((x > 0)) #+ 1e-12

# @jax.jit
def polynomial(t, alpha, params, thetas, temps):

    M, N = params.shape
    
    # Build powers of alpha: [1, alpha, alpha^2, ... alpha^(M-1)] (including factorials)
    alpha_powers = build_powers(alpha, M)  # shape (M,)
    alpha_powers = alpha_powers * Theta(t - thetas, temps) 

    # Build powers of t: [1, t, t^2, ... t^(N-1)]
    t_powers = build_powers(t, N)         # shape (N,)(including factorials)

    poly_val = alpha_powers @ params @ t_powers
    return poly_val


# @jax.jit
def relu_polynomial(t, alpha, params, thetas, temps, temps2):

    M, N = params.shape
    
    # Build powers of alpha: [1, alpha, alpha^2, ... alpha^(M-1)] (including factorials)
    alpha_powers = build_powers(alpha, M)  # shape (M,)
    alpha_powers = alpha_powers * Theta(t - thetas, temps) 

    # Build powers of t: [1, t, t^2, ... t^(N-1)]
    t_powers = build_powers(t, N)         # shape (N,)(including factorials)

    poly_val = alpha_powers @ ReLU(params @ t_powers, temps2)
    return poly_val

# Multiply two 2D polynomials using their coefficient arrays
def polynomial_multiply(c1, c2, M = None, N = None):

    if M is None:
        M = c1.shape[0] + c2.shape[0]
    if N is None:
        N = c1.shape[1] + c2.shape[1]

    full_conv = convolve2d(c1, c2, mode='full')
    truncated = full_conv
    return truncated



@jax.jit
def polynomial_sum(cs):

    # zero pad to the same size
    max_M = max([c.shape[0] for c in cs])
    max_N = max([c.shape[1] for c in cs])

    for i in range(len(cs)):
        M = cs[i].shape[0]
        N = cs[i].shape[1]
        cs[i] = jnp.pad(cs[i], ((0, max_M - M), (0, max_N - N)), mode='constant')

    cs = jnp.array(cs)
    return jnp.sum(cs, axis=0)


def polynomial_power(c, k):
    
    if k == 0:
        return jnp.array([[1,]])
    elif k == 1:
        return c
    else:
        return polynomial_multiply(c, polynomial_power(c, k-1))


# Debug Tool
def print_polynomial(c):

    # Print the polynomial in a human-readable grid.
    M, N = c.shape
    for m in range(M):
        s = ""
        for n in range(N):
            s += f"c_{m},{n} = {c[m, n]:.2e} "
        print(s)

            

def latex_polynomial(c, eps = 1e-6):


    # Print the polynomial in a latex-readable grid using pmatrix
    M, N = c.shape
    s = "\\begin{pmatrix}\n"
    for m in range(M):
        for n in range(N):

            # Check if the coefficient is close to an integer
            cmn = c[m, n]
            if cmn - eps <= round(cmn) <= cmn + eps:
                cmn = round(cmn)
                s+= f"{int(cmn)} "
            else:
                s += f"{c[m, n]:.2e} "
            if n < N - 1:
                s += "& "
        if m < M - 1:
            s += "\\\\ \n"
        else:
            s += "\\end{pmatrix}\n"
    return s



def reduce_order(c_mn):

    M, N = c_mn.shape
    epsilon = 1e-3

    first_nonzero_n = 0
    for n in range(N):
        if np.any(np.abs(c_mn[:, n]) > epsilon):
            first_nonzero_n = n
            break

    

    first_nonzero_m = 0
    for m in range(M):
        if np.any(np.abs(c_mn[m, :(first_nonzero_n+1)]) > epsilon):
            first_nonzero_m = m
            break



    lowest_order_coeff = c_mn[first_nonzero_m, first_nonzero_n]

    return c_mn[first_nonzero_m:, first_nonzero_n:] / lowest_order_coeff, first_nonzero_m, first_nonzero_n, lowest_order_coeff


def collapse_in_alpha(alpha, c_mn):

    M, N = c_mn.shape
    alpha_powers = build_powers(alpha, M)

    return alpha_powers @ c_mn 



# ##############################
# ########## CALCULUS ##########
# ##############################


def taylor_expand_in_alpha(function, order, a0=1e-16):
    """function(x, alpha, *params) -> scalar. Returns series(x, alpha, *params)."""
    def kth_d(g, k):  # k-th derivative wrt alpha at a0
        h = g
        for _ in range(k):
            h = jax.jacfwd(h)   # forward-mode is ideal for 1D alpha
        return h(a0)

    def series(x, alpha, *params):
        g = lambda a: function(x, a, *params)
        ks  = jnp.arange(order + 1, dtype=alpha.dtype)
        fac = jnp.cumprod(jnp.where(ks > 0, ks, 1)).astype(alpha.dtype)
        fac = jnp.where(ks == 0, 1, fac)

        coeffs = [g(a0)] + [kth_d(g, k) for k in range(1, order + 1)]
        coeffs = jnp.stack(coeffs).astype(alpha.dtype)

        return jnp.sum(coeffs * jnp.power(alpha - a0, ks) / fac)

    return series


def taylor_expand_in_alpha(function, order, a0=0.0):
    def kth_d(g, k):
        h = g
        for _ in range(k):
            h = jax.jacfwd(h)
        return h(a0)

    def series(x, alpha, *params):
        g = lambda a: function(x, a, *params)  # must be scalar-valued
        ks  = jnp.arange(order + 1, dtype=alpha.dtype)
        fac = jnp.cumprod(jnp.where(ks > 0, ks, 1)).astype(alpha.dtype)
        fac = jnp.where(ks == 0, 1, fac)
        coeffs = [g(a0)] + [kth_d(g, k) for k in range(1, order + 1)]
        coeffs = jnp.stack(coeffs).astype(alpha.dtype)
        return jnp.sum(coeffs * jnp.power(alpha - a0, ks) / fac)
    return series

# def taylor_expand_in_alpha(function, order, a0=1e-16):
#     """Return a callable that evaluates the Taylor series in alpha up to `order` around a0.
#        `function(x, alpha, *params)` must be scalar-valued in alpha for fixed x, params."""
#     a0 = jnp.asarray(a0, dtype=jnp.float32)  # or match your alpha dtype

#     def series(x, alpha, *params):
#         # g: scalar -> scalar (alpha -> function(x, alpha, *params))
#         g = lambda a: function(x, a, *params)

#         # 0th term
#         term0 = g(a0)

#         # k-th derivative at a0 via repeated jvp on a scalar alpha
#         def kth_derivative(k):
#             v = jnp.asarray(1.0, a0.dtype)
#             y = g(a0)
#             for _ in range(k):
#                 y, v = jax.jvp(g, (a0,), (v,))
#             return v  # tangent is the k-th derivative

#         # collect terms
#         ks = jnp.arange(order + 1, dtype=alpha.dtype)
#         # factorials without gamma (avoids inf/NaN corner cases)
#         fac = jnp.cumprod(jnp.where(ks > 0, ks, 1)).astype(alpha.dtype)
#         fac = jnp.where(ks == 0, 1, fac)

#         derivs = [term0] + [kth_derivative(int(k)) for k in range(1, order + 1)]
#         derivs = jnp.stack(derivs).astype(alpha.dtype)

#         return jnp.sum(derivs * jnp.power(alpha - a0, ks) / fac)

#     return series



# Given a function f(t, alpha, params), compute the Taylor coefficients f(t, alpha) \approx \sum_{mn} c_{mn} alpha^m t^n 
def taylor_expand_2d(f, t0, alpha0, M, N, params=None):
    

    # Make a helper that has the right signature for JAX differentiation.
    def f_base(t, alpha):
        return f(t, alpha, params)

    # Build partial derivatives wrt t in a list f_list[n] = (d^n/dt^n) f.

    f_list = [f_base]      # f_0
    for n_ in range(N):
        fn_plus_1 = jax.grad(f_list[-1], argnums=0)  # derivative wrt t
        f_list.append(fn_plus_1)

    # For each f_list[n], we get all derivatives wrt alpha at alpha0,    
    c = jnp.zeros((M+1, N+1), dtype=jnp.float32)

    def derivatives_wrt_alpha_up_to_order_M(fn_of_alpha, alpha0, M):
        """
        Return [ fn_of_alpha^{(0)}(alpha0), 
                 fn_of_alpha^{(1)}(alpha0),
                 ...
                 fn_of_alpha^{(M)}(alpha0) ] 
        by repeated application of jax.grad wrt alpha (argnums=0).
        """
        out = []
        current_g = fn_of_alpha
        for m_ in range(M+1):
            if m_ == 0:
                # 0th derivative => the function value
                out.append(current_g(alpha0))
            else:
                # 1st..Mth => derivative wrt alpha
                current_g = jax.grad(current_g, argnums=0)
                out.append(current_g(alpha0))
        return jnp.array(out)

    # Loop over n=0..N
    for n_ in range(N+1):
        # The function f_n(t, alpha)
        fn = f_list[n_]

        def fn_of_alpha(a):
            return fn(t0, a)

        # partials_alpha[m] = (d^m / dalpha^m) fn(t0, alpha) at alpha=alpha0
        partials_alpha = derivatives_wrt_alpha_up_to_order_M(fn_of_alpha, alpha0, M)
        # partials_alpha[m] = (∂^(m+n_)/∂t^n_ ∂alpha^m) f  at (t0, alpha0)

        # Fill c[m, n_] = partials_alpha[m] / (m! n_!)
        for m_ in range(M+1):
            val = partials_alpha[m_]
            denom = math.factorial(m_) * math.factorial(n_)
            c = c.at[m_, n_].set(val / denom)

    return c


def integrate_taylor_polynomial(c):
  

    M_plus_1, N_plus_1 = c.shape
    M = M_plus_1 - 1
    N = N_plus_1 - 1
    
    # New array will have shape (M+1, N+2)
    d = np.zeros((M_plus_1, N_plus_1 + 1), dtype=c.dtype)
    
    # d[m,n] = c[m,n-1]/n, except d[m,0] = 0.
    for m in range(M_plus_1):
        for n in range(1, N+2):
            d[m, n] = c[m, n-1] / n

    return d

@jax.jit
def derivative_t_polynomial(c):
    M_plus_1, N_plus_1 = c.shape
    return c[:, 1:] * jnp.arange(1, N_plus_1, dtype=c.dtype)

@jax.jit
def derivative_alpha_polynomial(c):
    M_plus_1, N_plus_1 = c.shape
    return c[1:, :] * jnp.arange(1, M_plus_1, dtype=c.dtype)[:, None]




def log_match(c_mn, M, N):

    K = M + N

    polynomial_coeffs = polynomial_sum([polynomial_power(c_mn, k+1) / (k+1.0) for k in range(K+1)])
    return polynomial_coeffs[:M+1, :N+1]



def matching_coeffs(p_t, M, N):

    # There has to be *some* perturbative info!
    if M == 0:
        raise ValueError("M must be greater than 0")
 
    
    p_mn = taylor_expand_2d(p_t, 0.0, 0.0, M, N)
    P_mn = integrate_taylor_polynomial(p_mn)


    # Divide out the lowest order term of p_mn
    p_mn_reduced, m_star, n_star, p_star = reduce_order(p_mn)
    p_mn_reduced = p_mn_reduced.at[0, 0].set(0.0)

    # p_matched = log_match(-p_mn_reduced, M, N)
    p_matched = log_match(-p_mn_reduced, M - m_star, N-n_star)

    
    P_matched = log_match(P_mn, M, N) 

    # We want to get rid of the extra orders in the polynomial, we only need M - m_star since g_star makes up for it
    P_matched = P_matched.at[M-m_star + 1:, :].set(0.0)
    P_matched = P_matched.at[:, N-n_star + 1:].set(0.0)



    g_mn = polynomial_sum([-p_matched, P_matched])
    g_star = jnp.zeros((M+1, N+1))
    g_star = g_star.at[m_star, n_star].set(-p_star)

    return g_mn, g_star