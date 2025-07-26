import numpy as np
import jax.numpy as jnp
import jax
import math


# ########## TAYLOR EXPANSIONS ##########

def taylor_expand_in_alpha(function, order):

    ps = [function,]
    if order > 0:
        for i in range(order):
            ps.append(jax.grad(ps[-1], argnums=1))

    # Build the Taylor expansion
    return_list = []
    for i in range(order+1):

        def order_n(x, alpha, params, i = i):

            near_zero = 1e-16
            factorial = jax.scipy.special.gamma(i + 1)
            return ps[i](x, near_zero, params) / factorial * jnp.power(alpha, i)
        
        return_list.append(order_n)

    return return_list


def eval_taylor_expansion(taylor_expansion_list, x, alpha, params):

    # Evaluate the Taylor expansion at x and alpha
    result = 0.0
    for i in range(len(taylor_expansion_list)):
        result += taylor_expansion_list[i](x, alpha, params)

    return result


# #############################
# ########## ALGEBRA ##########
# #############################


def _coefficients_from_expansion(exp):
    """
    Take a list of f_i(x, alpha, params) = c_i(x,params)*alpha**i
    and return the list of coefficient functions c_i(x,params).
    """
    coeffs = []
    for f in exp:
        coeffs.append(lambda x, params, f=f: f(x, 1.0, params))  # α=1 ⇒ coefficient
    return coeffs


def _expansion_from_coefficients(coeffs):
    """
    Take a list of coefficient functions c_i(x,params) and wrap them
    into g_i(x, alpha, params) = c_i(x,params)*alpha**i.
    """
    expansion = []
    for i, c in enumerate(coeffs):
        expansion.append(
            lambda x, alpha, params, c=c, i=i: c(x, params) * jnp.power(alpha, i)
        )
    return expansion



def taylor_add(*expansions):
    """
    Sum an arbitrary number of Taylor expansions.
    The result keeps the largest order that appears in any input.
    """
    coeff_lists = [_coefficients_from_expansion(e) for e in expansions]
    max_order  = max(len(c) for c in coeff_lists)

    # pad with a zero‑function so all lists have the same length
    zero = lambda x, params: 0.0
    coeff_lists = [c + [zero]*(max_order - len(c)) for c in coeff_lists]

    def _sum_coeffs(funcs_at_same_power):
        def c(x, params):
            s = 0.0
            for f in funcs_at_same_power:
                s += f(x, params)
            return s
        return c

    summed_coeffs = [
        _sum_coeffs([c_list[k] for c_list in coeff_lists])
        for k in range(max_order)
    ]
    return _expansion_from_coefficients(summed_coeffs)



def _mul_coeffs(c1, c2, order):
    out = []
    for k in range(order + 1):
        def c_k(x, params, k=k, c1=c1, c2=c2):
            s = 0.0
            for i in range(k + 1):
                if i < len(c1) and (k - i) < len(c2):
                    s += c1[i](x, params) * c2[k - i](x, params)
            return s
        out.append(c_k)
    return out


def taylor_multiply(exp1, exp2, order=None):
    """
    Multiply two Taylor expansions.
    If `order` is omitted the full (len1+len2‑2) series is returned.
    """
    c1 = _coefficients_from_expansion(exp1)
    c2 = _coefficients_from_expansion(exp2)

    if order is None:
        order = len(c1) + len(c2) - 2

    return _expansion_from_coefficients(_mul_coeffs(c1, c2, order))




def taylor_power(exp, k, order=None):
    """
    Raise a Taylor expansion to a non‑negative integer power k.
    """
    if k < 0:
        raise ValueError("Negative exponents not supported")
    if k == 0:
        # return the constant 1
        one_coeff = [lambda x, params: 1.0]
        return _expansion_from_coefficients(one_coeff)
    if k == 1:
        return exp

    # exponentiation by squaring
    def _pow(e, n):
        if n == 1:
            return e
        if n % 2 == 0:
            half = _pow(e, n // 2)
            return taylor_multiply(half, half, order)
        else:
            return taylor_multiply(e, _pow(e, n - 1), order)

    return _pow(exp, k)



# ----  definite‑integral of a series  ----
def _integrate_coeff(c, n_pts=1024):
    @jax.jit
    def F(t, params):
        grid = jnp.linspace(0.0, t, n_pts)
        vals = jax.vmap(lambda s: c(s, params))(grid)
        return jnp.trapz(vals, grid)
    return F


def integrate_taylor(series, n_pts=1024):
    coeffs =  _coefficients_from_expansion(series)
    int_coeffs = [_integrate_coeff(c, n_pts) for c in coeffs]
    return _expansion_from_coefficients(int_coeffs)



# ##############################
# ########## MATCHING ##########
# ##############################



def reduce_alpha(p_m):

    NUM_TRIALS = 1000
    epsilon = 1e-6

    M = len(p_m)

    # Find the first non-zero function in p_m
    first_nonzero_m = 0
    ts = np.linspace(0, 20, NUM_TRIALS)

    for m in range(M):
        func = jax.vmap(p_m[m], in_axes=(0, None, None))
        if np.any(np.abs(func(ts, 1, None)) > epsilon):
            first_nonzero_m = m
            break
    
    p_star = p_m[first_nonzero_m]
    p_m = p_m[first_nonzero_m:]
    r_m = []

    # Normalize each function in p_m by dividing by p_star
    for m in range(len(p_m)):
        
        def p_m_func(t, alpha, params, m = m):
            return p_m[m](t, alpha, params) / p_star(t, alpha, params)
        
        r_m.append(p_m_func)

    return r_m, p_star, first_nonzero_m


def scale(exp, scalar):
    """Multiply a Taylor expansion by a scalar."""
    coeffs = _coefficients_from_expansion(exp)
    scaled = [lambda x, p, c=c: scalar * c(x, p) for c in coeffs]
    return _expansion_from_coefficients(scaled)


def truncate(exp, M):
    """Keep only the first M+1 terms (α⁰ … αᴹ)."""
    return exp[: M + 1]


def taylor_log_match(exp, M):


    K = M                         # only powers that can still contribute ≤ M
    terms = []
    for k in range(K + 1):
        pow_k1  = taylor_power(exp, k + 1, order=M)     # (k+1)‑st power
        scaled  = scale(pow_k1, 1.0 / (k + 1))   # divide by (k+1)
        terms.append(scaled)

    summed = taylor_add(*terms)     # add all terms together
    return truncate(summed, M)






def matching_coeffs(p_t, M):
    """
    """

    # There has to be *some* perturbative info!
    if M == 0:
        raise ValueError("M must be greater than 0")
    
    # ########## Get the Taylor expansion of the function ##########
    p_m = (taylor_expand_in_alpha(p_t, M))
    P_m = integrate_taylor(p_m)

    # ########## Reduce the order of the Taylor expansion, extract leading coefficient ##########
    r_m, p_star, m_star = reduce_alpha(p_m) # r_m = p_m / p_star


    # ########## Get the logarithmic matching ##########
    def zero_func(t, alpha, params):
        return 0.0
    
    r_m[0] = zero_func()

    p_matched = taylor_log_match(-r_m, M - m_star)
    P_matched = taylor_log_match(P_m, M)

    p_matched = truncate(p_matched, M - m_star)
    P_matched = truncate(P_matched, M - m_star)


    # ########## Add the terms #########
    g_m = taylor_add(- p_matched, P_matched)
    g_star = -p_star
    g_star = g_star + [zero_func() for _ in range(len(g_m) - len(g_star))] # padding


    return g_m, g_star

