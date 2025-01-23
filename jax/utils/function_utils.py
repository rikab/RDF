import numpy as np
import jax.numpy as jnp
import jax
import math
from scipy.signal import convolve2d


# #################################
# ########## POLYNOMIALS ##########
# #################################

# Necessary for jax to work with 0^0@jax.jit

def build_powers(base, length):
   
    def body_fun(i, arr):
        # arr[i] = arr[i-1] * base
        return arr.at[i].set(arr[i-1] * base)
    
    # Initialize an array of zeros, then set arr[0] = 1
    arr = jnp.zeros((length,))
    arr = arr.at[0].set(1.0)
    
    # fori_loop will fill in arr[1], arr[2], ... arr[length-1]
    arr = jax.lax.fori_loop(1, length, body_fun, arr)
    return arr



@jax.jit
def polynomial(t, alpha, params):

    M, N = params.shape
    
    # Build powers of alpha: [1, alpha, alpha^2, ... alpha^(M-1)]
    alpha_powers = build_powers(alpha, M)  # shape (M,)

    # Build powers of t: [1, t, t^2, ... t^(N-1)]
    t_powers = build_powers(t, N)         # shape (N,)

    poly_val = alpha_powers @ params @ t_powers
    return poly_val


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



# ##############################
# ########## CALCULUS ##########
# ##############################


def taylor_expand_in_alpha(function, order):

    ps = [function,]
    if order > 0:
        for i in range(order):
            ps.append(jax.grad(ps[-1], argnums=1))

    def taylor_expansion(x, alpha, params):
        near_zero = 1e-16
        terms = jnp.array([p(x, near_zero, params) for p in ps])
        factorials = jax.scipy.special.gamma(jnp.arange(len(terms)) + 1)

        return jnp.sum(terms / factorials * jnp.power(alpha, jnp.arange(len(terms))))
    
    return taylor_expansion



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
    c = jnp.zeros((M+1, N+1), dtype=jnp.float64)

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