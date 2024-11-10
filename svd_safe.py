"""
SVD with backwards formula for complex matrices and safe inverse,
"safe" in light of degenerate or vanishing singular values

with various convenience wrappers
"""

import jax.numpy as np
from jax import custom_vjp
from functools import partial
from jax.lax import dynamic_slice_in_dim

from jax.numpy.linalg import multi_dot
from np_array_ops import Hc, T, Cc
from jax import value_and_grad

DEFAULT_EPS = 1e-12
DEFAULT_CUTOFF = 0.  # TODO changed this fpr custom jvp tracing.


# TODO extensive testing

# TODO doc all

# TODO norm_change also for tensor svd


@partial(custom_vjp, nondiff_argnums=(1,))
def svd(A, epsilon=DEFAULT_EPS):
    """
    SVD with VJP (backwards mode auto-diff) formula for complex matrices and safe inverse
    (for stability when eigenvalues are degenerate or zero)

    Computes `U`, `S`, `Vh` such that
    1)
        ```
        A
        == (U * S) @ Vh
        == U @ np.diag(S) @ Vh
        == U @ (S[:, None] * Vh)
        ```
    2) S is real, non-negative
    3) U and Vh are isometric (`Hc(U) @ U == eye(k)` and `Vh @ Hc(Vh) == eye(k)`)

    Parameters
    ----------
    A : jax.numpy.ndarray
        The matrix to perform the SVD on. Shape (m,n)
    epsilon : float
        The control parameter for safe inverse. 1/x is replaced by x/(x**2 + epsilon)
        Should be very small.

    Returns
    -------
    U : jax.numpy.ndarray
        2D array. shape (m,k) where k = min(m,n) and (m,n) = A.shape
    S : jax.numpy.ndarray
        1D array of real, non-negative singular values. shape (k,)  where k = min(m,n) and (m,n) = A.shape
    Vh : jax.numpy.ndarray
        2D array. shape (k,n) where k = min(m,n) and (m,n) = A.shape
    """
    assert epsilon > 0
    U , S , Vh = np.linalg.svd(A, full_matrices=False)
    return (U,S,Vh) 


def svd_reduced(A, tolerance=1e-12, epsilon=DEFAULT_EPS, return_norm_change=False):
    """
    Like `svd`, but ignores singular-values <= `tolerance`.
    """
    U, S, Vh = svd(A, epsilon)

    if tolerance > 0.:

        if return_norm_change:
            old_S_norm = np.linalg.norm(S)

            k = np.sum(S > tolerance)
            U = dynamic_slice_in_dim(U, 0, k, 1)
            S = dynamic_slice_in_dim(S, 0, k, 0)
            Vh = dynamic_slice_in_dim(Vh, 0, k, 0)
            norm_change = np.linalg.norm(S) / old_S_norm

            return U, S, Vh, norm_change

        else:
            k = np.sum(S > tolerance)
            # U = U[:, :k]
            # S = S[:k]
            # Vh = Vh[:k, :]
            U = dynamic_slice_in_dim(U, 0, k, 1)
            S = dynamic_slice_in_dim(S, 0, k, 0)
            Vh = dynamic_slice_in_dim(Vh, 0, k, 0)
            return U, S, Vh

    else:
        if return_norm_change:
            return U, S, Vh, 1.
        else:
            return U, S, Vh


def svd_truncated(A, chi_max=None, cutoff=DEFAULT_CUTOFF, epsilon=DEFAULT_EPS, return_norm_change=False):
    """
    Like `svd`, but keeps at most `chi_max` many singular values and ignores singular values below `cutoff`
    """
    if return_norm_change:
        U, S, Vh, norm_change = svd_reduced(A, tolerance=cutoff, epsilon=epsilon, return_norm_change=True)

        if chi_max is not None:
            k = np.min([chi_max, len(S)])
            old_S_norm = np.linalg.norm(S)
            U = dynamic_slice_in_dim(U, 0, k, 1)
            S = dynamic_slice_in_dim(S, 0, k, 0)
            Vh = dynamic_slice_in_dim(Vh, 0, k, 0)
            norm_change *= np.linalg.norm(S) / old_S_norm  # FIXME is this correct?

        return U, S, Vh, norm_change

    else:
        U, S, Vh = svd_reduced(A, tolerance=cutoff, epsilon=epsilon, return_norm_change=False)

        if chi_max is not None:
            k = np.min([chi_max, len(S)])
            U = dynamic_slice_in_dim(U, 0, k, 1)
            S = dynamic_slice_in_dim(S, 0, k, 0)
            Vh = dynamic_slice_in_dim(Vh, 0, k, 0)

        return U, S, Vh


def tensor_svd(A, u_legs, vh_legs, epsilon=DEFAULT_EPS):
    """
    SVD for tensors.
    (group legs, perform SVD, ungroup legs)

    Parameters
    ----------
    A : jax.numpy.ndarray
        The tensor to perform the svd on. shape (a1,a2,...)
    u_legs : List of int
        The indices/legs of `A` that are grouped to the first matrix index for SVD
        (they end up as legs of `U`). The order determines the leg-order of `U`
        Each integer in range(len(A.shape)) must appear exactly once in `u_legs` xor `vh_legs`
    vh_legs : List of int
        The indices/legs of `A` that are grouped to the second matrix index for SVD
        (they end up as legs of `Vh`). The order determines the leg-order of `Vh`
        Each integer in range(len(A.shape)) must appear exactly once in `u_legs` xor `vh_legs`
    epsilon : float
        The control parameter for safe inverse. 1/x is replaced by x/(x**2 + epsilon)
        Should be very small.


    Returns
    -------
    U : jax.numpy.ndarray
        Isometric tensor with shape (u1,u2,...,k) where ui = A.shape[u_legs[i]]
    S : jax.numpy.ndarray
        1D array of singular values with shape (k,)
    Vh : jax.numpy.ndarray
        Isometric tensor with shape (k,vh1,vh2,...) where vhi = A.shape[vh_legs[i]]
    """

    u_leg_dims = [A.shape[u_leg] for u_leg in u_legs]
    m = np.prod(u_leg_dims)
    vh_leg_dims = [A.shape[vh_leg] for vh_leg in vh_legs]
    n = np.prod(vh_leg_dims)
    A_mat = np.reshape(np.transpose(A, u_legs + vh_legs), [m, n])
    U_mat, S, Vh_mat = svd(A_mat, epsilon)
    k = len(S)
    U = np.reshape(U_mat, u_leg_dims + [k])
    Vh = np.reshape(Vh_mat, [k] + vh_leg_dims)

    return U, S, Vh


def tensor_svd_reduced(A, u_legs, vh_legs, tolerance=1e-12, epsilon=DEFAULT_EPS):
    """
    Like `tensor_svd`, but ignores singular-values <= `tolerance`.
    """

    u_leg_dims = [A.shape[u_leg] for u_leg in u_legs]
    m = np.prod(u_leg_dims)
    vh_leg_dims = [A.shape[vh_leg] for vh_leg in vh_legs]
    n = np.prod(vh_leg_dims)
    A_mat = np.reshape(np.transpose(A, u_legs + vh_legs), [m, n])
    U_mat, S, Vh_mat = svd_reduced(A_mat, tolerance, epsilon)
    k = len(S)
    U = np.reshape(U_mat, u_leg_dims + [k])
    Vh = np.reshape(Vh_mat, [k] + vh_leg_dims)

    return U, S, Vh


def tensor_svd_truncated(A, u_legs, vh_legs, chi_max=None, cutoff=DEFAULT_CUTOFF, epsilon=DEFAULT_EPS):
    """
    Like `tensor_svd`, but keeps at most `chi_max` many singular values and ignores singular values below `cutoff
    """

    u_leg_dims = [A.shape[u_leg] for u_leg in u_legs]
    m = np.prod(u_leg_dims)
    vh_leg_dims = [A.shape[vh_leg] for vh_leg in vh_legs]
    n = np.prod(vh_leg_dims)
    A_mat = np.reshape(np.transpose(A, u_legs + vh_legs), [m, n])
    U_mat, S, Vh_mat = svd_truncated(A_mat, chi_max, cutoff, epsilon)
    k = len(S)
    U = np.reshape(U_mat, u_leg_dims + [k])
    Vh = np.reshape(Vh_mat, [k] + vh_leg_dims)

    return U, S, Vh


def _safe_inverse(x, eps):
    return x / (x ** 2 + eps)


def _svd_fwd(A, epsilon):
    assert epsilon > 0
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    res = (U, S, Vh, A)
    return (U, S, Vh), res


def _svd_bwd(epsilon, res, g):
    # FIXME double check

    assert epsilon > 0
    dU, dS, dVh = g
    U, S, Vh, A = res

    # avoid re-computation of the following in multiple steps
    Uc = Cc(U)
    Ut = T(U)
    Vt = Cc(Vh)
    Vt_dV = np.dot(Vt, Hc(dVh))
    S_squared = S ** 2
    S_inv = _safe_inverse(S, epsilon)

    # matrices in the AD formula
    I = np.eye(len(S))
    F = _safe_inverse(S_squared[None, :] - S_squared[:, None], epsilon)
    F = F - I * F  # zeroes on the diagonal
    J = F * (np.dot(Ut, dU))
    K = F * Vt_dV
    L = I * Vt_dV

    # cc of projectors onto orthogonal complement of U (V)
    m, n = A.shape
    Pc_U_perp = np.eye(m) - np.dot(Uc, Ut)
    Pc_V_perp = np.eye(n) - np.dot(T(Vh), Vt)

    # AD formula
    dA = np.dot(Uc * dS, Vt) \
         + multi_dot([Uc, (J + Hc(J)) * S, Vt]) \
         + multi_dot([Uc * S, K + Hc(K), Vt]) \
         + .5 * multi_dot([Uc * S_inv, L - Hc(L), Vt]) \
         + multi_dot([Pc_U_perp, dU * S_inv, Vt]) \
         + multi_dot([Uc * S_inv, dVh, Pc_V_perp])

    return dA,


svd.defvjp(_svd_fwd, _svd_bwd)


