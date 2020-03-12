import numbers

import numpy as np

from .backend import get_backend


def compute_lipschitz_constants(Xs, kernelize="XTX"):
    """Compute Lipschitz constants of gradients of linear regression problems.

    Find the largest eigenvalue of X^TX for several X, using power iteration.

    Parameters
    ----------
    Xs : array of shape (n_kernels, n_samples, n_features) or
            (n_kernels, n_samples, n_samples)
        Multiple linear features or kernels.
    kernelize : str in {"XTX", "XXT", "X"}
        Whether to consider X^TX, XX^T, or directly X.

    Returns
    -------
    lipschitz : array of shape (n_kernels)
        Lipschitz constants.
    """
    backend = get_backend()

    if kernelize == "XXT":
        XTs = backend.transpose(Xs, (0, 2, 1))
        kernels = backend.matmul(Xs, XTs)
    elif kernelize == "XTX":
        XTs = backend.transpose(Xs, (0, 2, 1))
        kernels = backend.matmul(XTs, Xs)
    elif kernelize == "X":
        kernels = Xs
    else:
        raise ValueError("Unknown parameter kernelize=%r" % (kernelize, ))

    ys = backend.randn(*(kernels.shape[:2] + (1, )))
    ys = backend.asarray_like(ys, Xs)
    for i in range(10):
        ys /= backend.norm(ys, axis=1, keepdims=True) + 1e-16
        ys = backend.matmul(kernels, ys)
    evs = backend.norm(ys, axis=1)[:, 0]
    return evs


def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    """Test array equality, casting all arrays to numpy."""
    backend = get_backend()
    x = backend.to_numpy(x)
    y = backend.to_numpy(y)
    return np.testing.assert_array_almost_equal(x, y, decimal=decimal,
                                                err_msg=err_msg,
                                                verbose=verbose)


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
