import numbers

from sklearn.model_selection import check_cv

from ..backend import get_current_backend
from ..progress_bar import ProgressBar
from ..utils import compute_lipschitz_constants

from ._kernel_ridge import solve_kernel_ridge_conjugate_gradient
from ._kernel_ridge import solve_kernel_ridge_gradient_descent
from ._kernel_ridge import solve_kernel_ridge_neumann_series
from ._random_search import solve_multiple_kernel_ridge_random_search
from ._random_search import generate_dirichlet_samples


def solve_multiple_kernel_ridge_hyper_gradient(
        Ks, Y, score_func, cv_splitter=10, initial_dual_weights=None,
        initial_deltas=0, max_iter=100, tol=1e-2, max_iter_inner_dual=1,
        max_iter_inner_hyper=1, cg_tol=1e-3, n_targets_batch=None,
        hyper_gradient_method="conjugate", kernel_ridge_method="gradient",
        random_state=None, progress_bar=True):
    """Solve bilinear kernel ridge regression with cross-validation.

    The hyper-parameters deltas correspond to torch.log(gammas / alphas).

    Parameters
    ----------
    Ks : array of shape (n_kernels, n_samples, n_samples)
        Training kernel for each feature space.
    Y : array of shape (n_samples, n_targets)
        Training target data.
    score_func : callable
        Function used to compute the score of predictions.
    cv_splitter : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.
    initial_dual_weights : array of shape (n_samples, n_targets) or None
        Initial kernel ridge coefficients.
    initial_deltas : str, float, array of shape (n_kernels, n_targets)
        Initial log kernel weights for each target.
        If a str, initialize the deltas with different strategies:
            - 'dirichlet' : draw 10 Dirichlet samples and keep the best over cv
            - 'ridgecv' : fit a RidgeCV model over the average kernel
        If a float, initialize the deltas with this value.
    max_iter : int
        Maximum number of iteration for the outer loop.
    tol : float > 0, or None
        Tolerance for the stopping criterion.
    max_iter_inner_dual : int
        Maximum number of iterations for the dual weights conjugate gradient.
    max_iter_inner_hyper :
        Maximum number of iterations for the deltas gradient descent.
    cg_tol : float, or array of shape (max_iter)
        Tolerance for the conjugate gradients.
    n_targets_batch : int or None
        Size of the batch for computing predictions. Used for memory reasons.
        If None, uses all n_targets at once.
    hyper_gradient_method : str, "conjugate", "neumann", "direct"
        Method to compute the hypergradient.
    kernel_ridge_method : str, "conjugate" or "gradient"
        Algorithm used for the inner step.
    random_state : int, np.random.RandomState, or None
        Random generator state, used only in the Dirichlet sampling init.
    progress_bar : bool
        If True, display a progress bar over batches and iterations.

    Returns
    -------
    all_scores_mean : array of shape (max_iter * max_iter_inner_hyper,
            n_targets)
        Scores averaged over splits for each iteration.
    dual_weights : array of shape (n_samples, n_targets)
        Kernel ridge coefficients.
    deltas : array of shape (n_kernels, n_targets)
        Log kernel weights for each target.
    """
    backend = get_current_backend()

    n_samples, n_targets = Y.shape
    if n_targets_batch is None:
        n_targets_batch = n_targets

    cv_splitter = check_cv(cv_splitter)
    n_splits = cv_splitter.get_n_splits()

    Y, Ks = backend.check_arrays(Y, Ks)

    deltas, dual_weights = _init_multiple_kernel_ridge(
        initial_deltas, initial_dual_weights, Ks, Y, cv_splitter,
        n_targets_batch, random_state, progress_bar)

    name = "hypergradient_" + hyper_gradient_method
    if kernel_ridge_method == "conjugate":
        inner_function = solve_kernel_ridge_conjugate_gradient
    elif kernel_ridge_method == "gradient":
        inner_function = solve_kernel_ridge_gradient_descent
    else:
        raise ValueError("Unknown parameter kernel_ridge_method=%r." %
                         (kernel_ridge_method, ))

    if isinstance(cg_tol, (int, float)):
        cg_tol = backend.full_like(Y, shape=max_iter, fill_value=cg_tol)

    alpha = 1.0

    all_scores_mean = backend.zeros_like(
        Y, shape=(max_iter * max_iter_inner_hyper, n_targets))

    batch_iterates = backend.arange(0, n_targets, n_targets_batch)
    if progress_bar:
        bar = ProgressBar(title=name, max_value=len(batch_iterates) * max_iter)
    for bb, start in enumerate(batch_iterates):
        batch = slice(start, start + n_targets_batch)

        previous_solutions = [None] * n_splits
        step_sizes = [None] * n_splits

        dual_weights_cv = [
            backend.copy(dual_weights[train, batch])
            for train, _ in cv_splitter.split(Y)
        ]  # this might be a poor initialization

        for ii in range(max_iter):
            if progress_bar:
                bar.update(bb * max_iter + ii)

            ##########################
            # updates the dual weights

            # First pass needs more iterations to have something reasonable.
            # We also use conjugate gradient as it converges faster.
            if ii == 0:
                max_iter_inner_dual_ = 50
                cg_tol_ = min(1e-2, cg_tol[ii])
                inner_function_ = solve_kernel_ridge_conjugate_gradient
            else:
                max_iter_inner_dual_ = max_iter_inner_dual
                cg_tol_ = cg_tol[ii]
                inner_function_ = inner_function

            for kk, (train, val) in enumerate(cv_splitter.split(Y)):
                if hasattr(Y, "device"):
                    train = backend.asarray(train, device=Y.device)
                Ks_train = Ks[:, train[:, None], train]
                Y_train = Y[train, batch]

                dual_weights_cv[kk] = inner_function_(
                    Ks_train, Y_train, backend.exp(deltas[:, batch]),
                    initial_dual_weights=dual_weights_cv[kk], alpha=alpha,
                    max_iter=max_iter_inner_dual_, tol=cg_tol_)

            ###################
            # update the deltas
            deltas_old = backend.copy(deltas[:, batch])
            for jj in range(max_iter_inner_hyper):

                gradients = backend.zeros_like(deltas[:, batch])
                scores = backend.zeros_like(
                    Y, shape=(n_splits, deltas[:, batch].shape[1]))
                for kk, (train, val) in enumerate(cv_splitter.split(Y)):
                    if hasattr(Y, "device"):
                        val = backend.asarray(val, device=Y.device)
                        train = backend.asarray(train, device=Y.device)

                    Ks_val = Ks[:, val[:, None], train]
                    Ks_train = Ks[:, train[:, None], train]
                    Y_val = Y[val, batch]

                    (gradients_kk, step_sizes[kk], predictions,
                     previous_solutions[kk]) = _compute_delta_gradient(
                         Ks_val=Ks_val, Y_val=Y_val, deltas=deltas[:, batch],
                         dual_weights=dual_weights_cv[kk], Ks_train=Ks_train,
                         tol=cg_tol[ii],
                         hyper_gradient_method=hyper_gradient_method,
                         previous_solution=previous_solutions[kk])

                    gradients += gradients_kk * Y_val.shape[0] / n_samples

                    scores[kk] = score_func(Y_val, predictions)

                it = ii * max_iter_inner_hyper + jj
                all_scores_mean[it, batch] = scores.mean(0)

                # update deltas, using the minimum step size over splits
                step_size = backend.min(backend.stack(step_sizes), axis=0)
                deltas[:, batch] -= gradients * step_size[None, :]
                assert not backend.any(
                    backend.isinf(backend.exp(deltas[:, batch])))

            ####################
            # stopping criterion
            if tol is not None:
                if backend.max(
                        backend.abs(deltas_old - deltas[:, batch])) < tol:
                    break

        ##########################################
        # refit dual weights on the entire dataset
        dual_weights[:, batch] = solve_kernel_ridge_conjugate_gradient(
            Ks, Y[:, batch], backend.exp(deltas[:, batch]),
            initial_dual_weights=dual_weights[:, batch], alpha=alpha,
            max_iter=100, tol=1e-3)

    if progress_bar:
        bar.update(bar.max_value)

    return all_scores_mean, dual_weights, deltas


def _init_multiple_kernel_ridge(initial_deltas, initial_dual_weights, Ks, Y,
                                cv_splitter, n_targets_batch, random_state,
                                progress_bar):
    """Initialize deltas (log kernel weights) and dual_weights.

    Parameters
    ----------
    initial_deltas : str, float, array of shape (n_kernels, n_targets)
        Initial log kernel weights for each target.
        If a str, initialize the deltas with different strategies:
            - 'dirichlet' : draw 10 Dirichlet samples and keep the best over cv
            - 'ridgecv' : fit a RidgeCV model over the average kernel
        If a float, initialize the deltas with this value.
    initial_dual_weights : array of shape (n_samples, n_targets) or None
        Initial kernel ridge coefficients.
    Ks : array of shape (n_kernels, n_samples, n_samples)
        Training kernel for each feature space.
    Y : array of shape (n_samples, n_targets)
        Training target data.
    cv_splitter : int or scikit-learn splitter
        Cross-validation splitter. If an int, KFold is used.
    n_targets_batch : int or None
        Size of the batch for computing predictions. Used for memory reasons.
        If None, uses all n_targets at once.
    random_state : int, np.random.RandomState, or None
        Random generator state, used only in the Dirichlet sampling init.
    progress_bar : bool
        If True, display a progress bar over Dirichlet samples.

    Returns
    -------
    deltas : array of shape (n_kernels, n_targets)
        Initial deltas.
    dual_weights : array of shape (n_samples, n_targets)
        Initial kernel ridge coefficients.
    """
    backend = get_current_backend()

    n_kernels = Ks.shape[0]
    n_targets = Y.shape[1]
    dual_weights = None

    if n_targets_batch is None:
        n_targets_batch = n_targets

    if initial_deltas is None:
        initial_deltas = 0

    if ((isinstance(initial_deltas, str)
         and (initial_deltas in ['ridgecv', 'dirichlet']))):
        if initial_deltas == 'dirichlet':
            n_samples = 10

            gammas = generate_dirichlet_samples(n_samples, n_kernels,
                                                concentrations=[.1, 1.],
                                                random_state=random_state)
        else:
            gammas = backend.full_like(Y, shape=n_kernels,
                                       fill_value=1. / n_kernels)[None]
        alphas = backend.logspace(-10, 25, 31)

        results = solve_multiple_kernel_ridge_random_search(
            Ks, Y, gammas, alphas, cv_splitter=cv_splitter,
            n_targets_batch=n_targets_batch, compute_weights='dual',
            progress_bar=progress_bar)
        _, best_gammas, best_alphas, dual_weights = results

        deltas = backend.log(best_gammas / best_alphas[None, :])
        dual_weights = dual_weights * best_alphas

    elif isinstance(initial_deltas, numbers.Number):
        deltas = backend.full_like(Y, shape=(n_kernels, n_targets),
                                   fill_value=initial_deltas)

    else:
        deltas = backend.copy(backend.asarray_like(initial_deltas, Y))

    if dual_weights is None:
        if initial_dual_weights is not None:
            dual_weights = backend.copy(
                backend.asarray_like(initial_dual_weights, Y))

        else:
            dual_weights = backend.zeros_like(Y)

    Y, deltas, dual_weights = backend.check_arrays(Y, deltas, dual_weights)
    return deltas, dual_weights


def _compute_delta_loss(Ks_val, Y_val, deltas, dual_weights):
    """Compute the validation loss.

    Parameters
    ----------
    Ks_val : array of shape (n_kernels, n_samples_val, n_samples_train)
        Cross-kernels between training set and validation set.
    Y_val : array of shape (n_samples_val, n_targets)
        Target data on the validation set.
    deltas : array of shape (n_kernels, n_targets)
        Log of the kernel weights.
    dual_weights : array of shape (n_samples_train, n_targets)
        Kernel ridge weights.

    Returns
    -------
    loss : array of shape (n_targets)
        L2 validation loss.
    """
    backend = get_current_backend()
    exp_delta = backend.exp(deltas)
    chi_val = backend.matmul(Ks_val, dual_weights)
    exp_delta_chi_val = exp_delta[:, None, :] * chi_val
    predictions = exp_delta_chi_val.sum(0)
    residuals = predictions - Y_val
    loss = 0.5 * backend.norm(residuals, axis=0) ** 2
    return loss


def _compute_delta_gradient(Ks_val, Y_val, deltas, dual_weights, Ks_train=None,
                            tol=None, previous_solution=None,
                            hyper_gradient_method='conjugate'):
    """Compute the gradient over deltas on the validation dataset.

    Parameters
    ----------
    Ks_val : array of shape (n_kernels, n_samples_val, n_samples_train)
        Cross-kernels between training set and validation set.
    Y_val : array of shape (n_samples_val, n_targets)
        Target data on the validation set.
    deltas : array of shape (n_kernels, n_targets)
        Log of the kernel weights.
    dual_weights : array of shape (n_samples_train, n_targets)
        Kernel ridge weights.
    Ks_train : array of shape (n_kernels, n_samples_train, n_samples_train)
        Kernels on the training set.
        Not required if hyper_gradient_method = "direct".
    tol : float
        Tolerance for the conjugate method.
        Required if hyper_gradient_method = "conjugate".
    previous_solution
        Speed up hyper_gradient_method = "conjugate" by warm starting with the
        previous solution.
    hyper_gradient_method : str, in {"conjugate", "neumann", "direct"}
        Method used to compute the hyper gradient.

    Returns
    -------
    gradient : array of shape (n_kernels, n_targets)
        Gradient over deltas.
    step_size : array of shape (n_targets)
        Step size computed based on the direct gradient's Lipschitz constant.
    predictions : array of shape (n_samples_val, n_targets)
        Predictions on the validation set.
    solution : array of shape (n_samples_train, n_targets) or None
        Solution of the inverse Hessian in the indirect gradient.
    """
    backend = get_current_backend()

    # prepare quantities
    exp_delta = backend.exp(deltas)
    chi_val = backend.matmul(Ks_val, dual_weights)
    exp_delta_chi_val = exp_delta[:, None, :] * chi_val
    predictions = exp_delta_chi_val.sum(0)
    assert predictions.shape == Y_val.shape

    # direct gradient
    residuals = predictions - Y_val
    direct_gradient = (residuals[None] * exp_delta_chi_val).sum(1)
    assert direct_gradient.shape == deltas.shape

    # estimate a step size
    XTXs = _compute_deltas_hessian(exp_delta_chi_val, Y_val)
    # (these lipschitz constants only correspond to the direct gradient)
    lipschitz_1 = compute_lipschitz_constants(XTXs, "X")
    step_size = 1. / (lipschitz_1 + 1e-15)

    if hyper_gradient_method == 'direct':
        gradient = direct_gradient
        solution = None
    else:
        # compute the indirect gradient by inverting the Hessian

        # compute nabla_g_1
        tmp = backend.matmul(backend.transpose(Ks_val, (2, 0, 1)), residuals)
        tmp = backend.transpose(tmp, (2, 0, 1))
        nabla_g_1 = backend.matmul(
            tmp,
            backend.transpose(exp_delta, (1, 0))[:, :, None])
        nabla_g_1 = backend.transpose(nabla_g_1[:, :, 0], (1, 0))
        assert nabla_g_1.shape == dual_weights.shape

        # solve linear system (sum_i gamma[i]*K[i] + 1) @ X = nabla_g_1
        alpha = 1
        assert Ks_train is not None
        if hyper_gradient_method == 'conjugate':
            assert tol is not None
            solution = solve_kernel_ridge_conjugate_gradient(
                Ks_train, nabla_g_1, gammas=exp_delta,
                initial_dual_weights=previous_solution, max_iter=100, tol=tol,
                alpha=alpha)
        elif hyper_gradient_method == 'neumann':
            solution = solve_kernel_ridge_neumann_series(
                Ks_train, nabla_g_1, exp_delta, max_iter=5, factor=0.00001,
                alpha=alpha)
        else:
            raise ValueError("Unknown parameter hyper_gradient_method=%r." %
                             (hyper_gradient_method, ))

        # finish the indirect gradient
        chi_train = backend.matmul(Ks_train, dual_weights)
        exp_delta_chi_train = exp_delta[:, None, :] * chi_train
        indirect_gradient = (exp_delta_chi_train * solution[None, :, :]).sum(1)
        assert indirect_gradient.shape == deltas.shape

        gradient = direct_gradient - indirect_gradient
        assert not backend.any(backend.isinf(gradient))

    return gradient, step_size, predictions, solution


def _compute_deltas_hessian(exp_delta_chi, Y):
    """Compute the hessian of the direct gradient.

    The direct gradient correponds to a linear problem:
        argmin_delta ||exp(delta) @ chi - Y||^2
    where chi = Ks @ dual_weights.

    The Hessian is not just `chi.T @ chi` because of the exponential
    parametrization of deltas.

    Parameters
    ----------
    exp_delta_chi : array of shape (n_kernels, n_samples, n_targets)
        Precomputation of exp(delta) * (Ks @ dual_weights).
    Y : array of shape (n_samples, n_targets)
        Target data on the validation split.

    Returns
    -------
    XTXs : array of shape (n_targets, n_kernels, n_kernels)
        Hessian of the direct gradient.
    """
    backend = get_current_backend()

    XTXs = backend.matmul(backend.transpose(exp_delta_chi, (2, 0, 1)),
                          backend.transpose(exp_delta_chi, (2, 1, 0)))
    XTbs = backend.matmul(backend.transpose(exp_delta_chi, (2, 0, 1)),
                          backend.transpose(Y, (1, 0))[:, :, None])[:, :, 0]
    diagonal_view = backend.diagonal_view(XTXs, axis1=1, axis2=2)
    diagonal_view += diagonal_view + XTbs
    return XTXs
