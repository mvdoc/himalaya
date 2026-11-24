#!/usr/bin/env python
"""Profile the random-search solver on GPU, and time candidate optimizations.

This script does not change the library. It measures where the time goes in
``solve_multiple_kernel_ridge_random_search``, and micro-benchmarks alternative
implementations of the expensive steps, to decide which ones are worth
implementing. It runs with the "torch_cuda" backend (or "torch" on CPU, to
check that the script itself works).

Usage
-----
    python benchmarks/profile_random_search_gpu.py \
        --n_samples 3000 --n_targets 80000 --n_kernels 4 --n_iter 10 \
        --n_targets_batch 40000

Add ``--Y_in_cpu`` to keep the targets on CPU as with large datasets; the
breakdown then shows the time spent copying them to the GPU ("to_gpu").

Sections
--------
1. eigh         : GPU (cusolver / magma) versus CPU + transfer.
2. decomposition: eigh versus batched Cholesky, to get the same matrices.
3. predictions  : the (alphas x test x train) @ (train x targets) product and
                  the score, with and without TF32.
4. transfers    : GPU <-> CPU copies, pageable versus pinned memory.
5. solver       : end-to-end run with a breakdown per stage, and how much work
                  a refit deferred to the end of the search would save.
"""
import argparse
import time
from contextlib import contextmanager

import numpy as np
import torch

from himalaya.backend import set_backend
from himalaya.scoring import l2_neg_loss
from himalaya.kernel_ridge import solve_multiple_kernel_ridge_random_search
import himalaya.kernel_ridge._random_search as random_search_module


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(func, n_repeats=3):
    """Return the best wall time of func, after one untimed warmup call."""
    func()
    times = []
    for _ in range(n_repeats):
        sync()
        start = time.perf_counter()
        out = func()
        sync()
        times.append(time.perf_counter() - start)
        del out
    return min(times)


def rel_diff(a, b):
    return float((a - b).abs().max() / b.abs().max())


def make_data(n_samples, n_targets, n_kernels, dtype, seed=0):
    rng = np.random.RandomState(seed)
    Xs = [
        rng.randn(n_samples, n_samples // 2 + 100 * kk).astype(dtype)
        for kk in range(n_kernels)
    ]
    Ks = np.stack([X @ X.T for X in Xs])
    Y = rng.randn(n_samples, n_targets).astype(dtype) * 5
    for X in Xs:
        Y += X[:, :50] @ rng.randn(50, n_targets).astype(dtype)
    Y -= Y.mean(0)
    return Ks, Y


###############################################################################
def section_eigh(K, device):
    print("\n== 1. eigh of the (n_train, n_train) kernel ==")
    n = K.shape[0]
    K_cpu = K.cpu()
    print("  CPU (%d threads)            : %.3f s" %
          (torch.get_num_threads(), timeit(lambda: torch.linalg.eigh(K_cpu))))
    if device.type != "cuda":
        return

    def cpu_roundtrip():
        eigenvalues, V = torch.linalg.eigh(K.cpu())
        return eigenvalues.to(device), V.to(device)

    print("  CPU + transfers to/from GPU : %.3f s" % timeit(cpu_roundtrip))
    for library in ["cusolver", "magma"]:
        try:
            torch.backends.cuda.preferred_linalg_library(library)
            print("  GPU (%-8s)              : %.3f s" %
                  (library, timeit(lambda: torch.linalg.eigh(K))))
        except Exception as error:  # library not available in this build
            print("  GPU (%-8s)              : failed (%s)" % (library, error))
    torch.backends.cuda.preferred_linalg_library("default")
    print("  (n = %d, dtype = %s)" % (n, K.dtype))


def section_decomposition(Ktrain, Ktest, alphas):
    print("\n== 2. matrices = Ktest @ (Ktrain + alpha * Id)^-1, all alphas ==")
    n_train = Ktrain.shape[0]

    def with_eigh():
        eigenvalues, V = torch.linalg.eigh(Ktrain)
        Ktest_V = Ktest @ V
        weighting = 1 / (alphas[:, None] + eigenvalues)
        return (Ktest_V[None] * weighting[:, None, :]) @ V.T

    def with_cholesky():
        eye = torch.eye(n_train, dtype=Ktrain.dtype, device=Ktrain.device)
        regularized = Ktrain[None] + alphas[:, None, None] * eye
        L, info = torch.linalg.cholesky_ex(regularized)
        del regularized
        # solve (Ktrain + alpha * Id) @ X = Ktest.T, then transpose
        X = torch.cholesky_solve(Ktest.T[None].expand(len(alphas), -1, -1), L)
        return X.transpose(1, 2), info

    def with_cholesky_loop():
        # same, one alpha at a time (lower memory)
        out = []
        eye = torch.eye(n_train, dtype=Ktrain.dtype, device=Ktrain.device)
        for alpha in alphas:
            L, info = torch.linalg.cholesky_ex(Ktrain + alpha * eye)
            out.append(torch.cholesky_solve(Ktest.T, L).T)
        return torch.stack(out)

    reference = with_eigh()
    print("  eigh + products         : %.3f s" % timeit(with_eigh))
    for name, func in [("batched Cholesky + solve", with_cholesky),
                       ("looped Cholesky + solve ", with_cholesky_loop)]:
        try:
            out = func()
            info = None
            if isinstance(out, tuple):
                out, info = out
            ok = torch.isfinite(out).all(2).all(1)
            if info is not None:
                ok &= info == 0
            diff = rel_diff(out[ok], reference[ok]) if ok.any() else np.nan
            print("  %s: %.3f s   (max rel diff vs eigh on %d/%d valid "
                  "alphas: %.1e)" % (name, timeit(func), int(ok.sum()),
                                     len(alphas), diff))
            del out
        except Exception as error:  # e.g. out of memory
            print("  %s: failed (%s)" % (name, str(error)[:80]))
    return reference


def section_predictions(matrix, Ytrain, Ytest, device):
    print("\n== 3. predictions and scores, for one split and target batch ==")
    n_alphas, n_test, n_train = matrix.shape
    matrix_2d = matrix.reshape(-1, n_train)

    def predict():
        return (matrix_2d @ Ytrain).reshape(n_alphas, n_test, -1)

    predictions = predict()
    print("  shapes: matrix %s, Ytrain %s" %
          (tuple(matrix.shape), tuple(Ytrain.shape)))
    print("  matmul                  : %.3f s" % timeit(predict))
    print("  l2_neg_loss             : %.3f s" %
          timeit(lambda: l2_neg_loss(Ytest, predictions)))

    def score_inplace():
        tmp = predict()
        tmp -= Ytest
        tmp *= tmp
        return -tmp.sum(1) / n_test

    print("  matmul + in-place score : %.3f s" % timeit(score_inplace))

    if device.type == "cuda":
        scores = l2_neg_loss(Ytest, predictions)
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            print("  matmul with TF32        : %.3f s" % timeit(predict),
                  end="")
            print("   (max rel diff of scores: %.1e, NOT the same precision)"
                  % rel_diff(l2_neg_loss(Ytest, predict()), scores))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False


def section_transfers(n_samples, n_targets, dtype, device):
    print("\n== 4. transfers of an (n_samples, n_targets) array ==")
    if device.type != "cuda":
        print("  skipped (no GPU)")
        return
    on_gpu = torch.randn(n_samples, n_targets, dtype=dtype, device=device)
    pageable = torch.empty(n_samples, n_targets, dtype=dtype)
    try:
        pinned = torch.empty(n_samples, n_targets, dtype=dtype,
                             pin_memory=True)
    except Exception as error:
        print("  pinned memory not available (%s)" % error)
        pinned = None

    size_gb = on_gpu.numel() * on_gpu.element_size() / 1e9
    print("  size: %.2f GB" % size_gb)
    print("  GPU -> CPU, new array   : %.3f s" % timeit(lambda: on_gpu.cpu()))
    print("  GPU -> CPU, pageable    : %.3f s" %
          timeit(lambda: pageable.copy_(on_gpu)))
    print("  CPU -> GPU, pageable    : %.3f s" %
          timeit(lambda: pageable.to(device)))
    if pinned is not None:
        print("  GPU -> CPU, pinned      : %.3f s" %
              timeit(lambda: pinned.copy_(on_gpu)))
        print("  CPU -> GPU, pinned      : %.3f s" %
              timeit(lambda: pinned.to(device)))


class StageTimer:
    """Accumulate the (synchronized) wall time of wrapped functions."""

    def __init__(self):
        self.times = {}
        self.counts = {}

    def wrap(self, name, func):
        def wrapped(*args, **kwargs):
            sync()
            start = time.perf_counter()
            out = func(*args, **kwargs)
            sync()
            self.times[name] = (self.times.get(name, 0) + time.perf_counter() -
                                start)
            self.counts[name] = self.counts.get(name, 0) + 1
            return out

        return wrapped


@contextmanager
def patched(obj, name, new):
    old = getattr(obj, name)
    setattr(obj, name, new)
    try:
        yield
    finally:
        setattr(obj, name, old)


def section_solver(backend, Ks, Y, alphas, args):
    print("\n== 5. solve_multiple_kernel_ridge_random_search ==")
    params = dict(n_iter=args.n_iter, alphas=alphas, cv=args.cv,
                  return_weights="dual", random_state=0, progress_bar=False,
                  n_targets_batch=args.n_targets_batch,
                  n_alphas_batch=args.n_alphas_batch,
                  Y_in_cpu=args.Y_in_cpu)
    if args.Y_in_cpu:
        Y = Y.cpu()

    def run(**kwargs):
        return solve_multiple_kernel_ridge_random_search(
            Ks, Y, **{**params, **kwargs})

    run(n_iter=1)  # warmup
    for return_weights in ["dual", None]:
        sync()
        start = time.perf_counter()
        results = run(return_weights=return_weights)
        sync()
        print("  total, return_weights=%-6r: %.2f s" %
              (return_weights, time.perf_counter() - start))
        if return_weights == "dual":
            cv_scores = results[2]

    # breakdown per stage (synchronizing after each call adds some overhead)
    timer = StageTimer()
    score_func = timer.wrap("score_func", l2_neg_loss)
    with patched(backend, "eigh", timer.wrap("eigh", backend.eigh)), \
            patched(backend, "matmul",
                    timer.wrap("matmul", backend.matmul)), \
            patched(backend, "to_cpu",
                    timer.wrap("to_cpu", backend.to_cpu)), \
            patched(backend, "to_gpu",
                    timer.wrap("to_gpu", backend.to_gpu)), \
            patched(random_search_module, "_refit_dual_weights",
                    timer.wrap("refit (total)",
                               random_search_module._refit_dual_weights)):
        sync()
        start = time.perf_counter()
        run(score_func=score_func)
        sync()
        total = time.perf_counter() - start
    print("  breakdown (total %.2f s with synchronization):" % total)
    for name in sorted(timer.times, key=timer.times.get, reverse=True):
        print("    %-14s %7.2f s  %5.1f %%  (%d calls)" %
              (name, timer.times[name], 100 * timer.times[name] / total,
               timer.counts[name]))
    print("    (eigh, matmul and to_cpu include the calls made in the refit)")

    # work that a refit deferred to the end of the search would save
    cv_scores = cv_scores.cpu().numpy()
    epsilon = np.finfo(cv_scores.dtype).eps
    best = np.full(cv_scores.shape[1], -np.inf, dtype=cv_scores.dtype)
    best_iteration = np.zeros(cv_scores.shape[1], dtype=int)
    n_refit_targets, n_refit_iterations = 0, 0
    for ii, scores in enumerate(cv_scores):
        mask = scores > best + epsilon
        best[mask] = scores[mask]
        best_iteration[mask] = ii
        n_refit_targets += mask.sum()
        n_refit_iterations += mask.any()
    print("  refit now     : %d eigh, %d target solves (%.1f x n_targets)" %
          (n_refit_iterations, n_refit_targets,
           n_refit_targets / cv_scores.shape[1]))
    print("  refit deferred: %d eigh, %d target solves" %
          (len(np.unique(best_iteration)), cv_scores.shape[1]))


###############################################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", default=None,
                        help="torch_cuda (default if available) or torch")
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--n_targets", type=int, default=80000)
    parser.add_argument("--n_kernels", type=int, default=4)
    parser.add_argument("--n_alphas", type=int, default=20)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--cv", type=int, default=10,
                        help="number of folds (e.g. the number of runs)")
    parser.add_argument("--n_targets_batch", type=int, default=None)
    parser.add_argument("--n_alphas_batch", type=int, default=None)
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--Y_in_cpu", action="store_true",
                        help="keep Y on CPU in the solver (section 5)")
    parser.add_argument("--sections", default="1,2,3,4,5")
    args = parser.parse_args()

    if args.backend is None:
        args.backend = "torch_cuda" if torch.cuda.is_available() else "torch"
    backend = set_backend(args.backend)
    dtype = "float64" if args.float64 else "float32"
    sections = [int(ss) for ss in args.sections.split(",")]

    print("backend=%s torch=%s dtype=%s" %
          (args.backend, torch.__version__, dtype))
    if torch.cuda.is_available():
        print("GPU: %s, CUDA %s" %
              (torch.cuda.get_device_name(0), torch.version.cuda))
    print(vars(args))

    Ks, Y = make_data(args.n_samples, args.n_targets, args.n_kernels, dtype)
    Ks, Y = backend.asarray(Ks), backend.asarray(Y)
    device = Ks.device
    alphas = backend.asarray_like(np.logspace(1, 20, args.n_alphas), ref=Ks)

    # one cross-validation split, and one batch of targets
    n_test = args.n_samples // args.cv
    K = Ks.mean(0)
    Ktrain = K[n_test:, n_test:].contiguous()
    Ktest = K[:n_test, n_test:].contiguous()
    batch = slice(0, args.n_targets_batch or args.n_targets)
    Ytrain, Ytest = Y[n_test:, batch], Y[:n_test, batch]

    with torch.no_grad():
        if 1 in sections:
            section_eigh(Ktrain, device)
        if 2 in sections or 3 in sections:
            matrix = section_decomposition(Ktrain, Ktest, alphas)
        if 3 in sections:
            section_predictions(matrix, Ytrain, Ytest, device)
        del K, Ktrain, Ktest
        if 4 in sections:
            section_transfers(args.n_samples, min(args.n_targets, 20000),
                              Ks.dtype, device)
        if 5 in sections:
            section_solver(backend, Ks, Y, alphas, args)


if __name__ == "__main__":
    main()
