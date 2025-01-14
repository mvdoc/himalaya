Changelog
=========

Development Version
-------------------

Version 0.4.1
-------------
(*February 2023*)

- FIX avoid error in :class:`~himalaya.kernel_ridge.MultipleKernelRidgeCV`
  with ``solver_params(return_alphas=True)``.
- ENH add ``fit_intercept`` in
  :class:`~himalaya.kernel_ridge.MultipleKernelRidgeCV`.
- FIX torch 1.13.1 requires tensor masks to be on the same device as tensors.

Version 0.4.0
-------------
(*June 2022*)

- DOC explain how to implement a winner-take-all model.
- FIX comply with most recent scikit-learn's check_estimator.
- FIX avoid an indexing error in the hypergradient solver, when early stopping
  after different numbers of iterations for different batches.

Version 0.3.6
-------------
(*April 2022*)

- DOC improve documentation website, add estimator flowchart.
- TST improve test robustness.
- ENH add batching over targets in
  :func:`~himalaya.kernel_ridge.predict_weighted_kernel_ridge`.
- ENH add ``solver="auto"`` in :class:`~himalaya.kernel_ridge.KernelRidge`,
  which switches solver based on the presence of a separate alpha per target.

Version 0.3.5
-------------
(*February 2022*)

- MNT speed up examples on CPU, to build the doc faster on github actions.
- ENH add batching over targets in :class:`~himalaya.ridge.Ridge`,
  :class:`~himalaya.kernel_ridge.KernelRidge`, and
  :class:`~himalaya.kernel_ridge.WeightedKernelRidge`.
- ENH add warnings to guide the user between using
  :class:`~himalaya.ridge.Ridge` or
  :class:`~himalaya.kernel_ridge.KernelRidge`.
- ENH add user-friendly errors when the number of samples is inconsistent.
- ENH raise ValueError if the indices in cross-validation exceed number of
  samples.

Version 0.3.4
-------------
(*November 2021*)

- FIX :class:`~himalaya.ridge.Ridge` with ``n_samples < n_targets``.
- FIX update of alphas when ``local_alpha=False`` in
  :class:`~himalaya.kernel_ridge.MultipleKernelRidgeCV`.
- EXA refactor examples with new
  :func:`~himalaya.utils.generate_multikernel_dataset` function.
- MNT add github actions for running tests, building and publishing the doc,
  and publishing to PyPI.

Version 0.3.3
-------------
(*November 2021*)

- FIX :class:`~himalaya.kernel_ridge.KernelRidge` with
  ``n_samples < n_targets``.
- FIX random search with single alpha in
  :class:`~himalaya.kernel_ridge.MultipleKernelRidgeCV`.

Version 0.3.2
-------------
(*November 2021*)

- ENH add :func:`~himalaya.scoring.r2_score_split_svd` scoring function.
- ENH add :func:`~himalaya.scoring.correlation_score_split` scoring function.
- ENH add ``split`` parameter to the ``score`` method in
  :class:`~himalaya.kernel_ridge.WeightedKernelRidge`,
  :class:`~himalaya.kernel_ridge.MultipleKernelRidgeCV`, and
  :class:`~himalaya.ridge.GroupRidgeCV`.
- ENH add ``force_cpu`` parameter in all estimators.
- FIX remove deprecation warnings for cupy v9.
- DOC mention that pytorch 1.9+ is preferred.

Version 0.3.1
-------------
(*September 2021*)

- MNT Rename :class:`~himalaya.ridge.BandedRidgeCV` into
  :class:`~himalaya.ridge.GroupRidgeCV` (both names are available).
- ENH improve robustness to noise in the cross-validation scores.
- ENH start the random search with equal weights in
  :class:`~himalaya.kernel_ridge.MultipleKernelRidgeCV`
  and :class:`~himalaya.ridge.GroupRidgeCV`.
- FIX remove deprecation warnings with pytorch 1.8.
- TST improve test coverage.

Version 0.3.0
-------------
(*April 2021*)

- ENH add ``fit_intercept`` parameter in :class:`~himalaya.ridge.Ridge`,
  :class:`~himalaya.ridge.RidgeCV`, and :class:`~himalaya.ridge.BandedRidgeCV`.
- ENH add ``fit_intercept`` parameter in
  :class:`~himalaya.kernel_ridge.KernelRidge`,
  :class:`~himalaya.kernel_ridge.KernelRidgeCV`,
  :func:`~himalaya.kernel_ridge.solve_multiple_kernel_ridge_gradient_descent`,
  and :func:`~himalaya.kernel_ridge.solve_multiple_kernel_ridge_random_search`.
- ENH add :class:`~himalaya.kernel_ridge.KernelCenterer`.
- ENH allow change of backend midscript.
- ENH Add option to return selected alpha values in
  :func:`~himalaya.kernel_ridge.solve_multiple_kernel_ridge_random_search`.

Version 0.2.0
-------------
(*December 2020*)

Version 0.1.0
-------------
(*March 2020*)
