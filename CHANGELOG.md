# Changelog

All notable changes to NaiveMFLFRMixDClust (formerly MixClustVIjl) are documented here.
Versioning follows [Semantic Versioning](https://semver.org):
`MAJOR.MINOR.PATCH` - breaking changes bump MAJOR (or MINOR while pre-1.0),
new features bump MINOR, bug fixes bump PATCH.

---

## [0.5.0] - 2026-10-02

### Renamed (terminology: uninformative observations, not robust clustering)

The package no longer refers to outliers or robust clustering: observations flagged by
the coordinate inactivation rate are *uninformative* (background class 0). Old names
still work for one release and emit a deprecation warning.

| Old (deprecated) | New |
|---|---|
| `detect_outliers` | `detect_uninformative_observations` (new method taking only `pip`) |
| `outlier_indices` | `uninformative_indices` |
| `robust_cluster_assignments` | `extended_cluster_assignments` |
| `cluster_indices(...; robust=...)` | `cluster_indices(...; exclude_uninformative=...)` |
| margin keyword `robust` | `median_based` (Gaussian, LogNormal, Exponential, Gamma, Poisson), `smoothed` (Bernoulli, Multinomial) |

- Experiment files renamed: `05_outlier_absorption.jl` → `05_uninformative_absorption.jl`,
  `sim_outlier_detection.jl` → `sim_uninformative_detection.jl`; scripts, tests and
  docstrings updated accordingly.

### Bug fixes

- `cluster_indices(...; exclude_uninformative=true)` applies `threshold` again (since 0.4.0,
  the call went through the `:map_tau` rule, which ignores the threshold).
- The error message for an unknown detection mode now shows the mode.

---

## [0.4.0] - 2026-10-02

### Breaking changes

- **Default Dirichlet prior is now `u0 = 1/K`** (previously `u0 = 0.01`). The prior on the
  mixing weights is `Dir(ω; u⁽⁰⁾·1_K)`, i.e. the concentration vector
  `(u⁽⁰⁾, …, u⁽⁰⁾)`, as in the paper. `mixClust(...; u0 = nothing)` resolves to `1/K`.
  A warning is issued when `u0 > 1/K` (previously when `u0 ≥ 1`). Pass `u0 = 0.01`
  explicitly to reproduce 0.3.0 defaults.
- **`robust_cluster_assignments` is now an alias of `extended_cluster_assignments`**, whose
  defaults are `mode = :map_tau` and `threshold = :auto`.

### New features

- `predict_pips`: posterior inclusion probabilities for new observations.
- `calibrate_tau_inact`, `detect_uninformative_observations`: data-driven threshold for the
  coordinate inactivation rate and detection of uninformative observations.
- `extended_cluster_assignments`, `extended_responsibilities`, and the result properties
  `extended_labels` and `extended_responsibilities` (assignment to the background class 0).

### Performance

- The CAVI loop is about 15× faster. Variables assigned in both branches of the
  initialisation were captured by closures and boxed, which made the inner loops
  type-unstable; the iterations now run in a separate function (`_cavi_loop!`).
  Results are bitwise identical.

### Experiments

- `01_order_selection.jl` now sweeps the initial overfitting level `K_max` with
  `u0 = 1/K_max`; tutorials and validation report updated.

---

## [0.3.0] - 2026-09-29

### Breaking changes

- **Package renamed from `MixClustVIjl` to `NaiveMFLFRMixDClust`** (repository
  `toussile/NaiveMFLFRMixDClust.jl`) to match the accompanying paper. Replace
  `using MixClustVIjl` with `using NaiveMFLFRMixDClust`. The package UUID is unchanged.
- **`prune_and_merge_clusters` replaced by `compact_clusters`**. `mixClust` now
  compacts empty components by default (`compact=true`).
- **`MixClustResult` gains an `omega_history` field** (T × K trajectory of
  E_q[ω_k]). A backward-compatible constructor without it is provided.

### New features

- New margins: `BernoulliMargin`, `LogNormalMargin`, `ExponentialMargin`.
- Expected Hellinger Distance: `compute_ehd`, `compute_local_ehd`, `hellinger_divergence`.
- Background-component diagnostics: `compute_pi_0`, `coordinate_inactivation_rate`,
  `detect_outliers`, `robust_cluster_assignments`, `outlier_indices`, `cluster_indices`,
  and the result properties `inactivation_rate` and `pi_0`.
- `n_active_clusters`, `adjusted_rand_index`.
- Multi-start initialisation (`n_init`, `max_iter_init`) and explicit
  `feature_types` in `mixClust`.
- New plots: `plot_ehd`, `plot_local_ehd`, `plot_local_eig`,
  `plot_mixing_weights_evolution`, `plot_omega_history`.
- `experiments/`: simulation suite, real-data scripts and tutorials reproducing
  the results of the paper.

---

## [0.2.0] - 2026-07-05

### Breaking changes

- **`results.alpha_star` renamed to `results.u_star`** - aligns with the paper
  notation where $\bm{u} = (u_1, \dots, u_K)^T$ denotes the variational Dirichlet
  parameter for mixing proportions. Any code accessing `results.alpha_star` must be
  updated to `results.u_star`.
- **`alpha_0` keyword renamed to `u0`** in `mixClust(...)` - aligns with the paper
  notation $u^{(0)}$ for the symmetric Dirichlet hyperparameter. Any call using
  `mixClust(...; alpha_0=...)` must be updated to `mixClust(...; u0=...)`.

### Bug fixes

- **ELBO monotonicity** - the ELBO sequence is now guaranteed non-decreasing at
  every CAVI iteration. Two root causes were fixed:
  - The KL divergence between variational and prior distributions for margin
    parameters (`kl_from_prior`) was missing from the ELBO computation. Implemented
    for all four margin types (Gaussian/NIG, Poisson/Gamma, Gamma, Multinomial/Dirichlet).
  - The ELBO is now computed *before* `update_margin!`, using refreshed expectations
    from the just-updated $\bm{u}$ and $\bm{\delta}$ parameters, ensuring the logged
    sequence corresponds to a consistent set of variational parameters.

### Improvements

- README: Iris dataset example now loads data via `RDatasets.jl` instead of
  hardcoded matrix.
- README: `K_fit` renamed to `K_max` throughout examples for clarity.
- README: Features section now explicitly lists post-hoc cluster refinement
  (pruning & merging) as a distinct step.

---

## [0.1.0] - 2026-07-04

Initial release. Core features:

- CAVI inference for finite overfitted mixtures on heterogeneous data
  (Gaussian, Poisson, Gamma, Multinomial margins).
- Shared Feature Relevance Model (`SFRM`) and Local Feature Relevance Model (`LFRM`).
- Sparse Dirichlet prior for automatic cluster order selection.
- Post-hoc cluster refinement: size-based pruning and cosine-similarity merging.
- Post-hoc feature selection via Expected Information Gain (EIG).
- Multi-start screening strategy.
- Out-of-sample prediction (`predict_proba`, `predictive_log_likelihood`).
- 8 built-in diagnostic visualizations.
- Bundled datasets: UCI Heart Disease and synthetic clinicogenomic cohort.
