# NaiveMFLFRMixDClust

**Bayesian mixture model clustering for heterogeneous data via Variational Inference.**

`NaiveMFLFRMixDClust` fits finite mixture models to datasets containing any combination of
continuous, count, and categorical variables, using Coordinate Ascent Variational
Inference (CAVI). It simultaneously selects the number of clusters, identifies which
features actually drive the cluster structure, and produces diagnostic plots — all from
a single model fit.

---

## Features

- **Mixed data types**: Gaussian (continuous), Poisson (count), Gamma (positive
  continuous), and Multinomial (categorical) variables can be freely combined in the
  same model.
- **Automatic cluster order selection**: a sparse symmetric Dirichlet prior on an
  overfitted mixture automatically empties redundant components; no need to specify
  *K* in advance.
- **Post-hoc cluster refinement**: a deterministic pruning-and-merging step removes
  residual spurious components left by the variational approximation, producing a
  clean final partition.
- **Two feature-relevance models**
  - `SFRM()`: each feature has one shared relevance probability across all clusters.
  - `LFRM()`: each feature can be relevant in some clusters and irrelevant in others,
    capturing finer subgroup-specific signal.
- **Post-hoc EIG filter**: an Expected Information Gain criterion controls the False
  Discovery Rate for feature selection.
- **Multi-start screening**: automatic warm-start strategy selects the best
  initialization before the full CAVI run.
- **Out-of-sample prediction**: assign new observations to the learned cluster
  structure via `predict_proba`.
- **8 built-in visualizations**: covering the full analysis workflow from convergence
  diagnostics to cluster-specific feature profiles.

---

## Installation

### From GitHub

```julia
using Pkg
Pkg.add(url="https://github.com/toussile/NaiveMFLFRMixDClust.jl.git")
```

### Local development install

If you have cloned the repository locally:

```julia
using Pkg
Pkg.develop(path="/path/to/NaiveMFLFRMixDClust.jl")
```

### Updating

```julia
Pkg.update("NaiveMFLFRMixDClust")
```

Or in the Julia REPL package manager (`]` key):

```
] update NaiveMFLFRMixDClust
```

### From the Julia General Registry (planned)

Once registered:

```julia
Pkg.add("NaiveMFLFRMixDClust")
```

---

## Example: clustering a synthetic patient cohort

150 patients, 3 subtypes, 5 informative features (Gaussian, Gaussian, Poisson, Gamma,
Multinomial) + 2 uninformative noise features.

```julia
using NaiveMFLFRMixDClust

# ── 1. Generate the synthetic dataset ─────────────────────────────────────
cohort        = simulate_synthetic_cohort()   # seed=2026, n=150 by default
dataset       = cohort.data
feature_names = cohort.feature_names

# ── 2. Fit the model ───────────────────────────────────────────────────────
results = mixClust(dataset, 10;
                   model_setting = LFRM(),
                   u0            = 0.01,
                   max_iter      = 500,
                   tol           = 1e-4,
                   prune         = true,
                   n_init        = 10,
                   max_iter_init = 10)

println("Estimated number of clusters: K̂ = ", results.n_clusters)   # → 3

# ── 3. Feature selection via EIG ───────────────────────────────────────────
τ      = 0.10
eig    = compute_eig(results.margins, results.w, results.pip)
active = filter_features(eig, τ)
println("Active features (EIG ≥ $τ): ", feature_names[active])
# → ["age", "mutations", "tumour_size", "histology"]

# ── 4. Hard cluster assignments ────────────────────────────────────────────
println("Cluster sizes: ", results.cluster_sizes)

# ── 5. Visualizations ─────────────────────────────────────────────────────
plot_elbo(results)                            # convergence check
plot_cluster_sizes(results)                   # individuals per cluster
plot_assignment_confidence(results)           # crisp assignments?
plot_pips(results; feature_names)             # global PIPs
plot_eig(eig, τ; feature_names)              # EIG bar chart
plot_local_pips(results; feature_names)       # per-cluster PIPs (LFRM)
plot_profiles(results, dataset;               # feature z-scores per cluster
              feature_names, threshold=τ)
plot_assignments(results, dataset;            # 2-D scatter coloured by cluster
                 x_feature=1, y_feature=3,
                 feature_names)

# ── 6. Out-of-sample prediction ────────────────────────────────────────────
w_new  = predict_proba(results, dataset)
ll_new = predictive_log_likelihood(results, dataset)
println("Log-predictive density: ", round(ll_new, digits=1))
```

The UCI Heart Disease dataset used in the paper is bundled with the package
and can be loaded directly — see [Bundled datasets](#bundled-datasets) below.

---

## Example: Iris dataset (4 Gaussian features)

```julia
using NaiveMFLFRMixDClust, RDatasets, Statistics, Random

# Load and standardize
iris          = RDatasets.dataset("datasets", "iris")
standardize(v) = (v .- mean(v)) ./ std(v)
data          = [standardize(Vector(iris[:, col]))
                 for col in [:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]]
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

Random.seed!(2026)
results = mixClust(data, 8;
                   model_setting = SFRM(),
                   u0            = 0.01,
                   max_iter      = 500,
                   tol           = 1e-5,
                   prune         = true,
                   n_init        = 10,
                   max_iter_init = 10)

println("Estimated clusters: ", results.n_clusters)   # → 3  (true: 3)

eig    = compute_eig(results.margins, results.w, results.pip)
active = filter_features(eig, 0.10)
println("Active features: ", feature_names[active])
# → all 4 features active; petal_length and petal_width have highest EIG

# ── Visualizations ─────────────────────────────────────────────────────
plot_elbo(results)                            # convergence check
plot_cluster_sizes(results)                   # individuals per cluster
plot_assignment_confidence(results)           # crisp assignments?
plot_pips(results; feature_names)             # global PIPs
plot_eig(eig, τ; feature_names)              # EIG bar chart
plot_local_pips(results; feature_names)       # per-cluster PIPs (LFRM)
plot_profiles(results, data;               # feature z-scores per cluster
              feature_names, threshold=τ)
plot_assignments(results, data;            # 2-D scatter coloured by cluster
                 x_feature=1, y_feature=3,
                 feature_names)

```

Expected output: $\hat{K} = 3$, ARI ≈ 0.72, ELBO monotone.
`petal_length` (EIG ≈ 0.94) and `petal_width` (EIG ≈ 0.87) are the most
discriminative features, consistent with the classical analysis of this dataset.

---

## Data format

The `dataset` argument to `mixClust` is a `Vector{Any}` with one element per feature:

| Feature type | Julia type | Notes |
| :--- | :--- | :--- |
| Continuous (Gaussian) | `Vector{Float64}` | Standardize to mean 0, variance 1 |
| Count (Poisson) | `Vector{Float64}` | Non-negative integers as `Float64` |
| Positive continuous (Gamma) | `Vector{Float64}` | Strictly positive values |
| Categorical (Multinomial) | `Vector{Vector{Int}}` | One-hot vectors of length `n_levels` |

The type of each margin is inferred automatically: `Vector{Vector{Int}}` → Multinomial;
non-negative integer-valued `Vector{Float64}` → Poisson; strictly positive
`Vector{Float64}` → Gamma; otherwise Gaussian. Features of different types can appear
in any order.

To override the auto-detection for specific features, pass a `feature_types` vector:

```julia
# Force feature 3 to Gaussian even though all values are positive
results = mixClust(dataset, 10;
                   feature_types = [nothing, nothing, :gaussian, nothing, nothing])
```

Each entry is `:gaussian`, `:poisson`, `:gamma`, `:multinomial`, or `nothing` (auto-detect).
`mixClust` will print the resolved type of every feature at `@info` level so you can
verify the detection before the CAVI run starts.

---

## Bundled datasets

### `simulate_synthetic_cohort()`

Generates the synthetic clinicogenomic patient cohort used in the paper (150 patients,
3 subtypes, 5 informative + 2 noise features). Fully reproducible via a fixed seed:

```julia
cohort = simulate_synthetic_cohort()          # seed=2026, n=150 by default
cohort = simulate_synthetic_cohort(seed=42, n=300)  # custom seed and size
# cohort.data          — Vector{Any} with 7 features, ready for mixClust
# cohort.labels        — Vector{Int}: true cluster assignment (1, 2, or 3)
# cohort.feature_names — ["age", "bmi", "mutations", "tumour_size", "histology", ...]
```

### `load_heart_disease()`

The UCI Heart Disease (Cleveland) dataset (297 individuals, 13 features) is included
in the package and ready to use without any external files or downloads:

```julia
hd = load_heart_disease()
# hd.data          — Vector{Any} with 13 features, ready for mixClust
# hd.labels        — Vector{Int}: 0 = healthy, 1 = disease
# hd.feature_names — ["age", "sex", "cp", ...]

results = mixClust(hd.data, 10;
                   model_setting = SFRM(),
                   u0            = 0.01,
                   max_iter      = 2000,
                   tol           = 1e-5,
                   prune         = true)

eig    = compute_eig(results.margins, results.w, results.pip)
active = filter_features(eig, 0.10)
println("Active features: ", hd.feature_names[active])
```

Continuous features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) are standardized;
categorical and binary features are one-hot encoded as `Vector{Vector{Int}}`.

---

## API Reference

### Model fitting

```julia
results = mixClust(dataset, K_max;
                   model_setting  = SFRM(),   # or LFRM()
                   u0             = 0.01,     # u⁽⁰⁾: Dirichlet sparsity (< 1; smaller → more pruning)
                   delta_prior    = (1.0, 1.0), # Beta prior for relevance indicators
                   max_iter       = 500,      # CAVI iterations for the final run
                   tol            = 1e-4,     # convergence tolerance
                   prune          = true,     # post-hoc pruning & merging
                   size_threshold = 0.02,     # min relative cluster size to retain
                   merge_threshold= 0.85,     # cosine similarity above which clusters are merged
                   beta_estimation= :two_stage, # or :iterative
                   n_init         = 10,       # number of screening restarts
                   max_iter_init  = 10,       # iterations per screening run
                   feature_types  = nothing)  # optional override per feature (see below)
```

`results` is a `MixClustResult` struct.

**Stored fields:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `results.w` | `n × K̂ Matrix{Float64}` | Soft cluster assignments (rows sum to 1) |
| `results.labels` | `Vector{Int}` | Hard assignment: `argmax` of each row of `w` |
| `results.pip` | `n × p Matrix{Float64}` | Posterior Inclusion Probabilities |
| `results.margins` | `Vector{AbstractMargin}` | Fitted margin objects (one per feature) |
| `results.elbo_history` | `Vector{Float64}` | ELBO values over iterations |
| `results.u_star` | `Vector{Float64}` | Variational Dirichlet parameters (length K̂) |
| `results.delta_star` | `Array{Float64}` | Variational Beta parameters for relevance |

**Computed properties** (available via dot-syntax, no extra code needed):

| Property | Type | Description |
| :--- | :--- | :--- |
| `results.n_clusters` | `Int` | Estimated number of active clusters K̂ |
| `results.n_obs` | `Int` | Number of individuals n |
| `results.n_features` | `Int` | Number of features p |
| `results.cluster_sizes` | `Vector{Int}` | Number of individuals per cluster |

### Model settings

| Setting | Description |
| :--- | :--- |
| `SFRM()` | Shared Feature Relevance: one relevance probability per feature |
| `LFRM()` | Local Feature Relevance: one relevance probability per (feature, cluster) pair |

Use `LFRM()` when you expect some features to separate only a subset of clusters.
Use `SFRM()` as a faster baseline or when relevance is expected to be uniform.

### Feature relevance

```julia
eig     = compute_eig(results.margins, results.w, results.pip)
# → Vector{Float64} of length p

active  = filter_features(eig, 0.10)
# → Vector{Int} of indices where EIG ≥ threshold
```

### Out-of-sample prediction

```julia
w_new  = predict_proba(results, new_dataset)
# → (n_new × K̂) matrix; rows sum to 1

ll_new = predictive_log_likelihood(results, new_dataset)
# → scalar total log-predictive density
```

### Visualization

All plot functions accept optional keyword arguments forwarded to `Plots.jl`.

| Function | What it shows |
| :--- | :--- |
| `plot_elbo(results)` | ELBO convergence curve |
| `plot_cluster_sizes(results)` | Number of individuals per cluster |
| `plot_assignment_confidence(results; threshold=0.9)` | Histogram of max posterior responsibility |
| `plot_pips(results; feature_names)` | Bar chart of mean PIP per feature |
| `plot_eig(eig, τ; feature_names)` | Bar chart of EIG with threshold line |
| `plot_local_pips(results; feature_names)` | K × p heatmap of per-cluster PIPs (LFRM) |
| `plot_profiles(results, dataset; feature_names)` | Feature z-scores per cluster vs. background |
| `plot_assignments(results, dataset; x_feature, y_feature, feature_names)` | 2-D scatter coloured by cluster |

**Recommended workflow:**

```
plot_elbo              → convergence ok?
plot_cluster_sizes     → how many individuals per group?
plot_assignment_confidence → are assignments crisp?
plot_pips / plot_eig   → which features matter globally?
plot_local_pips        → which features matter per cluster? (LFRM)
plot_profiles          → what does each subtype look like?
plot_assignments       → where do individuals fall in feature space?
```

---

## Citation

If you use `NaiveMFLFRMixDClust` in your research, please cite:

```bibtex
@article{toussile2026mixclustvi,
  author  = {Toussile, Wilson and Fotso, Simeon and Takam Soh, Patrice},
  title   = {Local Feature Relevance and Cluster Order Selection in Model-Based
             Clustering for Mixed Data via Variational Inference},
  journal = {Computational Statistics \& Data Analysis},
  year    = {2026},
  note    = {Submitted}
}
```

---

## Roadmap

- [ ] Negative Binomial margin for overdispersed count data
- [ ] Block-diagonal covariance for continuous features (relax within-cluster independence)
- [ ] Native missing-data handling within CAVI

---

## License

[MIT License](LICENSE)
