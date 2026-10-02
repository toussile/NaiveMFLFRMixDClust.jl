"""
    AbstractMargin

Abstract supertype for all feature-margin models.
Each concrete subtype (e.g. [`GaussianMargin`](@ref), [`PoissonMargin`](@ref),
[`GammaMargin`](@ref), [`MultinomialMargin`](@ref)) stores the prior hyperparameters,
the variational posterior parameters (one set per cluster), and the fixed background
parameters estimated from the full dataset.
"""
abstract type AbstractMargin end

"""
    ModelSetting

Abstract supertype for the two feature-relevance models.
Use [`SFRM`](@ref) or [`LFRM`](@ref) as the `model_setting` argument to [`mixClust`](@ref).
"""
abstract type ModelSetting end

"""
    SFRM()

**Shared Feature Relevance Model.** Each feature has a single relevance
probability shared across all clusters. The relevance indicator `sⱼ` follows
a Bernoulli distribution with a Beta-distributed probability common to every cluster.

Use this model as a fast baseline or when the expectation is that a feature is
either universally informative or universally uninformative.

See also: [`LFRM`](@ref), [`mixClust`](@ref).
"""
struct SFRM <: ModelSetting end

"""
    LFRM()

**Local Feature Relevance Model.** Each feature–cluster pair `(j, k)` has its own
relevance probability, so a feature can be relevant in some clusters and irrelevant
in others. The relevance indicator `sᵢⱼ` depends on the cluster assignment of
individual `i`.

Use this model when you expect subgroup-specific signal: for example, a biomarker
that separates two out of three subtypes but is uninformative for the third.

See also: [`SFRM`](@ref), [`mixClust`](@ref).
"""
struct LFRM <: ModelSetting end

"""
    MixClustResult

Result returned by [`mixClust`](@ref) (and [`compact_clusters`](@ref)).

# Fields

## Stored fields

| Field | Type | Description |
|:--- |:--- |:--- |
| `w` | `Matrix{Float64}` (n × K̂) | Soft cluster assignments; each row is a probability vector summing to 1 |
| `labels` | `Vector{Int}` (length n) | Hard assignment: `labels[i] = argmax(w[i, :])` |
| `pip` | `Matrix{Float64}` (n × p) | Posterior Inclusion Probabilities `γᵢⱼ ∈ (0,1)` |
| `u_star` | `Vector{Float64}` (length K̂) | Variational Dirichlet parameters `u★ₖ = u⁽⁰⁾ + Σᵢ wᵢₖ` |
| `delta_star` | `Array{Float64}` | Variational Beta parameters for relevance: shape `(p,2)` under SFRM, `(K̂,p,2)` under LFRM |
| `margins` | `Vector{AbstractMargin}` (length p) | Fitted margin objects, one per feature |
| `elbo_history` | `Vector{Float64}` | ELBO value at each CAVI iteration (useful for convergence diagnostics) |
| `omega_history` | `Matrix{Float64}` | Trajectory of expected mixing weights E_q[ω_k] across iterations (T × K) |

## Computed properties

These virtual properties are computed on access via dot-syntax — no extra function calls needed.

| Property | Type | Description |
|:--- |:--- |:--- |
| `n_clusters` | `Int` | Estimated number of active clusters K̂ |
| `n_obs` | `Int` | Number of individuals n |
| `n_features` | `Int` | Number of features p |
| `cluster_sizes` | `Vector{Int}` | Number of individuals per cluster (hard assignment) |
| `inactivation_rate` | `Vector{Float64}` | Coordinate inactivation rate ρ̄ᵢ ∈ [0, 1] per observation |
| `pi_0` | `Float64` | Marginal prior/variational probability π₀ of the background component |

# Accessing results

```julia
results = mixClust(dataset, 10)

K_hat  = results.n_clusters           # estimated number of clusters
labels = results.labels                # hard assignments
rho    = results.inactivation_rate     # high values: uninformative observation
eig    = compute_eig(results.margins, results.w, results.pip)
```
"""
struct MixClustResult
    w::Matrix{Float64}              # n × K soft cluster assignment probabilities
    labels::Vector{Int}             # MAP cluster assignment: argmax of each row of w
    pip::Matrix{Float64}            # n × p Posterior Inclusion Probabilities γᵢⱼ
    u_star::Vector{Float64}         # K variational Dirichlet parameters u★
    delta_star::Array{Float64}      # Beta parameters for relevance: (p,2) SFRM or (K,p,2) LFRM
    margins::Vector{AbstractMargin} # fitted margin objects, one per feature
    elbo_history::Vector{Float64}   # ELBO history across CAVI iterations
    omega_history::Matrix{Float64}  # T × K trajectory of E_q[ω_k] across iterations
end

# Outer constructors converting margins if needed
function MixClustResult(w::Matrix{Float64}, labels::Vector{Int}, pip::Matrix{Float64},
                        u_star::Vector{Float64}, delta_star::Array{Float64},
                        margins::AbstractVector{<:AbstractMargin}, elbo_history::Vector{Float64},
                        omega_history::Matrix{Float64})
    MixClustResult(w, labels, pip, u_star, delta_star, Vector{AbstractMargin}(margins), elbo_history, omega_history)
end

# Backward compatibility constructor without omega_history
function MixClustResult(w::Matrix{Float64}, labels::Vector{Int}, pip::Matrix{Float64},
                        u_star::Vector{Float64}, delta_star::Array{Float64},
                        margins::AbstractVector{<:AbstractMargin}, elbo_history::Vector{Float64})
    MixClustResult(w, labels, pip, u_star, delta_star, Vector{AbstractMargin}(margins), elbo_history, Matrix{Float64}(undef, 0, size(w, 2)))
end

# Forward declarations for post_hoc methods
function compute_pi_0 end
function extended_responsibilities end
function extended_cluster_assignments end

const _VIRTUAL_PROPS = (:n_clusters, :n_obs, :n_features, :cluster_sizes, :inactivation_rate, :pi_0, :extended_responsibilities, :extended_labels)

function Base.getproperty(r::MixClustResult, s::Symbol)
    s === :n_clusters   && return size(getfield(r, :w), 2)
    s === :n_obs        && return size(getfield(r, :w), 1)
    s === :n_features   && return length(getfield(r, :margins))
    s === :cluster_sizes && begin
        K = size(getfield(r, :w), 2)
        lbl = getfield(r, :labels)
        return [count(==(k), lbl) for k in 1:K]
    end
    s === :inactivation_rate && begin
        pip = getfield(r, :pip)
        n, p = size(pip)
        rates = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            s_pip = 0.0
            for j in 1:p
                s_pip += pip[i, j]
            end
            rates[i] = 1.0 - s_pip / p
        end
        return rates
    end
    s === :pi_0 && return compute_pi_0(r)
    s === :extended_responsibilities && return extended_responsibilities(r)
    s === :extended_labels && return extended_cluster_assignments(r)
    return getfield(r, s)
end

Base.propertynames(::MixClustResult, private::Bool=false) =
    (:w, :labels, :pip, :u_star, :delta_star, :margins, :elbo_history, :omega_history,
     _VIRTUAL_PROPS...)

