using Random
using SpecialFunctions

function sigmoid(x::Real)
    x >= 0 ? (z = exp(-x); 1.0 / (1.0 + z)) : (z = exp(x); z / (1.0 + z))
end

# ── Input validation ──────────────────────────────────────────────────────────

const _VALID_MARGIN_TYPES = (:gaussian, :lognormal, :exponential, :bernoulli, :poisson, :gamma, :multinomial)

function _validate_input(data, K, u0, tol, delta_prior,
                         beta_estimation,
                         n_init, max_iter_init, max_iter,
                         user_feature_types, resolved_types)
    p = length(data)

    isempty(data) && throw(ArgumentError("data must contain at least one feature."))

    K < 2 && throw(ArgumentError(
        "K must be ≥ 2 (got K = $K). At least 2 mixture components are required."))

    n = length(data[1])
    n < 2 && throw(ArgumentError(
        "Features must have at least 2 observations (got n = $n)."))

    for j in 2:p
        nj = length(data[j])
        nj == n || throw(DimensionMismatch(
            "Feature $j has $nj observations but feature 1 has $n. " *
            "All features must have the same number of observations."))
    end

    n <= K && throw(ArgumentError(
        "The number of observations (n = $n) must exceed K = $K. " *
        "Reduce K or collect more data."))

    for j in 1:p
        y_j = data[j]
        eltype(y_j) <: AbstractVector && continue
        any(isnan, y_j) && throw(ArgumentError("Feature $j contains NaN values."))
        any(isinf, y_j) && throw(ArgumentError("Feature $j contains Inf values."))
    end

    u0 > 0 || throw(ArgumentError("u0 must be > 0 (got $u0)."))
    if u0 > 1 / K
        @warn "u0 = $u0 > 1/K = $(round(1 / K, digits = 4)): the symmetric Dirichlet " *
              "prior Dir(u0·1_K) on the mixing weights is not sparse. Automatic order " *
              "selection may not work reliably. Recommended range: u0 ≤ 1/K."
    end

    tol > 0 || throw(ArgumentError("tol must be > 0 (got $tol)."))

    d1, d0 = delta_prior
    d1 > 0 || throw(ArgumentError("delta_prior[1] (δ₁) must be > 0 (got $d1)."))
    d0 > 0 || throw(ArgumentError("delta_prior[2] (δ₀) must be > 0 (got $d0)."))

    beta_estimation in (:two_stage, :iterative) || throw(ArgumentError(
        "beta_estimation must be :two_stage or :iterative (got :$beta_estimation). " *
        "Did you mean :two_stage?"))


    n_init >= 1     || throw(ArgumentError("n_init must be ≥ 1 (got $n_init)."))
    max_iter_init >= 1 || throw(ArgumentError("max_iter_init must be ≥ 1 (got $max_iter_init)."))
    max_iter >= 1   || throw(ArgumentError("max_iter must be ≥ 1 (got $max_iter)."))

    if !isnothing(user_feature_types)
        length(user_feature_types) == p || throw(DimensionMismatch(
            "feature_types has $(length(user_feature_types)) entries but data has $p features."))
        for j in 1:p
            ft = user_feature_types[j]
            isnothing(ft) && continue
            ft in _VALID_MARGIN_TYPES || throw(ArgumentError(
                "feature_types[$j] = :$ft is not a recognised margin type. " *
                "Choose from: :gaussian, :lognormal, :exponential, :bernoulli, :poisson, :gamma, :multinomial."))
        end
    end

    # Per-feature consistency checks using the resolved types
    for j in 1:p
        ft  = resolved_types[j]
        y_j = data[j]

        if ft === :gamma || ft === :lognormal
            any(x -> x <= 0, y_j) && throw(ArgumentError(
                "Feature $j is assigned a :$ft margin (requires strictly positive values) " *
                "but contains values ≤ 0 (minimum = $(minimum(y_j))). " *
                "Fix the data or override with feature_types[$j] = :gaussian."))

        elseif ft === :exponential
            any(x -> x < 0, y_j) && throw(ArgumentError(
                "Feature $j is assigned an :exponential margin (requires non-negative values) " *
                "but contains negative values (minimum = $(minimum(y_j))). " *
                "Fix the data or override with feature_types[$j] = :gaussian."))

        elseif ft === :poisson
            any(x -> x < 0, y_j) && throw(ArgumentError(
                "Feature $j is assigned a Poisson margin (requires non-negative integer values) " *
                "but contains negative values (minimum = $(minimum(y_j))). " *
                "Override with feature_types[$j] = :gaussian if needed."))

        elseif ft === :bernoulli
            any(x -> x < 0 || x > 1, y_j) && throw(ArgumentError(
                "Feature $j is assigned a Bernoulli margin (requires values in [0,1]) " *
                "but contains values outside [0,1] (range: [$(minimum(y_j)), $(maximum(y_j))]). " *
                "For Binomial data, divide counts by the number of trials n first."))

        elseif ft === :multinomial
            C = length(y_j[1])
            for i in eachindex(y_j)
                length(y_j[i]) == C || throw(DimensionMismatch(
                    "Feature $j (Multinomial): observation $i has $(length(y_j[i])) categories " *
                    "but observation 1 has $C. All one-hot vectors must have the same length."))
                any(x -> x < 0, y_j[i]) && throw(ArgumentError(
                    "Feature $j (Multinomial): observation $i contains negative entries."))
            end
        end
    end
end

# ── Feature type inference and resolution ─────────────────────────────────────

function _infer_margin_type(y_j)
    eltype(y_j) <: AbstractVector                && return :multinomial
    all(x -> x == 0.0 || x == 1.0, y_j)         && return :bernoulli
    all(x -> x >= 0 && isinteger(x), y_j)        && return :poisson
    if all(x -> x > 0, y_j)
        if all(x -> x < 1, y_j)
            @warn "Feature contains continuous proportions in (0,1). " *
                  "No exact-conjugate Beta margin is available; the feature will be " *
                  "modelled with a Gamma margin (support mismatched). " *
                  "Recommended: apply the logit transform  y .= log.(y ./ (1 .- y))  " *
                  "before calling mixClust so that a Gaussian margin is used instead."
        end
        return :gamma
    end
    return :gaussian
end

"""
    _resolve_feature_types(data, user_types) -> Vector{Symbol}

Returns a fully-resolved vector of margin type symbols (one per feature).
`user_types` can be `nothing` (all auto-detected) or a vector where each entry
is a `Symbol` in `(:gaussian, :poisson, :gamma, :multinomial)` or `nothing`
(auto-detect that feature only).
"""
function _resolve_feature_types(data::AbstractVector, user_types)
    p = length(data)
    resolved = Vector{Symbol}(undef, p)
    for j in 1:p
        ft = isnothing(user_types) ? nothing : user_types[j]
        resolved[j] = isnothing(ft) ? _infer_margin_type(data[j]) : ft
    end
    return resolved
end

# ── Margin initialisation ─────────────────────────────────────────────────────

# Random soft initialization: dominant cluster gets weight 0.8, rest share 0.2.
function _init_w(n::Int, K::Int)
    w = zeros(n, K)
    for i in 1:n
        k = rand(1:K)
        w[i, k] = 0.8
        for j in 1:K
            j != k && (w[i, j] = 0.2 / (K - 1))
        end
    end
    return w
end

function _init_margins(data::AbstractVector, K::Int, feature_types::Vector{Symbol})
    p = length(data)
    margins = Vector{AbstractMargin}(undef, p)
    for j in 1:p
        y_j = data[j]
        ft  = feature_types[j]
        if ft === :multinomial
            margins[j] = MultinomialMargin(y_j, K)
        elseif ft === :bernoulli
            margins[j] = BernoulliMargin(y_j, K)
        elseif ft === :poisson
            margins[j] = PoissonMargin(y_j, K)
        elseif ft === :gamma
            margins[j] = GammaMargin(y_j, K)
        elseif ft === :lognormal
            margins[j] = LogNormalMargin(y_j, K)
        elseif ft === :exponential
            margins[j] = ExponentialMargin(y_j, K)
        else  # :gaussian
            margins[j] = GaussianMargin(y_j, K)
        end
    end
    return margins
end

# ── Single CAVI run ───────────────────────────────────────────────────────────
# If `init` is nothing → fresh random start.
# If `init` is a MixClustResult → warm-start from that state (used for the full run after screening).
function _cavi_once(data::AbstractVector, K::Int,
                    model_setting::ModelSetting,
                    u0, delta_prior,
                    max_iter::Int, tol,
                    beta_estimation::Symbol,
                    feature_types::Vector{Symbol};
                    init::Union{Nothing, MixClustResult} = nothing)

    p      = length(data)
    n      = length(data[1])
    d1, d0 = delta_prior

    if init === nothing
        margins    = _init_margins(data, K, feature_types)
        w          = _init_w(n, K)
        pip        = fill(0.9, n, p)
        u_star = fill(u0 + n / K, K)
        if model_setting isa SFRM
            delta_star = fill(1.0, p, 2)
            delta_star[:, 1] .= d1;  delta_star[:, 2] .= d0
        else
            delta_star = fill(1.0, K, p, 2)
            delta_star[:, :, 1] .= d1;  delta_star[:, :, 2] .= d0
        end
        for j in 1:p
            # BernoulliMargin uses a diverse grid initialisation; calling
            # update_margin! here with a random w would collapse all clusters
            # to p̂≈p_bg, destroying the diversity needed to escape the null
            # fixed-point.  Other margins are initialised at p_bg for all k
            # and need this call to differentiate.
            margins[j] isa BernoulliMargin && continue
            update_margin!(margins[j], data[j], w, pip[:, j])
        end

        # Pre-align w for BernoulliMargin features: use the diverse initial p_k
        # values to assign each observation to the cluster whose expected log
        # density is highest.  Without this, the random soft assignment gives
        # each cluster p̂_k ≈ p_bg, making all features appear irrelevant.
        bern_indices = findall(j -> margins[j] isa BernoulliMargin, 1:p)
        if !isempty(bern_indices)
            psi_sum_alpha_init = digamma(sum(u_star))
            E_ln_omega_init    = [digamma(u_star[k]) - psi_sum_alpha_init for k in 1:K]
            E_log_f_init       = Matrix{Float64}[expected_log_density(margins[j], data[j]) for j in bern_indices]
            log_w_init         = Matrix{Float64}(undef, n, K)
            for k in 1:K, i in 1:n
                log_w_init[i, k] = E_ln_omega_init[k] +
                    sum(pip[i, bern_indices[jj]] * E_log_f_init[jj][i, k]
                        for jj in eachindex(bern_indices))
            end
            for i in 1:n
                max_log = maximum(log_w_init[i, :])
                row = exp.(log_w_init[i, :] .- max_log)
                w[i, :] .= row ./ sum(row)
            end
        end
    else
        margins    = deepcopy(init.margins)
        w          = copy(init.w)
        pip        = copy(init.pip)
        u_star = copy(init.u_star)
        delta_star = copy(init.delta_star)
    end

    return _cavi_loop!(data, K, model_setting, u0, d1, d0, max_iter, tol, beta_estimation,
                       margins, w, pip, u_star, delta_star)
end

# CAVI iterations. Kept in a separate function (function barrier): the variables
# below are assigned in both branches of the initialisation above, and capturing
# them in closures there made them boxed and type-unstable (≈100× slower).
function _cavi_loop!(data::AbstractVector, K::Int, model_setting::ModelSetting,
                     u0, d1, d0, max_iter::Int, tol, beta_estimation::Symbol,
                     margins::Vector{AbstractMargin}, w::Matrix{Float64},
                     pip::Matrix{Float64}, u_star::Vector{Float64}, delta_star::Array{Float64})
    p = length(data)
    n = length(data[1])
    elbo_history = Float64[]
    omega_history = Vector{Vector{Float64}}()

    for it in 1:max_iter
        w_old   = copy(w)
        pip_old = copy(pip)

        # --- A. Parameter Expectations ---
        psi_sum_alpha = digamma(sum(u_star))
        E_ln_omega    = digamma.(u_star) .- psi_sum_alpha

        E_ln_gamma = model_setting isa SFRM ?
                     Matrix{Float64}(undef, p, 2) :
                     Array{Float64}(undef, K, p, 2)

        if model_setting isa SFRM
            for j in 1:p
                psi_sum = digamma(delta_star[j, 1] + delta_star[j, 2])
                E_ln_gamma[j, 1] = digamma(delta_star[j, 1]) - psi_sum
                E_ln_gamma[j, 2] = digamma(delta_star[j, 2]) - psi_sum
            end
        else
            for k in 1:K, j in 1:p
                psi_sum = digamma(delta_star[k, j, 1] + delta_star[k, j, 2])
                E_ln_gamma[k, j, 1] = digamma(delta_star[k, j, 1]) - psi_sum
                E_ln_gamma[k, j, 2] = digamma(delta_star[k, j, 2]) - psi_sum
            end
        end

        E_log_f = Matrix{Float64}[expected_log_density(margins[j], data[j]) for j in 1:p]
        log_f0  = Vector{Float64}[background_log_density(margins[j], data[j]) for j in 1:p]
        # E_ln_gamma / delta_star are Matrix (SFRM) or 3-d Array (LFRM); typed aliases
        # keep the hot loops below type-stable.
        Eg2 = model_setting isa SFRM ? (E_ln_gamma::Matrix{Float64}) : zeros(0, 0)
        Eg3 = model_setting isa SFRM ? zeros(0, 0, 0) : (E_ln_gamma::Array{Float64,3})
        ds2 = model_setting isa SFRM ? (delta_star::Matrix{Float64}) : zeros(0, 0)
        ds3 = model_setting isa SFRM ? zeros(0, 0, 0) : (delta_star::Array{Float64,3})

        # --- B. Update Latent Variables ---

        log_w = Matrix{Float64}(undef, n, K)
        if model_setting isa SFRM
            for k in 1:K, i in 1:n
                term_data = sum(pip[i, j] * E_log_f[j][i, k] for j in 1:p)
                log_w[i, k] = E_ln_omega[k] + term_data
            end
        else
            for k in 1:K, i in 1:n
                term_data = 0.0;  term_rel = 0.0
                for j in 1:p
                    term_data += pip[i, j] * E_log_f[j][i, k]
                    term_rel  += pip[i, j] * Eg3[k, j, 1] + (1.0 - pip[i, j]) * Eg3[k, j, 2]
                end
                log_w[i, k] = E_ln_omega[k] + term_data + term_rel
            end
        end

        for i in 1:n
            max_log = maximum(log_w[i, :])
            row_sum = 0.0
            for k in 1:K
                w[i, k] = exp(log_w[i, k] - max_log)
                row_sum += w[i, k]
            end
            w[i, :] ./= row_sum
        end

        if model_setting isa SFRM
            for j in 1:p
                B_j = Eg2[j, 1] - Eg2[j, 2]
                for i in 1:n
                    A_ij = sum(w[i, k] * E_log_f[j][i, k] for k in 1:K) - log_f0[j][i]
                    pip[i, j] = sigmoid(A_ij + B_j)
                end
            end
        else
            for j in 1:p, i in 1:n
                A_ij = 0.0;  B_ij = 0.0
                for k in 1:K
                    A_ij += w[i, k] * (E_log_f[j][i, k] - log_f0[j][i])
                    B_ij += w[i, k] * (Eg3[k, j, 1] - Eg3[k, j, 2])
                end
                pip[i, j] = sigmoid(A_ij + B_ij)
            end
        end

        # --- C. Update Parameter Distributions ---

        for k in 1:K
            u_star[k] = u0 + sum(w[:, k])
        end

        if model_setting isa SFRM
            for j in 1:p
                sum_pip = sum(pip[:, j])
                ds2[j, 1] = d1 + sum_pip
                ds2[j, 2] = d0 + (n - sum_pip)
            end
        else
            for k in 1:K, j in 1:p
                sum_w_pip = sum(w[i, k] * pip[i, j] for i in 1:n)
                sum_w     = sum(w[:, k])
                ds3[k, j, 1] = d1 + sum_w_pip
                ds3[k, j, 2] = d0 + (sum_w - sum_w_pip)
            end
        end

        # Refresh E_ln_omega and E_ln_gamma from the just-updated u_star and
        # delta_star before the ELBO computation.  E_log_f / log_f0 are kept from
        # the start of this iteration (old margins): computing the ELBO before
        # update_margin! with these consistent parameters guarantees that the
        # ELBO sequence is monotonically non-decreasing.
        psi_sum_alpha = digamma(sum(u_star))
        E_ln_omega    = digamma.(u_star) .- psi_sum_alpha

        if model_setting isa SFRM
            for j in 1:p
                psi_sum = digamma(ds2[j, 1] + ds2[j, 2])
                Eg2[j, 1] = digamma(ds2[j, 1]) - psi_sum
                Eg2[j, 2] = digamma(ds2[j, 2]) - psi_sum
            end
        else
            for k in 1:K, j in 1:p
                psi_sum = digamma(ds3[k, j, 1] + ds3[k, j, 2])
                Eg3[k, j, 1] = digamma(ds3[k, j, 1]) - psi_sum
                Eg3[k, j, 2] = digamma(ds3[k, j, 2]) - psi_sum
            end
        end

        # --- D. Compute ELBO (before margin update) ---
        E_data = 0.0
        for j in 1:p, i in 1:n
            term_active = sum(w[i, k] * E_log_f[j][i, k] for k in 1:K)
            E_data += pip[i, j] * term_active + (1.0 - pip[i, j]) * log_f0[j][i]
        end

        H_latent = 0.0
        for i in 1:n
            for k in 1:K
                w[i, k] > 1e-15 && (H_latent -= w[i, k] * log(w[i, k]))
            end
            for j in 1:p
                pv = pip[i, j]
                pv > 1e-15       && (H_latent -= pv * log(pv))
                pv < 1.0 - 1e-15 && (H_latent -= (1.0 - pv) * log(1.0 - pv))
            end
        end

        E_latent_prior = 0.0
        for i in 1:n
            for k in 1:K
                E_latent_prior += w[i, k] * E_ln_omega[k]
            end
            for j in 1:p
                if model_setting isa SFRM
                    E_latent_prior += pip[i, j] * Eg2[j, 1] + (1.0 - pip[i, j]) * Eg2[j, 2]
                else
                    E_latent_prior += sum(w[i, k] * (pip[i, j] * Eg3[k, j, 1] +
                                         (1.0 - pip[i, j]) * Eg3[k, j, 2]) for k in 1:K)
                end
            end
        end

        # Prior Dir(ω; u0·1_K): concentration u0 per component
        KL_omega = loggamma(sum(u_star)) - loggamma(K * u0)
        for k in 1:K
            KL_omega += loggamma(u0) - loggamma(u_star[k]) +
                        (u_star[k] - u0) * E_ln_omega[k]
        end

        ln_beta_d = loggamma(d1) + loggamma(d0) - loggamma(d1 + d0)
        KL_gamma  = 0.0
        if model_setting isa SFRM
            for j in 1:p
                lbs = loggamma(ds2[j,1]) + loggamma(ds2[j,2]) - loggamma(ds2[j,1]+ds2[j,2])
                KL_gamma += ln_beta_d - lbs + (ds2[j,1]-d1)*Eg2[j,1] + (ds2[j,2]-d0)*Eg2[j,2]
            end
        else
            for k in 1:K, j in 1:p
                lbs = loggamma(ds3[k,j,1]) + loggamma(ds3[k,j,2]) - loggamma(ds3[k,j,1]+ds3[k,j,2])
                KL_gamma += ln_beta_d - lbs + (ds3[k,j,1]-d1)*Eg3[k,j,1] + (ds3[k,j,2]-d0)*Eg3[k,j,2]
            end
        end

        KL_theta = sum(kl_from_prior(margins[j]) for j in 1:p)

        push!(elbo_history, E_data + H_latent + E_latent_prior - KL_omega - KL_gamma - KL_theta)
        push!(omega_history, u_star ./ sum(u_star))

        # --- E. Update Margin Parameters (after ELBO) ---
        # Kept after ELBO so the logged sequence uses consistent
        # (w_t, pip_t, alpha_t, delta_t, margins_{t-1}), which is monotone.
        for j in 1:p
            update_margin!(margins[j], data[j], w, pip[:, j])
            beta_estimation === :iterative && update_background!(margins[j], data[j], pip[:, j])
        end

        if max(maximum(abs.(w .- w_old)), maximum(abs.(pip .- pip_old))) < tol
            break
        end
    end

    labels = [argmax(w[i, :]) for i in 1:n]
    omega_mat = isempty(omega_history) ? Matrix{Float64}(undef, 0, K) : reduce(vcat, [v' for v in omega_history])
    return MixClustResult(w, labels, pip, u_star, delta_star, margins, elbo_history, omega_mat)
end

"""
    mixClust(data, K; model_setting, u0, delta_prior, max_iter, tol,
             compact, beta_estimation,
             n_init, max_iter_init, feature_types) -> MixClustResult

Fit a finite Bayesian mixture model via CAVI on heterogeneous data.

# Arguments

- `data`: `Vector` of length `p`; `data[j]` holds all observations for feature `j`.
  Use `Vector{Float64}` for numeric features and `Vector{Vector{Int}}` (one-hot) for
  Multinomial features. The margin type is inferred automatically unless overridden
  via `feature_types`.
- `K`: Maximum number of components (overfitted mixture; the active order K̂ ≤ K is
  selected automatically via the sparse Dirichlet prior).

# Keyword arguments

- `model_setting`: [`SFRM()`](@ref) (shared relevance, one probability per feature)
  or [`LFRM()`](@ref) (local relevance, one probability per feature–cluster pair).
  Default: `SFRM()`.
- `u0`: Hyperparameter `u⁽⁰⁾` of the symmetric Dirichlet prior on the mixing weights,
  whose concentration vector is `u⁽⁰⁾·1_K = (u⁽⁰⁾, …, u⁽⁰⁾)` (`K` = maximum number of
  components). Values `u⁽⁰⁾ ≤ 1/K` make the prior sparse and drive automatic order
  selection. Default: `nothing`, i.e. `u⁽⁰⁾ = 1/K`.
- `delta_prior`: `(δ₁, δ₀)` hyperparameters of the Beta(`δ₁`, `δ₀`) prior on
  each relevance indicator. `(1.0, 1.0)` is a uniform prior. Default: `(1.0, 1.0)`.
- `max_iter`: Maximum CAVI iterations for the final run. Default: `500`.
- `tol`: Convergence threshold on the maximum absolute change in `w` and `pip`
  between consecutive iterations. Default: `1e-4`.
- `compact`: If `true`, drop empty mixture components (clusters with zero observations
  under MAP assignment) upon convergence. Default: `true`.
- `beta_estimation`: `:two_stage` (background parameters estimated once on the full
  dataset before CAVI, then held fixed) or `:iterative` (re-estimated at every CAVI
  step as a weighted M-step). Default: `:two_stage`.
- `n_init`: Number of random restarts in the screening phase. Default: `10`.
- `max_iter_init`: Number of CAVI iterations per screening run. Default: `10`.
- `feature_types`: Optional vector of length `p` to override the automatic margin
  detection for specific features. Each entry can be one of
  `:gaussian`, `:poisson`, `:gamma`, `:multinomial`, or `nothing` (auto-detect).
  Example: `feature_types = [:gaussian, nothing, :poisson]` forces feature 1 to
  Gaussian and feature 3 to Poisson, while feature 2 is auto-detected.
  Default: `nothing` (all features auto-detected).

# Auto-detection rules

| Data shape | Detected margin |
|:--- |:--- |
| `Vector{Vector{Int}}` (one-hot) | Multinomial |
| `Vector{Float64}`, all values non-negative integers | Poisson |
| `Vector{Float64}`, all values strictly positive | Gamma |
| `Vector{Float64}`, otherwise | Gaussian |

Use `feature_types` to override when auto-detection is ambiguous (e.g., a positive
continuous variable you wish to model as Gaussian instead of Gamma).

# Two-phase strategy

`n_init` short runs of `max_iter_init` iterations each select the most promising
initialization (highest ELBO). One full run of `max_iter` iterations warm-started
from that state produces the final result. Total cost:
`n_init × max_iter_init + max_iter` iterations.

# Returns

A [`MixClustResult`](@ref) containing soft assignments `w`, hard labels, PIPs,
fitted margins, ELBO history, and variational parameters.
"""
function mixClust(data::AbstractVector, K::Int;
                  model_setting::ModelSetting = SFRM(),
                  u0                          = nothing,
                  delta_prior                 = (1.0, 1.0),
                  max_iter::Int               = 500,
                  tol                         = 1e-4,
                  compact::Bool               = true,
                  beta_estimation::Symbol     = :two_stage,
                  n_init::Int                 = 10,
                  max_iter_init::Int          = 10,
                  feature_types               = nothing)
    u0 = isnothing(u0) ? 1.0 / K : u0   # default: u⁽⁰⁾ = 1/K

    # Resolve margin types first (needed by validation)
    resolved_types = _resolve_feature_types(data, feature_types)

    # Validate all inputs; throws ArgumentError / DimensionMismatch on bad input
    _validate_input(data, K, u0, tol, delta_prior,
                    beta_estimation,
                    n_init, max_iter_init, max_iter,
                    feature_types, resolved_types)

    # Log resolved margin types so users can verify the auto-detection
    if length(data) <= 15
        _names = Dict(:gaussian => "Gaussian", :bernoulli => "Bernoulli",
                      :poisson => "Poisson", :gamma => "Gamma", :multinomial => "Multinomial")
        @info "mixClust: margin types for $(length(data)) features:" *
              join(["\n  [$(j)] $(get(_names, resolved_types[j], string(resolved_types[j])))"
                    for j in eachindex(resolved_types)])
    else
        @info "mixClust: $(length(data)) features resolved (" *
              "$(count(==(:gaussian), resolved_types)) Gaussian, " *
              "$(count(==(:poisson), resolved_types)) Poisson, " *
              "$(count(==(:multinomial), resolved_types)) Multinomial, " *
              "$(count(==(:gamma), resolved_types)) Gamma, " *
              "$(count(==(:bernoulli), resolved_types)) Bernoulli)"
    end

    # Phase 1: n_init short screening runs
    best_screen = _cavi_once(data, K, model_setting, u0, delta_prior,
                             max_iter_init, tol, beta_estimation, resolved_types)
    for _ in 2:n_init
        candidate = _cavi_once(data, K, model_setting, u0, delta_prior,
                               max_iter_init, tol, beta_estimation, resolved_types)
        if last(candidate.elbo_history) > last(best_screen.elbo_history)
            best_screen = candidate
        end
    end

    # Phase 2: full run warm-started from the best screening state
    best = _cavi_once(data, K, model_setting, u0, delta_prior,
                      max_iter, tol, beta_estimation, resolved_types; init = best_screen)

    return compact ? compact_clusters(best, data) : best
end

