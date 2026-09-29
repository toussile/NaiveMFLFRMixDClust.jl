"""
    compute_local_ehd(margins::Vector{<:AbstractMargin}, w::AbstractMatrix, pip::AbstractVecOrMat) -> Matrix{Float64}

Computes the cluster-specific Expected Hellinger Distance (EHD_{k,j}) for each cluster k and feature j,
intrinsically bounded in [0, 1]:
    \\mathbf{EHD}_{k,j} = \\bar{\\gamma}_{k,j} * H^2(f_{k,j}, f_{0,j}) \\in [0, 1]
where H^2(f_{k,j}, f_{0,j}) \\in [0, 1] is the squared Hellinger distance between cluster distribution
f_{k,j} and background distribution f_{0,j}, and \\bar{\\gamma}_{k,j} is the estimated activation
probability of feature j in cluster k.

Returns a K x p matrix representing the local EHD of each feature across clusters.
"""
function compute_local_ehd(margins::Vector{<:AbstractMargin}, w::AbstractMatrix, pip::AbstractVecOrMat)
    n, K = size(w)
    p = length(margins)
    
    ehd_local = zeros(Float64, K, p)
    for k in 1:K
        sum_w_k = sum(w[:, k])
        if sum_w_k <= 1e-12
            continue
        end
        
        for j in 1:p
            # 1. Feature activation probability in cluster k: \bar{\gamma}_{k,j}
            gamma_kj = if ndims(pip) == 1
                Float64(pip[j])
            else
                sum(w[:, k] .* pip[:, j]) / sum_w_k
            end
            
            # 2. Squared Hellinger distance H^2(f_{k,j}, f_{0,j}) \in [0, 1]
            h2 = hellinger_divergence(margins[j]) # Vector of length K
            h2_kj = clamp(h2[k], 0.0, 1.0)
            
            # 3. Local Expected Hellinger Distance intrinsically bounded in [0, 1]
            ehd_local[k, j] = clamp(gamma_kj * h2_kj, 0.0, 1.0)
        end
    end
    
    return ehd_local
end

"""
    compute_ehd(margins::Vector{<:AbstractMargin}, w::AbstractMatrix, pip::AbstractVecOrMat) -> Vector{Float64}

Computes the population-level Expected Hellinger Distance (EHD_j) for each feature j,
intrinsically bounded in [0, 1]:
    \\mathbf{EHD}_j = \\sum_{k=1}^K \\bar{\\omega}_k * \\mathbf{EHD}_{k,j} \\in [0, 1]
where \\bar{\\omega}_k = \\frac{1}{n} \\sum_{i=1}^n w_{i,k} are estimated cluster mixing weights.

Returns a vector of length p representing the global EHD of each feature.
"""
function compute_ehd(margins::Vector{<:AbstractMargin}, w::AbstractMatrix, pip::AbstractVecOrMat)
    n, K = size(w)
    p = length(margins)
    
    ehd_local = compute_local_ehd(margins, w, pip) # K x p
    
    # Compute estimated cluster proportions w_bar
    w_bar = [sum(w[:, k]) / n for k in 1:K]
    
    ehd = zeros(Float64, p)
    for j in 1:p
        val = 0.0
        for k in 1:K
            val += w_bar[k] * ehd_local[k, j]
        end
        ehd[j] = clamp(val, 0.0, 1.0)
    end
    
    return ehd
end

# Backward compatibility aliases
const compute_local_eig = compute_local_ehd
const compute_eig = compute_ehd

compute_ehd(results::MixClustResult) = compute_ehd(results.margins, results.w, results.pip)
compute_local_ehd(results::MixClustResult) = compute_local_ehd(results.margins, results.w, results.pip)



"""
    filter_features(eig::Vector{Float64}, threshold::Real) -> Vector{Int}

Returns the indices of the features whose Expected Information Gain exceeds the given threshold.
"""
function filter_features(eig::Vector{Float64}, threshold::Real)
    return findall(x -> x >= threshold, eig)
end

"""
    predict_proba(results::MixClustResult, new_data::AbstractVector) -> Matrix{Float64}

Computes the posterior subtype assignment probabilities q(z^*_i = k | y^*_i) for new observations.
Returns an n_new x K matrix where rows sum to 1.
"""
function predict_proba(results::MixClustResult, new_data::AbstractVector)
    p = length(results.margins)
    n_new = length(new_data[1])
    K = length(results.u_star)
    
    # 1. Compute expected mixing weights omega_bar
    sum_alpha = sum(results.u_star)
    omega_bar = results.u_star ./ sum_alpha
    
    # 2. Compute expected feature inclusion probabilities gamma_bar (K x p)
    gamma_bar = Matrix{Float64}(undef, K, p)
    if ndims(results.delta_star) == 2
        # Model 1
        for j in 1:p
            g = results.delta_star[j, 1] / (results.delta_star[j, 1] + results.delta_star[j, 2])
            gamma_bar[:, j] .= g
        end
    else
        # Model 2
        for k in 1:K
            for j in 1:p
                gamma_bar[k, j] = results.delta_star[k, j, 1] / (results.delta_star[k, j, 1] + results.delta_star[k, j, 2])
            end
        end
    end
    
    # 3. Precompute predictive densities for each feature j
    P = [predictive_density(results.margins[j], new_data[j]) for j in 1:p]
    B = [exp.(background_log_density(results.margins[j], new_data[j])) for j in 1:p]
    
    # 4. Compute log-likelihood contribution for each patient i and cluster k
    log_q = Matrix{Float64}(undef, n_new, K)
    for i in 1:n_new
        for k in 1:K
            val = log(max(omega_bar[k], 1e-15))
            for j in 1:p
                g = gamma_bar[k, j]
                dens = g * P[j][i, k] + (1.0 - g) * B[j][i]
                val += log(max(dens, 1e-300))
            end
            log_q[i, k] = val
        end
    end
    
    # 5. Exponentiate and normalize (softmax per individual)
    w_pred = Matrix{Float64}(undef, n_new, K)
    for i in 1:n_new
        max_log = maximum(log_q[i, :])
        row_exp = exp.(log_q[i, :] .- max_log)
        sum_exp = sum(row_exp)
        w_pred[i, :] = sum_exp > 0 ? row_exp ./ sum_exp : fill(1.0 / K, K)
    end
    
    return w_pred
end

"""
    predictive_log_likelihood(results::MixClustResult, new_data::AbstractVector) -> Float64

Computes the total out-of-sample log-predictive density log p(y^* | y) on a test dataset.
"""
function predictive_log_likelihood(results::MixClustResult, new_data::AbstractVector)
    p = length(results.margins)
    n_new = length(new_data[1])
    K = length(results.u_star)
    
    sum_alpha = sum(results.u_star)
    omega_bar = results.u_star ./ sum_alpha
    
    gamma_bar = Matrix{Float64}(undef, K, p)
    if ndims(results.delta_star) == 2
        for j in 1:p
            g = results.delta_star[j, 1] / (results.delta_star[j, 1] + results.delta_star[j, 2])
            gamma_bar[:, j] .= g
        end
    else
        for k in 1:K
            for j in 1:p
                gamma_bar[k, j] = results.delta_star[k, j, 1] / (results.delta_star[k, j, 1] + results.delta_star[k, j, 2])
            end
        end
    end
    
    P = [predictive_density(results.margins[j], new_data[j]) for j in 1:p]
    B = [exp.(background_log_density(results.margins[j], new_data[j])) for j in 1:p]
    
    total_log_lik = 0.0
    for i in 1:n_new
        log_terms = Vector{Float64}(undef, K)
        for k in 1:K
            val = log(max(omega_bar[k], 1e-15))
            for j in 1:p
                g = gamma_bar[k, j]
                dens = g * P[j][i, k] + (1.0 - g) * B[j][i]
                val += log(max(dens, 1e-300))
            end
            log_terms[k] = val
        end
        max_log = maximum(log_terms)
        total_log_lik += max_log + log(sum(exp.(log_terms .- max_log)))
    end
    
    return total_log_lik
end


"""
    n_active_clusters(w; threshold=0.02) -> Int

Return the number of clusters whose mean responsibility exceeds `threshold`.
This is the single canonical definition used throughout the package and scripts.
"""
function n_active_clusters(w::AbstractMatrix; threshold::Real=0.02)
    n = size(w, 1)
    return sum(k -> sum(w[:, k]) / n >= threshold, 1:size(w, 2))
end

n_active_clusters(results::MixClustResult; threshold::Real=0.02) = n_active_clusters(results.w; threshold=threshold)


"""
    compact_clusters(results::MixClustResult, data::Union{AbstractVector, Nothing}=nothing) -> MixClustResult

Drop unassigned mixture components (clusters with zero observations under MAP assignment,
i.e. `count(==(k), results.labels) == 0`).

This reflects the intrinsic sparse Dirichlet order selection (u⁽⁰⁾ < 1) where empty
components are naturally driven to zero weight. Unlike heuristic pruning, this operation
drops only strictly unpopulated components and does not perform heuristic merges.
"""
function compact_clusters(results::MixClustResult, data::Union{AbstractVector, Nothing}=nothing)
    w = copy(results.w)
    pip = copy(results.pip)
    u_star = copy(results.u_star)
    delta_star = copy(results.delta_star)
    n, K = size(w)
    p = length(results.margins)

    # Active clusters: those with at least one observation assigned under MAP
    lbl = results.labels
    active_clusters = sort(unique(lbl))

    # If all components are populated, no compaction is needed
    if length(active_clusters) == K || isempty(active_clusters)
        return results
    end

    K_new = length(active_clusters)

    # 1. Compact soft assignments w and renormalize
    w = w[:, active_clusters]
    for i in 1:n
        s = sum(w[i, :])
        if s > 0
            w[i, :] ./= s
        else
            w[i, :] .= 1.0 / K_new
        end
    end

    # 2. Compact u_star
    u_star = u_star[active_clusters]

    # 3. Compact delta_star if 3D (LFRM)
    if ndims(delta_star) == 3
        delta_star = delta_star[active_clusters, :, :]
    end

    # 4. Compact or refit margins
    new_margins = Vector{AbstractMargin}(undef, p)
    if data !== nothing
        for j in 1:p
            y_j = data[j]
            m_old = results.margins[j]
            if m_old isa MultinomialMargin
                new_margins[j] = MultinomialMargin(y_j, K_new; varphi=m_old.varphi)
            elseif m_old isa BernoulliMargin
                new_margins[j] = BernoulliMargin(y_j, K_new; alpha_0=m_old.alpha_0, beta_0=m_old.beta_0)
            elseif m_old isa PoissonMargin
                new_margins[j] = PoissonMargin(y_j, K_new; a_0=m_old.a_0, b_0=m_old.b_0)
            elseif m_old isa GammaMargin
                new_margins[j] = GammaMargin(y_j, K_new; alpha_0=m_old.alpha_0, beta_0=m_old.beta_0)
            elseif m_old isa LogNormalMargin
                new_margins[j] = LogNormalMargin(y_j, K_new; mu_0=m_old.mu_0, kappa_0=m_old.kappa_0, a_0=m_old.a_0, b_0=m_old.b_0)
            elseif m_old isa ExponentialMargin
                new_margins[j] = ExponentialMargin(y_j, K_new; alpha_0=m_old.alpha_0, beta_0=m_old.beta_0)
            else
                new_margins[j] = GaussianMargin(y_j, K_new; mu_0=m_old.mu_0, kappa_0=m_old.kappa_0, a_0=m_old.a_0, b_0=m_old.b_0)
            end
            update_margin!(new_margins[j], y_j, w, pip[:, j])
        end
    else
        for j in 1:p
            m_old = results.margins[j]
            if m_old isa GaussianMargin
                new_margins[j] = GaussianMargin(
                    m_old.mu_0, m_old.kappa_0, m_old.a_0, m_old.b_0,
                    m_old.mu_star[active_clusters], m_old.kappa_star[active_clusters],
                    m_old.a_star[active_clusters], m_old.b_star[active_clusters],
                    m_old.mu_bg, m_old.tau_bg
                )
            elseif m_old isa LogNormalMargin
                new_margins[j] = LogNormalMargin(
                    m_old.mu_0, m_old.kappa_0, m_old.a_0, m_old.b_0,
                    m_old.mu_star[active_clusters], m_old.kappa_star[active_clusters],
                    m_old.a_star[active_clusters], m_old.b_star[active_clusters],
                    m_old.mu_bg, m_old.tau_bg
                )
            elseif m_old isa ExponentialMargin
                new_margins[j] = ExponentialMargin(
                    m_old.alpha_0, m_old.beta_0,
                    m_old.alpha_star[active_clusters], m_old.beta_star[active_clusters],
                    m_old.lambda_bg
                )
            elseif m_old isa MultinomialMargin
                new_margins[j] = MultinomialMargin(
                    m_old.varphi, m_old.varphi_star[active_clusters, :], m_old.phi_bg
                )
            elseif m_old isa PoissonMargin
                new_margins[j] = PoissonMargin(
                    m_old.a_0, m_old.b_0,
                    m_old.a_star[active_clusters], m_old.b_star[active_clusters],
                    m_old.lambda_bg
                )
            elseif m_old isa BernoulliMargin
                new_margins[j] = BernoulliMargin(
                    m_old.alpha_0, m_old.beta_0,
                    m_old.alpha_star[active_clusters], m_old.beta_star[active_clusters],
                    m_old.p_bg
                )
            elseif m_old isa GammaMargin
                new_margins[j] = GammaMargin(
                    m_old.alpha_0, m_old.beta_0,
                    m_old.alpha_star[active_clusters], m_old.beta_star[active_clusters],
                    m_old.a_bg, m_old.b_bg, m_old.a_cl
                )
            end
        end
    end

    labels = [argmax(w[i, :]) for i in 1:n]
    omega_hist = isempty(results.omega_history) ? Matrix{Float64}(undef, 0, K_new) : results.omega_history[:, active_clusters]
    return MixClustResult(w, labels, pip, u_star, delta_star, new_margins, results.elbo_history, omega_hist)
end


"""
    coordinate_inactivation_rate(pip::AbstractMatrix) -> Vector{Float64}
    coordinate_inactivation_rate(results::MixClustResult) -> Vector{Float64}

Computes the coordinate inactivation rate \$\\bar{\\rho}_i \\in [0, 1]\$ for each observation \$i = 1, \\dots, n\$:
    \$\\bar{\\rho}_i = 1 - \\frac{1}{p} \\sum_{j=1}^p \\varphi_{i,j}^*\$
where \$\\varphi_{i,j}^* = q^*(s_{i,j}=1)\$ is the posterior inclusion probability of feature \$j\$ for individual \$i\$.
Higher values indicate greater deviation from all cluster structures towards background noise.
"""
function coordinate_inactivation_rate(pip::AbstractMatrix)
    n, p = size(pip)
    rho = Vector{Float64}(undef, n)
    for i in 1:n
        rho[i] = 1.0 - sum(view(pip, i, :)) / p
    end
    return rho
end

coordinate_inactivation_rate(results::MixClustResult) = coordinate_inactivation_rate(results.pip)

"""
    detect_outliers(results::MixClustResult; threshold::Real=0.55, mode::Symbol=:inactivation_rate) -> BitVector
    detect_outliers(pip::AbstractMatrix; threshold::Real=0.55, mode::Symbol=:inactivation_rate) -> BitVector

Identifies observations as background outliers/anomalies based on coordinate inactivation.

# Modes
- `:inactivation_rate` (default): Flags observations with \$\\bar{\\rho}_i \\ge \\text{threshold}\$.
- `:map_strict`: Flags observations whose joint MAP feature selection profile is the null vector \$\\bm{s}_i = \\bm{0}_p\$ (i.e. \$\\varphi_{i,j}^* < 0.5\$ for all \$j = 1, \\dots, p\$).
- `:map_relaxed`: Flags observations where the fraction of inactive coordinates (\$\\varphi_{i,j}^* < 0.5\$) is \$\\ge \\text{threshold}\$.
"""
function detect_outliers(pip::AbstractMatrix; threshold::Real=0.55, mode::Symbol=:inactivation_rate)
    n, p = size(pip)
    if mode === :inactivation_rate
        rho = coordinate_inactivation_rate(pip)
        return rho .>= threshold
    elseif mode === :map_strict
        out = falses(n)
        for i in 1:n
            out[i] = all(pip[i, :] .< 0.5)
        end
        return out
    elseif mode === :map_relaxed
        out = falses(n)
        for i in 1:n
            out[i] = count(x -> x < 0.5, view(pip, i, :)) / p >= threshold
        end
        return out
    else
        throw(ArgumentError("Unknown outlier detection mode: \$mode. Choose from :inactivation_rate, :map_strict, :map_relaxed"))
    end
end

detect_outliers(results::MixClustResult; threshold::Real=0.55, mode::Symbol=:inactivation_rate) =
    detect_outliers(results.pip; threshold=threshold, mode=mode)

"""
    robust_cluster_assignments(results::MixClustResult; threshold::Real=0.55, mode::Symbol=:inactivation_rate) -> Vector{Int}

Returns robust cluster assignments \$\\widehat{z}_i^{\\mathrm{robust}} \\in \\{0, 1, \\dots, K\\}\$, where:
- `0` indicates an anomalous observation rejected as background noise,
- `k \\in \\{1, \\dots, K\\}` indicates assignment to cluster `k` via standard MAP.
"""
function robust_cluster_assignments(results::MixClustResult; threshold::Real=0.55, mode::Symbol=:inactivation_rate)
    is_outlier = detect_outliers(results; threshold=threshold, mode=mode)
    z_robust = copy(results.labels)
    z_robust[is_outlier] .= 0
    return z_robust
end

"""
    compute_pi_0(results::MixClustResult) -> Float64

Compute the marginal prior / variational probability π₀ of the null background
component (where all coordinates are inactive: sᵢ = 0), given by:
    π₀ = ∑_{k=1}^K ω̄ₖ ∏_{j=1}^p (1 - γ̄ₖⱼ)
where ω̄ₖ = u★ₖ / ∑ₗ u★ₗ and γ̄ₖⱼ is the variational posterior mean of γₖⱼ.
"""
function compute_pi_0(results::MixClustResult)
    u_star = results.u_star
    sum_u = sum(u_star)
    omega_bar = sum_u > 0 ? u_star ./ sum_u : fill(1.0 / length(u_star), length(u_star))
    p = length(results.margins)
    K = length(u_star)

    if ndims(results.delta_star) == 2
        log_prod = 0.0
        for j in 1:p
            d1 = results.delta_star[j, 1]
            d0 = results.delta_star[j, 2]
            p_inact = d0 / (d1 + d0)
            log_prod += log(max(p_inact, 1e-12))
        end
        return exp(log_prod)
    else
        pi_0 = 0.0
        for k in 1:K
            log_prod_k = 0.0
            for j in 1:p
                d1 = results.delta_star[k, j, 1]
                d0 = results.delta_star[k, j, 2]
                p_inact = d0 / (d1 + d0)
                log_prod_k += log(max(p_inact, 1e-12))
            end
            pi_0 += omega_bar[k] * exp(log_prod_k)
        end
        return pi_0
    end
end

"""
    outlier_indices(results::MixClustResult; threshold::Real=0.55, mode::Symbol=:inactivation_rate) -> Vector{Int}

Return indices of observations flagged as background outliers.
"""
function outlier_indices(results::MixClustResult; threshold::Real=0.55, mode::Symbol=:inactivation_rate)
    return findall(detect_outliers(results; threshold=threshold, mode=mode))
end

"""
    cluster_indices(results::MixClustResult, k::Int; threshold::Real=0.55, robust::Bool=false) -> Vector{Int}

Return indices of observations assigned to cluster `k`.
If `robust=true`, background outliers (under `threshold`) are excluded.
"""
function cluster_indices(results::MixClustResult, k::Int; threshold::Real=0.55, robust::Bool=false)
    assignments = robust ? robust_cluster_assignments(results; threshold=threshold) : results.labels
    return findall(==(k), assignments)
end


