mutable struct BernoulliMargin <: AbstractMargin
    # Beta prior hyperparameters
    alpha_0::Float64
    beta_0::Float64

    # Variational parameters (length K): q(p_kj) = Beta(alpha_star[k], beta_star[k])
    alpha_star::Vector{Float64}
    beta_star::Vector{Float64}

    # Background parameter (fixed, estimated from marginal)
    p_bg::Float64
end

function BernoulliMargin(y_j::AbstractVector, K::Int;
                         alpha_0=1.0, beta_0=1.0, robust::Bool=true)
    p_bg = robust ? (1.0 + sum(y_j)) / (length(y_j) + 2.0) : clamp(mean(y_j), 1e-8, 1.0 - 1e-8)
    n     = length(y_j)
    # Spread initial p_k values across [0.05, 0.95] with small jitter, randomly
    # shuffled so cluster ordering is not predetermined.  Pseudo-counts are scaled
    # to n/K so the diverse initialisation survives the first CAVI iteration.
    offsets    = K == 1 ? [0.0] : collect(range(-0.4, 0.4; length=K)) .+ randn(K) .* 0.05
    p_init     = clamp.(p_bg .+ offsets[randperm(K)], 0.05, 0.95)
    n_eff      = max(n / K, 5.0)
    alpha_star = alpha_0 .+ p_init .* n_eff
    beta_star  = beta_0  .+ (1.0 .- p_init) .* n_eff
    return BernoulliMargin(alpha_0, beta_0, alpha_star, beta_star, p_bg)
end

function update_margin!(margin::BernoulliMargin, y_j::AbstractVector,
                        w::AbstractMatrix, gamma_j::AbstractVector)
    n, K = size(w)
    for k in 1:K
        swg  = 0.0
        swgy = 0.0
        for i in 1:n
            wg    = w[i, k] * gamma_j[i]
            swg  += wg
            swgy += wg * y_j[i]
        end
        margin.alpha_star[k] = margin.alpha_0 + swgy
        margin.beta_star[k]  = margin.beta_0  + (swg - swgy)
    end
end

function expected_log_density(margin::BernoulliMargin, y_j::AbstractVector)
    n = length(y_j)
    K = length(margin.alpha_star)
    eld = Matrix{Float64}(undef, n, K)
    for k in 1:K
        psi_sum   = digamma(margin.alpha_star[k] + margin.beta_star[k])
        E_log_p   = digamma(margin.alpha_star[k]) - psi_sum
        E_log_1mp = digamma(margin.beta_star[k])  - psi_sum
        for i in 1:n
            eld[i, k] = y_j[i] * E_log_p + (1.0 - y_j[i]) * E_log_1mp
        end
    end
    return eld
end

function background_log_density(margin::BernoulliMargin, y_j::AbstractVector)
    lp   = log(margin.p_bg)
    l1mp = log(1.0 - margin.p_bg)
    return [y_j[i] * lp + (1.0 - y_j[i]) * l1mp for i in eachindex(y_j)]
end

# Plug-in KL(Bern(p̄_k) ‖ Bern(p_bg)) using posterior mean p̄_k = α*/(α*+β*)
function expected_kl_divergence(margin::BernoulliMargin)
    K    = length(margin.alpha_star)
    kl   = Vector{Float64}(undef, K)
    lp_bg   = log(margin.p_bg)
    l1mp_bg = log(1.0 - margin.p_bg)
    for k in 1:K
        p̄  = clamp(margin.alpha_star[k] / (margin.alpha_star[k] + margin.beta_star[k]),
                   1e-15, 1.0 - 1e-15)
        kl[k] = max(p̄ * (log(p̄) - lp_bg) + (1.0 - p̄) * (log(1.0 - p̄) - l1mp_bg), 0.0)
    end
    return kl
end

function predictive_density(margin::BernoulliMargin, y_new::AbstractVector)
    K     = length(margin.alpha_star)
    n_new = length(y_new)
    pred  = Matrix{Float64}(undef, n_new, K)
    for k in 1:K
        p̄ = margin.alpha_star[k] / (margin.alpha_star[k] + margin.beta_star[k])
        for i in 1:n_new
            pred[i, k] = p̄^y_new[i] * (1.0 - p̄)^(1.0 - y_new[i])
        end
    end
    return pred
end

function update_background!(margin::BernoulliMargin, y_j::AbstractVector,
                            gamma_j::AbstractVector)
    sum_w  = 0.0
    sum_wy = 0.0
    for i in eachindex(y_j)
        w_i    = 1.0 - gamma_j[i]
        sum_w  += w_i
        sum_wy += w_i * y_j[i]
    end
    sum_w > 1e-5 && (margin.p_bg = clamp(sum_wy / sum_w, 1e-8, 1.0 - 1e-8))
end

# KL(Beta(α*,β*) ‖ Beta(α₀,β₀)) summed over K clusters
function kl_from_prior(margin::BernoulliMargin)
    kl = 0.0
    a0, b0     = margin.alpha_0, margin.beta_0
    ln_B_prior = loggamma(a0) + loggamma(b0) - loggamma(a0 + b0)
    for k in eachindex(margin.alpha_star)
        a = margin.alpha_star[k]
        b = margin.beta_star[k]
        kl += ln_B_prior - (loggamma(a) + loggamma(b) - loggamma(a + b)) +
              (a - a0) * digamma(a) + (b - b0) * digamma(b) -
              (a + b - a0 - b0) * digamma(a + b)
    end
    return kl
end

function hellinger_divergence(margin::BernoulliMargin)
    K = length(margin.alpha_star)
    h2 = zeros(Float64, K)
    p_0 = clamp(margin.p_bg, 1e-10, 1.0 - 1e-10)

    for k in 1:K
        p_k = clamp(margin.alpha_star[k] / (margin.alpha_star[k] + margin.beta_star[k]), 1e-10, 1.0 - 1e-10)
        bc = sqrt(p_k * p_0) + sqrt((1.0 - p_k) * (1.0 - p_0))
        h2[k] = clamp(1.0 - bc, 0.0, 1.0)
    end
    return h2
end

