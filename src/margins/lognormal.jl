mutable struct LogNormalMargin <: AbstractMargin
    # Prior hyperparameters (on log scale)
    mu_0::Float64
    kappa_0::Float64
    a_0::Float64
    b_0::Float64
    
    # Variational parameters (length K)
    mu_star::Vector{Float64}
    kappa_star::Vector{Float64}
    a_star::Vector{Float64}
    b_star::Vector{Float64}
    
    # Background parameters (fixed, on log scale)
    mu_bg::Float64
    tau_bg::Float64 # precision 1/σ^2
end

function LogNormalMargin(y_j::AbstractVector, K::Int; 
                         mu_0=nothing, kappa_0=0.05, a_0=3.0, b_0=nothing,
                         median_based::Bool=true,
                         robust::Union{Nothing, Bool}=nothing)
    median_based = _background_flag(median_based, robust, :median_based, :LogNormalMargin)
    # Check positivity
    any(x -> x <= 0, y_j) && throw(ArgumentError("LogNormalMargin requires strictly positive observations."))
    
    z_j = log.(Float64.(y_j))
    
    if median_based
        med = Float64(median(z_j))
        mad_val = 1.4826 * Float64(median(abs.(z_j .- med)))
        sigma_bg = mad_val > 1e-6 ? mad_val : (std(z_j) > 1e-6 ? Float64(std(z_j)) : 1.0)
        mu_bg = med
        var_bg = sigma_bg^2
        tau_bg = 1.0 / var_bg
    else
        mu_bg = mean(z_j)
        var_bg = var(z_j)
        tau_bg = var_bg > 0 ? 1.0 / var_bg : 1.0
    end
    
    prior_mu_0 = isnothing(mu_0) ? mu_bg : mu_0
    prior_b_0 = isnothing(b_0) ? a_0 * max(var_bg, 1e-5) : b_0
    
    # Initialize variational parameters with a slight random perturbation to break symmetry
    mu_star = mu_bg .+ randn(K) .* (sqrt(max(var_bg, 1e-5)) * 0.1)
    kappa_star = fill(kappa_0 + 1.0, K)
    a_star = fill(a_0 + 0.5, K)
    b_star = fill(prior_b_0 + max(var_bg, 1e-5) * 0.5, K)
    
    return LogNormalMargin(prior_mu_0, kappa_0, a_0, prior_b_0, mu_star, kappa_star, a_star, b_star, mu_bg, tau_bg)
end

function update_margin!(margin::LogNormalMargin, y_j::AbstractVector, w::AbstractMatrix, gamma_j::AbstractVector)
    n, K = size(w)
    z_j = log.(y_j)
    for k in 1:K
        sum_w_gamma = 0.0
        sum_w_gamma_z = 0.0
        sum_w_gamma_z2 = 0.0
        for i in 1:n
            val = w[i, k] * gamma_j[i]
            sum_w_gamma += val
            sum_w_gamma_z += val * z_j[i]
            sum_w_gamma_z2 += val * (z_j[i]^2)
        end
        
        margin.kappa_star[k] = margin.kappa_0 + sum_w_gamma
        margin.mu_star[k] = (margin.kappa_0 * margin.mu_0 + sum_w_gamma_z) / margin.kappa_star[k]
        margin.a_star[k] = margin.a_0 + 0.5 * sum_w_gamma
        
        b_term = margin.b_0 + 0.5 * (sum_w_gamma_z2 + margin.kappa_0 * (margin.mu_0^2) - margin.kappa_star[k] * (margin.mu_star[k]^2))
        margin.b_star[k] = max(b_term, 1e-10)
    end
end

function expected_log_density(margin::LogNormalMargin, y_j::AbstractVector)
    n = length(y_j)
    K = length(margin.mu_star)
    eld = Matrix{Float64}(undef, n, K)
    z_j = log.(y_j)
    
    for k in 1:K
        # E[ln tau] = digamma(a) - ln(b)
        e_ln_tau = digamma(margin.a_star[k]) - log(margin.b_star[k])
        # E[tau] = a / b
        e_tau = margin.a_star[k] / margin.b_star[k]
        
        base_const = 0.5 * (-log(2 * pi) + e_ln_tau) - 0.5 / margin.kappa_star[k]
        for i in 1:n
            eld[i, k] = -z_j[i] + base_const - 0.5 * e_tau * ((z_j[i] - margin.mu_star[k])^2)
        end
    end
    return eld
end

function background_log_density(margin::LogNormalMargin, y_j::AbstractVector)
    n = length(y_j)
    bld = Vector{Float64}(undef, n)
    z_j = log.(y_j)
    base_const = 0.5 * (-log(2 * pi) + log(margin.tau_bg))
    for i in 1:n
        bld[i] = -z_j[i] + base_const - 0.5 * margin.tau_bg * ((z_j[i] - margin.mu_bg)^2)
    end
    return bld
end

function expected_kl_divergence(margin::LogNormalMargin)
    K = length(margin.mu_star)
    kl = Vector{Float64}(undef, K)
    # KL divergence is invariant under invertible smooth mapping z = ln(y)
    for k in 1:K
        a_star = margin.a_star[k]
        b_star = margin.b_star[k]
        kappa_star = margin.kappa_star[k]
        mu_star = margin.mu_star[k]
        
        e_tau = a_star / b_star
        e_ln_tau = digamma(a_star) - log(b_star)
        
        term1 = 0.5 * (log(margin.tau_bg) - e_ln_tau)
        term2 = 0.5 * margin.tau_bg * ((mu_star - margin.mu_bg)^2 + 1.0 / kappa_star + b_star / a_star)
        
        kl[k] = max(term1 + term2 - 0.5, 0.0)
    end
    return kl
end

function predictive_density(margin::LogNormalMargin, y_new::AbstractVector)
    n_new = length(y_new)
    K = length(margin.mu_star)
    pred = Matrix{Float64}(undef, n_new, K)
    z_new = log.(y_new)
    
    for k in 1:K
        df = 2.0 * margin.a_star[k]
        scale_sq = (margin.b_star[k] * (margin.kappa_star[k] + 1.0)) / (margin.a_star[k] * margin.kappa_star[k])
        scale = sqrt(scale_sq)
        
        log_norm_const = loggamma((df + 1.0) / 2.0) - loggamma(df / 2.0) - 0.5 * log(pi * df) - log(scale)
        for i in 1:n_new
            diff = (z_new[i] - margin.mu_star[k]) / scale
            log_val = -z_new[i] + log_norm_const - ((df + 1.0) / 2.0) * log(1.0 + (diff^2) / df)
            pred[i, k] = exp(log_val)
        end
    end
    return pred
end

function kl_from_prior(margin::LogNormalMargin)
    kl = 0.0
    for k in 1:length(margin.mu_star)
        a  = margin.a_star[k];   b  = margin.b_star[k]
        κ  = margin.kappa_star[k]; μ = margin.mu_star[k]
        a0 = margin.a_0;          b0 = margin.b_0
        κ0 = margin.kappa_0;      μ0 = margin.mu_0
        kl += (a - a0) * digamma(a) - loggamma(a) + loggamma(a0) +
              a0 * log(b / b0) - a + b0 * a / b - 0.5 +
              0.5 * log(κ / κ0) + 0.5 * κ0 / κ +
              0.5 * κ0 * (μ - μ0)^2 * a / b
    end
    return kl
end

function hellinger_divergence(margin::LogNormalMargin)
    K = length(margin.mu_star)
    h2 = zeros(Float64, K)
    tau_0 = max(margin.tau_bg, 1e-10)
    mu_0 = margin.mu_bg

    for k in 1:K
        tau_k = max(margin.a_star[k] / margin.b_star[k], 1e-10)
        mu_k = margin.mu_star[k]

        ratio = (4.0 * tau_k * tau_0) / ((tau_k + tau_0)^2)
        exponent = -0.25 * (tau_k * tau_0 * (mu_k - mu_0)^2) / (tau_k + tau_0)
        bc = (ratio^0.25) * exp(exponent)
        h2[k] = clamp(1.0 - bc, 0.0, 1.0)
    end
    return h2
end

