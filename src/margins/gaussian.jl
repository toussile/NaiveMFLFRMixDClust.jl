mutable struct GaussianMargin <: AbstractMargin
    # Prior hyperparameters
    mu_0::Float64
    kappa_0::Float64
    a_0::Float64
    b_0::Float64
    
    # Variational parameters (length K)
    mu_star::Vector{Float64}
    kappa_star::Vector{Float64}
    a_star::Vector{Float64}
    b_star::Vector{Float64}
    
    # Background parameters (fixed)
    mu_bg::Float64
    tau_bg::Float64 # precision 1/σ^2
end

function GaussianMargin(y_j::AbstractVector, K::Int; 
                        mu_0=nothing, kappa_0=0.05, a_0=3.0, b_0=nothing,
                        robust::Bool=true)
    if robust
        med = Float64(median(y_j))
        mad_val = 1.4826 * Float64(median(abs.(y_j .- med)))
        sigma_bg = mad_val > 1e-6 ? mad_val : (std(y_j) > 1e-6 ? Float64(std(y_j)) : 1.0)
        mu_bg = med
        var_bg = sigma_bg^2
        tau_bg = 1.0 / var_bg
    else
        mu_bg = mean(y_j)
        var_bg = var(y_j)
        tau_bg = var_bg > 0 ? 1.0 / var_bg : 1.0
    end
    
    prior_mu_0 = isnothing(mu_0) ? mu_bg : mu_0
    prior_b_0 = isnothing(b_0) ? a_0 * max(var_bg, 1e-5) : b_0
    
    # Initialize variational parameters with a slight random perturbation to break symmetry
    mu_star = mu_bg .+ randn(K) .* (sqrt(max(var_bg, 1e-5)) * 0.1)
    kappa_star = fill(kappa_0 + 1.0, K)
    a_star = fill(a_0 + 0.5, K)
    b_star = fill(prior_b_0 + max(var_bg, 1e-5) * 0.5, K)
    
    return GaussianMargin(prior_mu_0, kappa_0, a_0, prior_b_0, mu_star, kappa_star, a_star, b_star, mu_bg, tau_bg)
end

function update_margin!(margin::GaussianMargin, y_j::AbstractVector, w::AbstractMatrix, gamma_j::AbstractVector)
    n, K = size(w)
    for k in 1:K
        sum_w_gamma = 0.0
        sum_w_gamma_y = 0.0
        sum_w_gamma_y2 = 0.0
        for i in 1:n
            val = w[i, k] * gamma_j[i]
            sum_w_gamma += val
            sum_w_gamma_y += val * y_j[i]
            sum_w_gamma_y2 += val * (y_j[i]^2)
        end
        
        margin.kappa_star[k] = margin.kappa_0 + sum_w_gamma
        margin.mu_star[k] = (margin.kappa_0 * margin.mu_0 + sum_w_gamma_y) / margin.kappa_star[k]
        margin.a_star[k] = margin.a_0 + 0.5 * sum_w_gamma
        
        b_term = margin.b_0 + 0.5 * (sum_w_gamma_y2 + margin.kappa_0 * (margin.mu_0^2) - margin.kappa_star[k] * (margin.mu_star[k]^2))
        margin.b_star[k] = max(b_term, 1e-10)
    end
end

function expected_log_density(margin::GaussianMargin, y_j::AbstractVector)
    n = length(y_j)
    K = length(margin.mu_star)
    eld = Matrix{Float64}(undef, n, K)
    for k in 1:K
        psi_a = digamma(margin.a_star[k])
        ln_b = log(margin.b_star[k])
        a_over_b = margin.a_star[k] / margin.b_star[k]
        inv_kappa = 1.0 / margin.kappa_star[k]
        mu_k = margin.mu_star[k]
        
        for i in 1:n
            eld[i, k] = -0.5 * log(2π) + 0.5 * (psi_a - ln_b) - 0.5 * a_over_b * (y_j[i] - mu_k)^2 - 0.5 * inv_kappa
        end
    end
    return eld
end

function background_log_density(margin::GaussianMargin, y_j::AbstractVector)
    n = length(y_j)
    bld = Vector{Float64}(undef, n)
    ln_tau = log(margin.tau_bg)
    for i in 1:n
        bld[i] = -0.5 * log(2π) + 0.5 * ln_tau - 0.5 * margin.tau_bg * (y_j[i] - margin.mu_bg)^2
    end
    return bld
end

function expected_kl_divergence(margin::GaussianMargin)
    K = length(margin.mu_star)
    kl = Vector{Float64}(undef, K)
    for k in 1:K
        psi_a = digamma(margin.a_star[k])
        ln_b = log(margin.b_star[k])
        a_minus_1 = max(margin.a_star[k] - 1.0, 1e-5)
        b_over_aminus1 = margin.b_star[k] / a_minus_1
        
        term1 = psi_a - ln_b - log(margin.tau_bg) - 1.0
        term2 = margin.tau_bg * b_over_aminus1
        term3 = margin.tau_bg * ((margin.mu_star[k] - margin.mu_bg)^2 + b_over_aminus1 / margin.kappa_star[k])
        
        kl[k] = 0.5 * (term1 + term2 + term3)
    end
    return kl
end

function predictive_density(margin::GaussianMargin, y_new::AbstractVector)
    n_new = length(y_new)
    K = length(margin.mu_star)
    pred = Matrix{Float64}(undef, n_new, K)
    
    for k in 1:K
        a = margin.a_star[k]
        b = margin.b_star[k]
        mu = margin.mu_star[k]
        kappa = margin.kappa_star[k]
        
        log_coeff = loggamma(a + 0.5) - loggamma(a) - 0.5 * (log(2 * pi) + log(b) + log(1.0 + 1.0/kappa))
        denom = 2 * b * (1.0 + 1.0/kappa)
        for i in 1:n_new
            log_val = log_coeff - (a + 0.5) * log(1.0 + (y_new[i] - mu)^2 / denom)
            pred[i, k] = exp(log_val)
        end
    end
    return pred
end

function kl_from_prior(margin::GaussianMargin)
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

function update_background!(margin::GaussianMargin, y_j::AbstractVector, gamma_j::AbstractVector)
    n = length(y_j)
    sum_w = 0.0
    sum_wy = 0.0
    for i in 1:n
        w_i = 1.0 - gamma_j[i]
        sum_w += w_i
        sum_wy += w_i * y_j[i]
    end
    
    if sum_w > 1e-5
        mu_bg = sum_wy / sum_w
        sum_wy2 = 0.0
        for i in 1:n
            w_i = 1.0 - gamma_j[i]
            sum_wy2 += w_i * (y_j[i] - mu_bg)^2
        end
        var_bg = sum_wy2 / sum_w
        tau_bg = var_bg > 1e-10 ? 1.0 / var_bg : 1e10
        
        margin.mu_bg = mu_bg
        margin.tau_bg = tau_bg
    end
end

function hellinger_divergence(margin::GaussianMargin)
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


