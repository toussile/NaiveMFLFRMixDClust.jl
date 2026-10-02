mutable struct GammaMargin <: AbstractMargin
    # Prior hyperparameters
    alpha_0::Float64
    beta_0::Float64
    
    # Variational parameters (length K)
    alpha_star::Vector{Float64}
    beta_star::Vector{Float64}
    
    # Background parameters (fixed)
    a_bg::Float64
    b_bg::Float64
    
    # Cluster-specific shape parameter (fixed across clusters)
    a_cl::Float64
end

function GammaMargin(y_j::AbstractVector, K::Int; 
                     alpha_0=3.0, beta_0=nothing, median_based::Bool=true,
                                                  robust::Union{Nothing, Bool}=nothing)
    median_based = _background_flag(median_based, robust, :median_based, :GammaMargin)
    if median_based
        med = Float64(median(y_j))
        mad_val = 1.4826 * Float64(median(abs.(y_j .- med)))
        sigma_bg = mad_val > 1e-6 ? mad_val : (std(y_j) > 1e-6 ? Float64(std(y_j)) : 1.0)
        mu_bg = max(med, 1e-4)
        var_bg = sigma_bg^2
        a_bg = max((mu_bg / sigma_bg)^2, 1e-3)
        b_bg = max(mu_bg / var_bg, 1e-4)
    else
        mu_bg = mean(y_j)
        var_bg = var(y_j)
        a_bg = var_bg > 0 ? (mu_bg^2) / var_bg : 1.0
        b_bg = var_bg > 0 ? mu_bg / var_bg : 1.0
    end
    
    # Cluster shape parameter is fixed and equal to background shape
    a_cl = a_bg
    
    prior_beta_0 = isnothing(beta_0) ? alpha_0 / max(b_bg, 1e-5) : beta_0
    
    # Initialize variational parameters with a slight random perturbation to break symmetry
    alpha_star = fill(alpha_0, K) .+ rand(K) .* 0.5
    beta_star = fill(prior_beta_0, K) .+ rand(K) .* (b_bg * 0.1)
    
    return GammaMargin(alpha_0, prior_beta_0, alpha_star, beta_star, a_bg, b_bg, a_cl)
end

function update_margin!(margin::GammaMargin, y_j::AbstractVector, w::AbstractMatrix, gamma_j::AbstractVector)
    n, K = size(w)
    for k in 1:K
        sum_w_gamma = 0.0
        sum_w_gamma_y = 0.0
        for i in 1:n
            val = w[i, k] * gamma_j[i]
            sum_w_gamma += val
            sum_w_gamma_y += val * y_j[i]
        end
        margin.alpha_star[k] = margin.alpha_0 + margin.a_cl * sum_w_gamma
        margin.beta_star[k] = margin.beta_0 + sum_w_gamma_y
    end
end

function expected_log_density(margin::GammaMargin, y_j::AbstractVector)
    n = length(y_j)
    K = length(margin.alpha_star)
    eld = Matrix{Float64}(undef, n, K)
    
    log_gamma_a_cl = loggamma(margin.a_cl)
    
    for k in 1:K
        psi_alpha = digamma(margin.alpha_star[k])
        ln_beta = log(margin.beta_star[k])
        alpha_over_beta = margin.alpha_star[k] / margin.beta_star[k]
        
        for i in 1:n
            ln_y = log(max(y_j[i], 1e-15))
            eld[i, k] = margin.a_cl * (psi_alpha - ln_beta) - log_gamma_a_cl + (margin.a_cl - 1.0) * ln_y - alpha_over_beta * y_j[i]
        end
    end
    return eld
end

function background_log_density(margin::GammaMargin, y_j::AbstractVector)
    n = length(y_j)
    bld = Vector{Float64}(undef, n)
    
    a = margin.a_bg
    b = margin.b_bg
    log_gamma_a = loggamma(a)
    a_ln_b = a * log(b)
    
    for i in 1:n
        ln_y = log(max(y_j[i], 1e-15))
        bld[i] = a_ln_b - log_gamma_a + (a - 1.0) * ln_y - b * y_j[i]
    end
    return bld
end

function expected_kl_divergence(margin::GammaMargin)
    K = length(margin.alpha_star)
    kl = Vector{Float64}(undef, K)
    
    a_cl = margin.a_cl
    a_bg = margin.a_bg
    b_bg = margin.b_bg
    
    # Precompute terms independent of cluster k
    term_shape = (a_cl - a_bg) * digamma(a_cl) - loggamma(a_cl) + loggamma(a_bg)
    
    for k in 1:K
        psi_alpha = digamma(margin.alpha_star[k])
        ln_beta = log(margin.beta_star[k])
        alpha_minus_1 = max(margin.alpha_star[k] - 1.0, 1e-5)
        
        term_kl_rate = a_bg * (psi_alpha - ln_beta - log(b_bg)) + a_cl * b_bg * (margin.beta_star[k] / alpha_minus_1) - a_cl
        
        kl[k] = term_shape + term_kl_rate
    end
    return kl
end

function predictive_density(margin::GammaMargin, y_new::AbstractVector)
    n_new = length(y_new)
    K = length(margin.alpha_star)
    pred = Matrix{Float64}(undef, n_new, K)
    
    a_cl = margin.a_cl
    log_gamma_a_cl = loggamma(a_cl)
    
    for k in 1:K
        alpha_k = margin.alpha_star[k]
        beta_k = margin.beta_star[k]
        
        log_coeff = loggamma(alpha_k + a_cl) - loggamma(alpha_k) - log_gamma_a_cl + alpha_k * log(beta_k)
        
        for i in 1:n_new
            y = y_new[i]
            log_val = log_coeff + (a_cl - 1.0) * log(max(y, 1e-15)) - (alpha_k + a_cl) * log(beta_k + y)
            pred[i, k] = exp(log_val)
        end
    end
    return pred
end

function kl_from_prior(margin::GammaMargin)
    kl = 0.0
    a0 = margin.alpha_0;  b0 = margin.beta_0
    for k in 1:length(margin.alpha_star)
        a = margin.alpha_star[k];  b = margin.beta_star[k]
        kl += (a - a0) * digamma(a) - loggamma(a) + loggamma(a0) +
              a0 * log(b / b0) - a + b0 * a / b
    end
    return kl
end

function update_background!(margin::GammaMargin, y_j::AbstractVector, gamma_j::AbstractVector)
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
        
        margin.a_bg = var_bg > 1e-10 ? (mu_bg^2) / var_bg : 1.0
        margin.b_bg = var_bg > 1e-10 ? mu_bg / var_bg : 1.0
    end
end

function hellinger_divergence(margin::GammaMargin)
    K = length(margin.alpha_star)
    h2 = Vector{Float64}(undef, K)
    a_cl = margin.a_cl
    a_bg = margin.a_bg
    b_bg = max(margin.b_bg, 1e-15)
    
    half_a_sum = 0.5 * (a_cl + a_bg)
    log_gamma_term = loggamma(half_a_sum) - 0.5 * (loggamma(a_cl) + loggamma(a_bg))
    
    for k in 1:K
        bk = max(margin.alpha_star[k] / margin.beta_star[k], 1e-15)
        log_bc = log_gamma_term + 0.5 * a_cl * log(bk) + 0.5 * a_bg * log(b_bg) - half_a_sum * log(0.5 * (bk + b_bg))
        bc = clamp(exp(log_bc), 0.0, 1.0)
        h2[k] = clamp(1.0 - bc, 0.0, 1.0)
    end
    return h2
end



