mutable struct ExponentialMargin <: AbstractMargin
    # Prior hyperparameters
    alpha_0::Float64
    beta_0::Float64
    
    # Variational parameters (length K): q(λ_k) = Gamma(alpha_star[k], beta_star[k])
    alpha_star::Vector{Float64}
    beta_star::Vector{Float64}
    
    # Background parameter (fixed rate)
    lambda_bg::Float64
end

function ExponentialMargin(y_j::AbstractVector, K::Int; 
                           alpha_0=2.0, beta_0=nothing, robust::Bool=true)
    any(x -> x < 0, y_j) && throw(ArgumentError("ExponentialMargin requires non-negative observations."))
    
    if robust
        med = Float64(median(y_j))
        lambda_bg = log(2.0) / max(med, 1e-4)
    else
        m = Float64(mean(y_j))
        lambda_bg = 1.0 / max(m, 1e-4)
    end
    
    prior_beta_0 = isnothing(beta_0) ? alpha_0 / max(lambda_bg, 1e-5) : beta_0
    
    alpha_star = fill(alpha_0, K) .+ rand(K) .* 0.5
    beta_star = fill(prior_beta_0, K)
    
    return ExponentialMargin(alpha_0, prior_beta_0, alpha_star, beta_star, lambda_bg)
end

function update_margin!(margin::ExponentialMargin, y_j::AbstractVector, w::AbstractMatrix, gamma_j::AbstractVector)
    n, K = size(w)
    for k in 1:K
        sum_w_gamma = 0.0
        sum_w_gamma_y = 0.0
        for i in 1:n
            val = w[i, k] * gamma_j[i]
            sum_w_gamma += val
            sum_w_gamma_y += val * y_j[i]
        end
        margin.alpha_star[k] = margin.alpha_0 + sum_w_gamma
        margin.beta_star[k] = margin.beta_0 + sum_w_gamma_y
    end
end

function expected_log_density(margin::ExponentialMargin, y_j::AbstractVector)
    n = length(y_j)
    K = length(margin.alpha_star)
    eld = Matrix{Float64}(undef, n, K)
    
    for k in 1:K
        # E[ln lambda] = digamma(alpha) - ln(beta)
        e_ln_lambda = digamma(margin.alpha_star[k]) - log(margin.beta_star[k])
        # E[lambda] = alpha / beta
        e_lambda = margin.alpha_star[k] / margin.beta_star[k]
        
        for i in 1:n
            eld[i, k] = e_ln_lambda - e_lambda * y_j[i]
        end
    end
    return eld
end

function background_log_density(margin::ExponentialMargin, y_j::AbstractVector)
    n = length(y_j)
    bld = Vector{Float64}(undef, n)
    ln_lbg = log(margin.lambda_bg)
    for i in 1:n
        bld[i] = ln_lbg - margin.lambda_bg * y_j[i]
    end
    return bld
end

function expected_kl_divergence(margin::ExponentialMargin)
    K = length(margin.alpha_star)
    kl = Vector{Float64}(undef, K)
    ln_lbg = log(margin.lambda_bg)
    
    for k in 1:K
        a = margin.alpha_star[k]
        b = margin.beta_star[k]
        e_ln_lambda = digamma(a) - log(b)
        e_inv_lambda = b / max(a - 1.0, 0.1)
        
        # KL(Exp(λ) || Exp(λ_0)) = ln(λ/λ_0) + λ_0/λ - 1
        val = e_ln_lambda - ln_lbg + margin.lambda_bg * e_inv_lambda - 1.0
        kl[k] = max(val, 0.0)
    end
    return kl
end

function predictive_density(margin::ExponentialMargin, y_new::AbstractVector)
    n_new = length(y_new)
    K = length(margin.alpha_star)
    pred = Matrix{Float64}(undef, n_new, K)
    
    for k in 1:K
        a = margin.alpha_star[k]
        b = margin.beta_star[k]
        # Lomax density: p(y) = a * b^a / (b + y)^(a + 1)
        ln_const = log(a) + a * log(b)
        for i in 1:n_new
            y_val = max(y_new[i], 0.0)
            ln_p = ln_const - (a + 1.0) * log(b + y_val)
            pred[i, k] = exp(ln_p)
        end
    end
    return pred
end

function kl_from_prior(margin::ExponentialMargin)
    kl = 0.0
    for k in 1:length(margin.alpha_star)
        a  = margin.alpha_star[k]
        b  = margin.beta_star[k]
        a0 = margin.alpha_0
        b0 = margin.beta_0
        kl += (a - a0) * digamma(a) - loggamma(a) + loggamma(a0) +
              a0 * log(b / b0) + a * (b0 - b) / b
    end
    return kl
end

function hellinger_divergence(margin::ExponentialMargin)
    K = length(margin.alpha_star)
    h2 = zeros(Float64, K)
    lam_0 = max(margin.lambda_bg, 1e-10)

    for k in 1:K
        lam_k = max(margin.alpha_star[k] / margin.beta_star[k], 1e-10)
        bc = (2.0 * sqrt(lam_k * lam_0)) / (lam_k + lam_0)
        h2[k] = clamp(1.0 - bc, 0.0, 1.0)
    end
    return h2
end

