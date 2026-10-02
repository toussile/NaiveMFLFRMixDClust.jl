mutable struct PoissonMargin <: AbstractMargin
    # Prior hyperparameters
    a_0::Float64
    b_0::Float64
    
    # Variational parameters (length K)
    a_star::Vector{Float64}
    b_star::Vector{Float64}
    
    # Background parameter (fixed rate)
    lambda_bg::Float64
end

function PoissonMargin(y_j::AbstractVector, K::Int; 
                       a_0=2.0, b_0=nothing, median_based::Bool=true,
                                             robust::Union{Nothing, Bool}=nothing)
    median_based = _background_flag(median_based, robust, :median_based, :PoissonMargin)
    if median_based
        med = Float64(median(y_j))
        if med > 0
            lambda_bg = max(0.1, med)
        else
            # 10% trimmed mean fallback when median is zero
            s_y = sort(y_j)
            trim_idx = max(1, round(Int, 0.1 * length(y_j)))
            trimmed_val = mean(s_y[trim_idx:end])
            lambda_bg = max(0.1, Float64(trimmed_val))
        end
    else
        lambda_bg = mean(y_j)
    end
    
    prior_b_0 = isnothing(b_0) ? a_0 / max(lambda_bg, 1e-5) : b_0
    
    # Initialize variational parameters with a slight random perturbation to break symmetry
    a_star = fill(a_0, K) .+ rand(K) .* 0.5
    b_star = fill(prior_b_0, K)
    
    return PoissonMargin(a_0, prior_b_0, a_star, b_star, lambda_bg)
end

function update_margin!(margin::PoissonMargin, y_j::AbstractVector, w::AbstractMatrix, gamma_j::AbstractVector)
    n, K = size(w)
    for k in 1:K
        sum_w_gamma_y = 0.0
        sum_w_gamma = 0.0
        for i in 1:n
            val = w[i, k] * gamma_j[i]
            sum_w_gamma_y += val * y_j[i]
            sum_w_gamma += val
        end
        margin.a_star[k] = margin.a_0 + sum_w_gamma_y
        margin.b_star[k] = margin.b_0 + sum_w_gamma
    end
end

function expected_log_density(margin::PoissonMargin, y_j::AbstractVector)
    n = length(y_j)
    K = length(margin.a_star)
    eld = Matrix{Float64}(undef, n, K)
    for k in 1:K
        psi_a = digamma(margin.a_star[k])
        ln_b = log(margin.b_star[k])
        a_over_b = margin.a_star[k] / margin.b_star[k]
        
        for i in 1:n
            log_fact = loggamma(y_j[i] + 1)
            eld[i, k] = y_j[i] * (psi_a - ln_b) - a_over_b - log_fact
        end
    end
    return eld
end

function background_log_density(margin::PoissonMargin, y_j::AbstractVector)
    n = length(y_j)
    bld = Vector{Float64}(undef, n)
    ln_lambda = log(max(margin.lambda_bg, 1e-15))
    for i in 1:n
        log_fact = loggamma(y_j[i] + 1)
        bld[i] = y_j[i] * ln_lambda - margin.lambda_bg - log_fact
    end
    return bld
end

function expected_kl_divergence(margin::PoissonMargin)
    K = length(margin.a_star)
    kl = Vector{Float64}(undef, K)
    ln_lambda_bg = log(max(margin.lambda_bg, 1e-15))
    for k in 1:K
        a_over_b = margin.a_star[k] / margin.b_star[k]
        psi_a_plus_1 = digamma(margin.a_star[k] + 1.0)
        ln_b = log(margin.b_star[k])
        
        kl[k] = a_over_b * (psi_a_plus_1 - ln_b - ln_lambda_bg - 1.0) + margin.lambda_bg
    end
    return kl
end

function predictive_density(margin::PoissonMargin, y_new::AbstractVector)
    n_new = length(y_new)
    K = length(margin.a_star)
    pred = Matrix{Float64}(undef, n_new, K)
    
    for k in 1:K
        a = margin.a_star[k]
        b = margin.b_star[k]
        
        log_b = log(b)
        log_b_plus_1 = log(b + 1.0)
        
        for i in 1:n_new
            y = y_new[i]
            log_val = loggamma(a + y) - loggamma(a) - loggamma(y + 1.0) + a * log_b - (a + y) * log_b_plus_1
            pred[i, k] = exp(log_val)
        end
    end
    return pred
end

function kl_from_prior(margin::PoissonMargin)
    kl = 0.0
    a0 = margin.a_0;  b0 = margin.b_0
    for k in 1:length(margin.a_star)
        a = margin.a_star[k];  b = margin.b_star[k]
        kl += (a - a0) * digamma(a) - loggamma(a) + loggamma(a0) +
              a0 * log(b / b0) - a + b0 * a / b
    end
    return kl
end

function update_background!(margin::PoissonMargin, y_j::AbstractVector, gamma_j::AbstractVector)
    n = length(y_j)
    sum_w = 0.0
    sum_wy = 0.0
    for i in 1:n
        w_i = 1.0 - gamma_j[i]
        sum_w += w_i
        sum_wy += w_i * y_j[i]
    end
    
    if sum_w > 1e-5
        margin.lambda_bg = max(sum_wy / sum_w, 1e-10)
    end
end

function hellinger_divergence(margin::PoissonMargin)
    K = length(margin.a_star)
    h2 = Vector{Float64}(undef, K)
    lambda0 = max(margin.lambda_bg, 1e-15)
    sqrt_lambda0 = sqrt(lambda0)
    for k in 1:K
        lambdak = max(margin.a_star[k] / margin.b_star[k], 1e-15)
        bc = exp(-0.5 * (sqrt(lambdak) - sqrt_lambda0)^2)
        bc = clamp(bc, 0.0, 1.0)
        h2[k] = clamp(1.0 - bc, 0.0, 1.0)
    end
    return h2
end



