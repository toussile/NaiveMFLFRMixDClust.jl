mutable struct MultinomialMargin <: AbstractMargin
    # Prior hyperparameters
    varphi::Vector{Float64}         # Dirichlet prior parameter of length C_j
    
    # Variational parameters (size K x C_j)
    varphi_star::Matrix{Float64}
    
    # Background parameters (fixed probabilities of length C_j)
    phi_bg::Vector{Float64}
end

function MultinomialMargin(y_j::AbstractVector, K::Int; 
                           varphi=nothing, smoothed::Bool=true,
                                           robust::Union{Nothing, Bool}=nothing)
    smoothed = _background_flag(smoothed, robust, :smoothed, :MultinomialMargin)
    n = length(y_j)
    C_j = length(y_j[1])
    
    # Estimate background parameters from global count sums
    sum_y = zeros(C_j)
    for i in 1:n
        sum_y .+= y_j[i]
    end
    total_counts = sum(sum_y)
    if smoothed
        phi_bg = (sum_y .+ 0.5) ./ (total_counts + 0.5 * C_j)
    else
        phi_bg = total_counts > 0 ? sum_y ./ total_counts : fill(1.0 / C_j, C_j)
    end
    
    # Ensure Dirichlet prior parameter
    prior_varphi = isnothing(varphi) ? phi_bg .* 3.0 : varphi
    
    # Initialize variational parameters with slight random perturbations
    varphi_star = Matrix{Float64}(undef, K, C_j)
    for k in 1:K
        varphi_star[k, :] = prior_varphi .+ rand(C_j) .* 0.5
    end
    
    return MultinomialMargin(prior_varphi, varphi_star, phi_bg)
end

function log_multinomial_coeff(y_i::AbstractVector)
    N = sum(y_i)
    term = loggamma(N + 1)
    for val in y_i
        term -= loggamma(val + 1)
    end
    return term
end

function update_margin!(margin::MultinomialMargin, y_j::AbstractVector, w::AbstractMatrix, gamma_j::AbstractVector)
    n, K = size(w)
    C_j = length(margin.varphi)
    for k in 1:K
        for c in 1:C_j
            sum_w_gamma_y = 0.0
            for i in 1:n
                sum_w_gamma_y += w[i, k] * gamma_j[i] * y_j[i][c]
            end
            margin.varphi_star[k, c] = margin.varphi[c] + sum_w_gamma_y
        end
    end
end

function expected_log_density(margin::MultinomialMargin, y_j::AbstractVector)
    n = length(y_j)
    K = size(margin.varphi_star, 1)
    C_j = length(margin.varphi)
    eld = Matrix{Float64}(undef, n, K)
    
    # Precompute Psi values for the sums
    psi_sum_varphi = Vector{Float64}(undef, K)
    for k in 1:K
        psi_sum_varphi[k] = digamma(sum(margin.varphi_star[k, :]))
    end
    
    for k in 1:K
        psi_varphi = [digamma(margin.varphi_star[k, c]) for c in 1:C_j]
        for i in 1:n
            log_coeff = log_multinomial_coeff(y_j[i])
            sum_term = 0.0
            for c in 1:C_j
                sum_term += y_j[i][c] * (psi_varphi[c] - psi_sum_varphi[k])
            end
            eld[i, k] = log_coeff + sum_term
        end
    end
    return eld
end

function background_log_density(margin::MultinomialMargin, y_j::AbstractVector)
    n = length(y_j)
    C_j = length(margin.varphi)
    bld = Vector{Float64}(undef, n)
    
    # Precompute log background probabilities
    log_phi_bg = [log(max(margin.phi_bg[c], 1e-15)) for c in 1:C_j]
    
    for i in 1:n
        log_coeff = log_multinomial_coeff(y_j[i])
        sum_term = 0.0
        for c in 1:C_j
            sum_term += y_j[i][c] * log_phi_bg[c]
        end
        bld[i] = log_coeff + sum_term
    end
    return bld
end

function expected_kl_divergence(margin::MultinomialMargin)
    K = size(margin.varphi_star, 1)
    C_j = length(margin.varphi)
    kl = Vector{Float64}(undef, K)
    
    for k in 1:K
        varphi_sum_k = sum(margin.varphi_star[k, :])
        psi_sum_plus_1 = digamma(varphi_sum_k + 1.0)
        
        sum_term = 0.0
        for c in 1:C_j
            prop = margin.varphi_star[k, c] / varphi_sum_k
            psi_c_plus_1 = digamma(margin.varphi_star[k, c] + 1.0)
            ln_bg = log(max(margin.phi_bg[c], 1e-15))
            
            sum_term += prop * (psi_c_plus_1 - psi_sum_plus_1 - ln_bg)
        end
        kl[k] = sum_term
    end
    return kl
end

function predictive_density(margin::MultinomialMargin, y_new::AbstractVector)
    n_new = length(y_new)
    K = size(margin.varphi_star, 1)
    C_j = length(margin.varphi)
    pred = Matrix{Float64}(undef, n_new, K)
    
    for k in 1:K
        varphi_k = margin.varphi_star[k, :]
        sum_varphi_k = sum(varphi_k)
        
        log_dir_coeff = loggamma(sum_varphi_k) - sum(loggamma.(varphi_k))
        
        for i in 1:n_new
            y = y_new[i]
            N = sum(y)
            log_mult_coeff = loggamma(N + 1.0) - sum(loggamma.(y .+ 1.0))
            
            sum_y_varphi = sum(varphi_k .+ y)
            log_factor = sum(loggamma.(varphi_k .+ y)) - loggamma(sum_y_varphi)
            
            log_val = log_mult_coeff + log_dir_coeff + log_factor
            pred[i, k] = exp(log_val)
        end
    end
    return pred
end

function kl_from_prior(margin::MultinomialMargin)
    K   = size(margin.varphi_star, 1)
    C   = length(margin.varphi)
    phi0 = margin.varphi
    log_B_prior = sum(loggamma.(phi0)) - loggamma(sum(phi0))
    kl = 0.0
    for k in 1:K
        phi_k     = margin.varphi_star[k, :]
        sum_phi_k = sum(phi_k)
        log_B_q   = sum(loggamma.(phi_k)) - loggamma(sum_phi_k)
        psi_sum   = digamma(sum_phi_k)
        kl_k = log_B_prior - log_B_q
        for c in 1:C
            kl_k += (phi_k[c] - phi0[c]) * (digamma(phi_k[c]) - psi_sum)
        end
        kl += kl_k
    end
    return kl
end

function update_background!(margin::MultinomialMargin, y_j::AbstractVector, gamma_j::AbstractVector)
    n = length(y_j)
    C_j = length(margin.varphi)
    sum_wy = zeros(C_j)
    sum_w = 0.0
    for i in 1:n
        w_i = 1.0 - gamma_j[i]
        sum_wy .+= w_i .* y_j[i]
        sum_w += w_i * sum(y_j[i])
    end
    
    if sum_w > 1e-5
        sum_total = sum(sum_wy)
        if sum_total > 1e-10
            margin.phi_bg = sum_wy ./ sum_total
        else
            margin.phi_bg = fill(1.0 / C_j, C_j)
        end
    end
end

function hellinger_divergence(margin::MultinomialMargin)
    K = size(margin.varphi_star, 1)
    C_j = length(margin.varphi)
    h2 = Vector{Float64}(undef, K)
    phi0 = margin.phi_bg
    for k in 1:K
        phi_k = margin.varphi_star[k, :]
        sum_phi_k = sum(phi_k)
        pi_k = phi_k ./ sum_phi_k
        bc = 0.0
        for c in 1:C_j
            bc += sqrt(max(pi_k[c], 0.0) * max(phi0[c], 0.0))
        end
        bc = clamp(bc, 0.0, 1.0)
        h2[k] = clamp(1.0 - bc, 0.0, 1.0)
    end
    return h2
end



