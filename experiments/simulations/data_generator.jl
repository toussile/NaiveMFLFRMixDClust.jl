# ==============================================================================
# data_generator.jl — Synthetic Data Generation for Heterogeneous Clustering
# ==============================================================================

using Random

# --- Helper Samplers ---

function rand_multinomial(N::Int, p_vec::Vector{Float64})
    C = length(p_vec)
    counts = zeros(Int, C)
    cum_p = cumsum(p_vec)
    for _ in 1:N
        u = rand()
        idx = findfirst(x -> u <= x, cum_p)
        counts[isnothing(idx) ? C : idx] += 1
    end
    return counts
end

function rand_poisson(lambda::Real)
    L = exp(-lambda)
    k = 0
    p = 1.0
    while p > L
        k += 1
        p *= rand()
    end
    return k - 1
end

function rand_gamma_shape2(rate::Real)
    u1 = rand()
    u2 = rand()
    return -log(u1 * u2) / rate
end

"""
    generate_synthetic_dataset(n, p_act, p_noise, K_true=3; seed=42, model_type=1)

Generate a heterogeneous dataset with 4 exponential-family blocks:
- Gaussian (continuous)
- Poisson (count)
- Multinomial (categorical, C=3, N=15)
- Gamma (positive skewed, shape=2.0)

Signal tiers: f_j ∈ {1.00, 0.75, 0.50, 0.25} within each active block.
model_type = 1: Global relevance (SFRM)
model_type = 2: Local relevance (LFRM, active in clusters 1 & 2, noise in cluster 3)
"""
function generate_synthetic_dataset(n::Int, p_act::Int, p_noise::Int, K_true::Int=3;
                                   seed::Int=42, model_type::Int=1)
    Random.seed!(seed)

    # 1. Balanced cluster assignments
    true_z = Int[]
    for k in 1:K_true
        size_k = floor(Int, n / K_true)
        if k == K_true
            size_k = n - length(true_z)
        end
        append!(true_z, fill(k, size_k))
    end

    p_total = p_act + p_noise
    dataset = Vector{Any}(undef, p_total)
    active_mask = fill(false, p_total)

    # Distribute features across the 4 margin types
    act_per_type   = floor(Int, p_act / 4)
    noise_per_type = floor(Int, p_noise / 4)

    act_counts   = [act_per_type, act_per_type, act_per_type, p_act - 3 * act_per_type]
    noise_counts = [noise_per_type, noise_per_type, noise_per_type, p_noise - 3 * noise_per_type]

    feature_idx = 1

    # --- 1. GAUSSIAN FEATURES ---
    means_base = [-2.0, 0.0, 2.0]
    mu_bg = 0.0
    for j in 1:act_counts[1]
        y_j = Float64[]
        scale = model_type == 1 ? max(1.0 - 0.25 * (j - 1), 0.25) : 1.0
        means = mu_bg .+ scale .* (means_base .- mu_bg)
        is_local = (model_type == 2 && j % 2 == 0)

        for k in true_z
            if k == 1
                push!(y_j, means[1] + randn() * 0.5)
            elseif k == 2
                push!(y_j, means[2] + randn() * 0.5)
            else # k == 3
                if is_local
                    push!(y_j, mu_bg + randn() * 1.2) # background noise
                else
                    push!(y_j, means[3] + randn() * 0.5)
                end
            end
        end
        dataset[feature_idx] = y_j
        active_mask[feature_idx] = true
        feature_idx += 1
    end
    for _ in 1:noise_counts[1]
        dataset[feature_idx] = randn(n) .* 1.2
        feature_idx += 1
    end

    # --- 2. POISSON FEATURES ---
    rates_base = [1.0, 6.0, 15.0]
    lambda_bg = 5.0
    for j in 1:act_counts[2]
        y_j = Float64[]
        scale = model_type == 1 ? max(1.0 - 0.25 * (j - 1), 0.25) : 1.0
        rates = lambda_bg .+ scale .* (rates_base .- lambda_bg)
        is_local = (model_type == 2 && j % 2 == 0)

        for k in true_z
            if k == 1
                push!(y_j, Float64(rand_poisson(rates[1])))
            elseif k == 2
                push!(y_j, Float64(rand_poisson(rates[2])))
            else # k == 3
                if is_local
                    push!(y_j, Float64(rand_poisson(lambda_bg)))
                else
                    push!(y_j, Float64(rand_poisson(rates[3])))
                end
            end
        end
        dataset[feature_idx] = y_j
        active_mask[feature_idx] = true
        feature_idx += 1
    end
    for _ in 1:noise_counts[2]
        dataset[feature_idx] = [Float64(rand_poisson(lambda_bg)) for _ in 1:n]
        feature_idx += 1
    end

    # --- 3. MULTINOMIAL FEATURES ---
    probs_base = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ]
    p_bg = [1/3, 1/3, 1/3]
    for j in 1:act_counts[3]
        y_j = Vector{Int}[]
        scale = model_type == 1 ? max(1.0 - 0.25 * (j - 1), 0.25) : 1.0
        probs = [p_bg .+ scale .* (pb .- p_bg) for pb in probs_base]
        is_local = (model_type == 2 && j % 2 == 0)

        for k in true_z
            if k == 1
                push!(y_j, rand_multinomial(15, probs[1]))
            elseif k == 2
                push!(y_j, rand_multinomial(15, probs[2]))
            else # k == 3
                if is_local
                    push!(y_j, rand_multinomial(15, p_bg))
                else
                    push!(y_j, rand_multinomial(15, probs[3]))
                end
            end
        end
        dataset[feature_idx] = y_j
        active_mask[feature_idx] = true
        feature_idx += 1
    end
    for _ in 1:noise_counts[3]
        dataset[feature_idx] = [rand_multinomial(15, p_bg) for _ in 1:n]
        feature_idx += 1
    end

    # --- 4. GAMMA FEATURES ---
    rates_gamma_base = [1.0, 5.0, 0.2]
    rate_gamma_bg = 2.0
    for j in 1:act_counts[4]
        y_j = Float64[]
        scale = model_type == 1 ? max(1.0 - 0.25 * (j - 1), 0.25) : 1.0
        rates_g = rate_gamma_bg .+ scale .* (rates_gamma_base .- rate_gamma_bg)
        is_local = (model_type == 2 && j % 2 == 0)

        for k in true_z
            if k == 1
                push!(y_j, rand_gamma_shape2(rates_g[1]))
            elseif k == 2
                push!(y_j, rand_gamma_shape2(rates_g[2]))
            else # k == 3
                if is_local
                    push!(y_j, rand_gamma_shape2(rate_gamma_bg))
                else
                    push!(y_j, rand_gamma_shape2(rates_g[3]))
                end
            end
        end
        dataset[feature_idx] = y_j
        active_mask[feature_idx] = true
        feature_idx += 1
    end
    for _ in 1:noise_counts[4]
        dataset[feature_idx] = [rand_gamma_shape2(rate_gamma_bg) for _ in 1:n]
        feature_idx += 1
    end

    return dataset, true_z, active_mask
end
