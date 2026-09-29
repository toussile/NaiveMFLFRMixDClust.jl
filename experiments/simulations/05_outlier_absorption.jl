# ==============================================================================
# 05_outlier_absorption.jl — Outlier Absorption and Robust Clustering
# ==============================================================================
# Evaluates robust partition recovery and outlier detection under background
# contamination (Section 5.5, Table 4 in manuscript).
#
# Feature ordering from generate_synthetic_dataset(n, p_act=4, p_noise=4, K0):
#   [1] Gaussian active,    [2] Gaussian noise
#   [3] Poisson active,     [4] Poisson noise
#   [5] Multinomial active, [6] Multinomial noise
#   [7] Gamma active,       [8] Gamma noise
# ==============================================================================

include(joinpath(@__DIR__, "common.jl"))

"""
    _safe_vcat(a, b)

Concatenate two feature vectors, preserving correct element types.
For Multinomial features (elements are Vector{Int}), forces the result
to be `Vector{Vector{Int}}` so that `_infer_margin_type` in `cavi.jl`
correctly detects the `:multinomial` margin type via `eltype <: AbstractVector`.
For scalar features, returns a plain `vcat`.
"""
function _safe_vcat(a, b)
    # Detect Multinomial features: first element is a vector
    if length(a) > 0 && first(a) isa AbstractVector
        return Vector{Vector{Int}}(vcat(a, b))
    else
        return vcat(a, b)
    end
end

function run_outlier_absorption(; n_rep::Int = 20)
    @info "Starting Simulation 5: Outlier Absorption & Robust Clustering (n_rep = $n_rep)"

    n_in  = 120
    n_out = 15
    K0, K_fit = 3, 8
    p_act, p_noise = 4, 4  # → 4 features per margin, 8 total
    u0      = 0.01
    tau_out = 0.58

    std_ari_vals = zeros(Float64, n_rep)
    rob_ari_vals = zeros(Float64, n_rep)
    tpr_vals     = zeros(Float64, n_rep)
    fpr_vals     = zeros(Float64, n_rep)
    k_vals       = zeros(Int, n_rep)

    Threads.@threads for rep in 1:n_rep
        seed = GLOBAL_SEED + 5000 + rep

        # ── Generate inliers (120 obs, 4 active + 4 noise features) ──────────
        data_in, true_z_in, _ = generate_synthetic_dataset(
            n_in, p_act, p_noise, K0; seed = seed, model_type = 1
        )

        # ── Generate outliers from background distributions ───────────────────
        # Use a local MersenneTwister for thread safety; use project-native
        # samplers (no Distributions.jl dependency).
        local_rng = MersenneTwister(seed + 999)

        # Background parameters (matching those in data_generator.jl):
        #   Gaussian:     μ_bg = 0.0, σ = 1.2   (noise σ)
        #   Poisson:      λ_bg = 5.0
        #   Multinomial:  p_bg = [1/3, 1/3, 1/3], N = 15
        #   Gamma:        shape = 2, rate = 2.0  (mean = 1)
        #
        # Feature order to match data_in:
        #   [1] Gaussian active,    [2] Gaussian noise
        #   [3] Poisson active,     [4] Poisson noise
        #   [5] Multinomial active, [6] Multinomial noise
        #   [7] Gamma active,       [8] Gamma noise
        p_bg_mult = fill(1.0 / 3.0, 3)
        feat1_out = randn(local_rng, n_out) .* 1.2                        # Gaussian_act bg
        feat2_out = randn(local_rng, n_out) .* 1.2                        # Gaussian_noise bg
        feat3_out = Float64[rand_poisson(5.0) for _ in 1:n_out]           # Poisson_act bg
        feat4_out = Float64[rand_poisson(5.0) for _ in 1:n_out]           # Poisson_noise bg
        feat5_out = [rand_multinomial(15, p_bg_mult) for _ in 1:n_out]    # Mult_act bg
        feat6_out = [rand_multinomial(15, p_bg_mult) for _ in 1:n_out]    # Mult_noise bg
        feat7_out = [rand_gamma_shape2(2.0) for _ in 1:n_out]             # Gamma_act bg
        feat8_out = [rand_gamma_shape2(2.0) for _ in 1:n_out]             # Gamma_noise bg

        data_out = [feat1_out, feat2_out, feat3_out, feat4_out,
                    feat5_out, feat6_out, feat7_out, feat8_out]

        # ── Combine inliers + outliers ────────────────────────────────────────
        data_all   = [_safe_vcat(data_in[j], data_out[j]) for j in 1:8]
        true_z_all = vcat(true_z_in, zeros(Int, n_out))  # 0 = outlier

        # ── Fit overfitted LFRM mixture with sparse Dirichlet prior ──────────
        res = mixClust(data_all, K_fit; model_setting = LFRM(),
                       u0 = u0, compact = false, max_iter = 500, tol = 1e-4)

        k_vals[rep] = map_cluster_order(res)

        # ── Compute per-individual coordinate inactivation rate ───────────────
        # In LFRM, res.pip is N×J (per individual, per feature).
        # coordinate_inactivation_rate(res) computes:
        #   ρ_i = 1 − (1/J) Σ_j φ_{i,j}*   ∈ [0, 1]
        # High ρ_i → individual i is largely in the background.
        rho = coordinate_inactivation_rate(res)  # N-vector

        # ── 1. Standard MAP ARI ───────────────────────────────────────────────
        std_labels = res.labels
        std_ari_vals[rep] = adjusted_rand_index(true_z_all, std_labels)

        # ── 2. Robust MAP: flag high-inactivation individuals as outliers ─────
        rob_labels = copy(std_labels)
        is_flagged = rho .>= tau_out
        rob_labels[is_flagged] .= 0
        rob_ari_vals[rep] = adjusted_rand_index(true_z_all, rob_labels)

        # ── 3. Detection metrics ──────────────────────────────────────────────
        true_outliers = true_z_all .== 0
        tp = sum(is_flagged .& true_outliers)
        fp = sum(is_flagged .& .!true_outliers)
        fn = sum(.!is_flagged .& true_outliers)
        tn = sum(.!is_flagged .& .!true_outliers)

        tpr_vals[rep] = tp / max(tp + fn, 1)
        fpr_vals[rep] = fp / max(fp + tn, 1)
    end

    results = (;
        std_ari = (; mean = mean(std_ari_vals), std = std(std_ari_vals)),
        rob_ari = (; mean = mean(rob_ari_vals), std = std(rob_ari_vals)),
        tpr     = (; mean = mean(tpr_vals),     std = std(tpr_vals)),
        fpr     = (; mean = mean(fpr_vals),     std = std(fpr_vals)),
        k_order = (; mean = mean(k_vals),       std = std(k_vals))
    )

    println("\n=== Simulation 5 Results: Outlier Detection and Robust Partition Recovery ===")
    println("Metric                    | Value")
    println("-"^52)
    @printf("Standard MAP ARI          | %5.3f ± %5.3f\n", results.std_ari.mean, results.std_ari.std)
    @printf("Robust MAP ARI (τ = 0.58) | %5.3f ± %5.3f\n", results.rob_ari.mean, results.rob_ari.std)
    @printf("Outlier TPR               | %5.1f%% ± %4.1f%%\n", results.tpr.mean * 100, results.tpr.std * 100)
    @printf("Outlier FPR               | %5.1f%% ± %4.1f%%\n", results.fpr.mean * 100, results.fpr.std * 100)
    @printf("Estimated Cluster Order K̂ | %4.2f ± %4.2f\n", results.k_order.mean, results.k_order.std)

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_outlier_absorption()
end
