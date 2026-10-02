# ==============================================================================
# 03_shared_vs_local_saliency.jl — Shared (SFRM) vs. Local (LFRM) Feature Saliency
# ==============================================================================
# Evaluates detection of locally active features (active in clusters 1 & 2,
# background in cluster 3) vs. globally active features and noise features.
# Computes standardized Kent EIG ∈ [0, 1) and local EIG.
# Reproduces Table 2 and Figure: local_vs_global.png.
# ==============================================================================

include(joinpath(@__DIR__, "common.jl"))

function run_shared_vs_local(; n_rep::Int = DEFAULT_N_REP)
    @info "Starting Simulation 3: Shared vs. Local Feature Saliency (n_rep = $n_rep)"

    n, K0, K_fit = 240, 3, 8
    p_act, p_noise = 16, 64
    u0 = 0.01

    # Indices: in model_type = 2, within active features:
    # Odd j: globally active (8 features: 2 Gauss, 2 Poisson, 2 Mult, 2 Gamma)
    # Even j: locally active (8 features: 2 Gauss, 2 Poisson, 2 Mult, 2 Gamma)
    # Noise: remaining 64 features
    # Let's track them explicitly from the generator layout
    # act_counts = [4, 4, 4, 4], noise_counts = [16, 16, 16, 16]
    # In each block: 4 active, 16 noise.
    # Among the 4 active: j=1,3 are global; j=2,4 are local.
    global_indices = Int[]
    local_indices  = Int[]
    noise_indices  = Int[]

    offset = 0
    for block in 1:4
        # Active in block
        for j in 1:4
            feat_idx = offset + j
            if j % 2 == 1
                push!(global_indices, feat_idx)
            else
                push!(local_indices, feat_idx)
            end
        end
        # Noise in block
        for j in 1:16
            push!(noise_indices, offset + 4 + j)
        end
        offset += 20
    end

    sfrm_global_eig = zeros(Float64, n_rep)
    sfrm_local_eig  = zeros(Float64, n_rep)
    sfrm_noise_eig  = zeros(Float64, n_rep)

    lfrm_global_eig = zeros(Float64, n_rep)
    lfrm_local_eig  = zeros(Float64, n_rep)
    lfrm_noise_eig  = zeros(Float64, n_rep)

    Threads.@threads for rep in 1:n_rep
        seed = GLOBAL_SEED + rep
        data, true_z, active_mask = generate_synthetic_dataset(
            n, p_act, p_noise, K0; seed = seed, model_type = 2
        )

        # 1. Fit SFRM
        res_sfrm = mixClust(data, K_fit; model_setting = SFRM(),
                            u0 = u0, compact = false, max_iter = 500, tol = 1e-4)
        eig_sfrm = compute_eig(res_sfrm.margins, res_sfrm.w, res_sfrm.pip)

        sfrm_global_eig[rep] = mean(eig_sfrm[global_indices])
        sfrm_local_eig[rep]  = mean(eig_sfrm[local_indices])
        sfrm_noise_eig[rep]  = mean(eig_sfrm[noise_indices])

        # 2. Fit LFRM
        res_lfrm = mixClust(data, K_fit; model_setting = LFRM(),
                            u0 = u0, compact = false, max_iter = 500, tol = 1e-4)
        eig_lfrm = compute_eig(res_lfrm.margins, res_lfrm.w, res_lfrm.pip)

        lfrm_global_eig[rep] = mean(eig_lfrm[global_indices])
        lfrm_local_eig[rep]  = mean(eig_lfrm[local_indices])
        lfrm_noise_eig[rep]  = mean(eig_lfrm[noise_indices])
    end

    results = (;
        sfrm = (;
            global_mean = mean(sfrm_global_eig), global_std = std(sfrm_global_eig),
            local_mean  = mean(sfrm_local_eig),  local_std  = std(sfrm_local_eig),
            noise_mean  = mean(sfrm_noise_eig),  noise_std  = std(sfrm_noise_eig)
        ),
        lfrm = (;
            global_mean = mean(lfrm_global_eig), global_std = std(lfrm_global_eig),
            local_mean  = mean(lfrm_local_eig),  local_std  = std(lfrm_local_eig),
            noise_mean  = mean(lfrm_noise_eig),  noise_std  = std(lfrm_noise_eig)
        )
    )

    # Print LaTeX Table 2
    println("\n=== LaTeX Table 2 Output: Standardized Kent EIG by Feature Class ===")
    println("\\begin{tabular}{lccc}")
    println("\\toprule")
    println("Model Setting & Globally Active & Locally Active & Noise \\\\")
    println("\\midrule")
    @printf("\\textbf{SFRM}  & \$%5.3f \\pm %5.3f\$ & \$%5.3f \\pm %5.3f\$ & \$%5.3f \\pm %5.3f\$ \\\\\n",
            results.sfrm.global_mean, results.sfrm.global_std,
            results.sfrm.local_mean,  results.sfrm.local_std,
            results.sfrm.noise_mean,  results.sfrm.noise_std)
    @printf("\\textbf{LFRM}  & \$%5.3f \\pm %5.3f\$ & \$%5.3f \\pm %5.3f\$ & \$%5.3f \\pm %5.3f\$ \\\\\n",
            results.lfrm.global_mean, results.lfrm.global_std,
            results.lfrm.local_mean,  results.lfrm.local_std,
            results.lfrm.noise_mean,  results.lfrm.noise_std)
    println("\\bottomrule")
    println("\\end{tabular}\n")

    # Barplot for Figure: Side-by-side grouped bars with two distinct bars per category
    categories = ["Globally Active", "Locally Active", "Noise"]
    sfrm_vals  = [results.sfrm.global_mean, results.sfrm.local_mean, results.sfrm.noise_mean]
    lfrm_vals  = [results.lfrm.global_mean, results.lfrm.local_mean, results.lfrm.noise_mean]
    sfrm_errs  = [results.sfrm.global_std,  results.sfrm.local_std,  results.sfrm.noise_std]
    lfrm_errs  = [results.lfrm.global_std,  results.lfrm.local_std,  results.lfrm.noise_std]

    x_base = [1.0, 2.0, 3.0]
    bar_w = 0.28

    p = bar(x_base .- bar_w/2, sfrm_vals,
            yerror = sfrm_errs,
            bar_width = bar_w,
            label = "SFRM (Shared)",
            color = :steelblue,
            ylabel = "Mean Expected Hellinger Distance (EHD)",
            title = "Feature Saliency: SFRM vs. LFRM (n = 240, K₀ = 3)",
            xticks = (x_base, categories),
            legend = :topright,
            ylims = (0.0, 0.35),
            grid = true)

    bar!(p, x_base .+ bar_w/2, lfrm_vals,
         yerror = lfrm_errs,
         bar_width = bar_w,
         label = "LFRM (Cluster-Specific)",
         color = :darkorange)

    savefig(p, joinpath(FIGURES_DIR, "local_vs_global.png"))

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_shared_vs_local()
end
