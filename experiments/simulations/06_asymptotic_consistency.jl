# ==============================================================================
# 06_asymptotic_consistency.jl — Asymptotic Consistency and Modular Robust Assignment
# ==============================================================================
# Evaluates in-model cluster contraction and partition recovery across sample sizes:
#   n ∈ {50, 80, 120, 200, 250, 300, 400}
# with u⁽⁰⁾ = 0.001, K_fit = 12, true K₀ = 3, p_act = 16, p_noise = 64.
# Contrasts Standard MAP classification vs. Modular Robust Assignment (outlier rejection).
# Reproduces Table S6.2 and Figures: consistency_k.png & consistency_ari.png.
# ==============================================================================

include(joinpath(@__DIR__, "common.jl"))

function run_asymptotic_consistency(; n_rep::Int = DEFAULT_N_REP)
    @info "Starting Simulation: Asymptotic Consistency & Robust Assignment (n_rep = $n_rep)"

    n_grid = [50, 80, 120, 200, 250, 300, 400]
    K0, K_fit = 3, 12
    p_act, p_noise = 16, 64
    u0 = 0.001

    results = []

    for n in n_grid
        k_std_vals  = zeros(Int, n_rep)
        ari_std_vals = zeros(Float64, n_rep)
        k_rob_vals  = zeros(Int, n_rep)
        ari_rob_vals = zeros(Float64, n_rep)
        time_vals   = zeros(Float64, n_rep)

        Threads.@threads for rep in 1:n_rep
            seed = GLOBAL_SEED + 1000 * rep + n
            data, true_z, _ = generate_synthetic_dataset(
                n, p_act, p_noise, K0; seed = seed, model_type = 1
            )

            t_start = time()
            # Standard single-run CAVI with in-model order selection
            res = mixClust(data, K_fit; model_setting = SFRM(),
                           u0 = u0, compact = false, max_iter = 500, tol = 1e-4)
            t_elapsed = time() - t_start

            # 1. Standard MAP Hard Assignment
            k_std_vals[rep]  = map_cluster_order(res)
            ari_std_vals[rep] = adjusted_rand_index(true_z, res.labels)

            # 2. Modular Post-Hoc Robust Assignment (via effective responsibility >= 2%)
            w = res.w
            eff_clusters = findall(k -> sum(w[:, k]) / n >= 0.02, 1:size(w, 2))
            k_rob_vals[rep] = length(eff_clusters)
            z_rob = zeros(Int, n)
            for i in 1:n
                z_rob[i] = eff_clusters[argmax(w[i, eff_clusters])]
            end
            ari_rob_vals[rep] = adjusted_rand_index(true_z, z_rob)

            time_vals[rep]   = t_elapsed
        end

        push!(results, (;
            n,
            mean_k_std   = mean(k_std_vals),
            std_k_std    = std(k_std_vals),
            mean_ari_std = mean(ari_std_vals),
            std_ari_std  = std(ari_std_vals),
            mean_k_rob   = mean(k_rob_vals),
            std_k_rob    = std(k_rob_vals),
            mean_ari_rob = mean(ari_rob_vals),
            std_ari_rob  = std(ari_rob_vals),
            mean_time    = mean(time_vals)
        ))

        @printf("n = %3d | Std: K̂ = %4.2f ± %4.2f, ARI = %5.3f ± %5.3f | Rob: K̂ = %4.2f ± %4.2f, ARI = %5.3f ± %5.3f | Time = %4.2fs\n",
                n, mean(k_std_vals), std(k_std_vals), mean(ari_std_vals), std(ari_std_vals),
                mean(k_rob_vals), std(k_rob_vals), mean(ari_rob_vals), std(ari_rob_vals), mean(time_vals))
        flush(stdout)
    end

    # Generate Figures
    n_vals      = [r.n for r in results]
    k_std_means = [r.mean_k_std for r in results]
    k_std_stds  = [r.std_k_std for r in results]
    k_rob_means = [r.mean_k_rob for r in results]
    k_rob_stds  = [r.std_k_rob for r in results]

    ari_std_means = [r.mean_ari_std for r in results]
    ari_std_stds  = [r.std_ari_std for r in results]
    ari_rob_means = [r.mean_ari_rob for r in results]
    ari_rob_stds  = [r.std_ari_rob for r in results]

    # Figure 1: Estimated Order K̂ vs n
    p1 = plot(n_vals, k_std_means; yerror = k_std_stds,
              marker = :circle, lw = 2, ms = 5,
              color = :navy, label = "Standard MAP (K̂)",
              xlabel = "Sample size n",
              ylabel = "Estimated number of clusters K̂",
              title = "Asymptotic Order Selection (K_fit = 12, True K₀ = 3)",
              legend = :topright, ylims = (2.0, 5.0), grid = true)
    plot!(p1, n_vals, k_rob_means; yerror = k_rob_stds,
          marker = :diamond, lw = 2, ms = 5, ls = :dash,
          color = :darkcyan, label = "Robust Assignment (K̂)")
    hline!(p1, [K0]; label = "True K₀ = 3", ls = :dot, lw = 2, color = :crimson)
    savefig(p1, joinpath(FIGURES_DIR, "consistency_k.png"))

    # Figure 2: ARI vs n
    p2 = plot(n_vals, ari_std_means; yerror = ari_std_stds,
              marker = :square, lw = 2, ms = 5,
              color = :darkgreen, label = "Standard MAP (ARI)",
              xlabel = "Sample size n",
              ylabel = "Adjusted Rand Index (ARI)",
              title = "Asymptotic Partition Accuracy",
              legend = :bottomright, ylims = (0.85, 1.01), grid = true)
    plot!(p2, n_vals, ari_rob_means; yerror = ari_rob_stds,
          marker = :star5, lw = 2, ms = 5, ls = :dash,
          color = :teal, label = "Robust Assignment (ARI)")
    savefig(p2, joinpath(FIGURES_DIR, "consistency_ari.png"))

    # Print LaTeX Table
    println("\n=== LaTeX Table S6.2 Output ===")
    println("\\begin{tabular}{ccccc}")
    println("\\toprule")
    println("Sample Size \$n\$ & \$\\widehat{K}\$ (Standard MAP) & ARI (Standard MAP) & \$\\widehat{K}\$ (Robust) & ARI (Robust) \\\\")
    println("\\midrule")
    for r in results
        @printf("%3d & \$%3.1f \\pm %4.2f\$ & \$%5.3f \\pm %5.3f\$ & \$%3.1f \\pm %4.2f\$ & \$%5.3f \\pm %5.3f\$ \\\\\n",
                r.n, r.mean_k_std, r.std_k_std, r.mean_ari_std, r.std_ari_std,
                r.mean_k_rob, r.std_k_rob, r.mean_ari_rob, r.std_ari_rob)
    end
    println("\\bottomrule")
    println("\\end{tabular}\n")

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_asymptotic_consistency()
end
