# ==============================================================================
# 01_order_selection.jl — In-Model Cluster Order Selection via Overfitted Mixtures
# ==============================================================================
# Evaluates in-model cluster-emptying under the default sparse symmetric Dirichlet prior
# ω ~ Dir(1/K_fit 1_K) across initial overfitted specifications
# K_fit ∈ {6, 8, 10, 12, 16, 20} with True K_0 = 3.
# Reproduces Table S1 and Figures: order_selection_k.png & order_selection_ari.png.
# ==============================================================================

include(joinpath(@__DIR__, "common.jl"))

function run_order_selection(; n_rep::Int = DEFAULT_N_REP)
    @info "Starting Simulation 1: In-Model Cluster Order Selection (n_rep = $n_rep)"

    n, K0 = 140, 3
    K_fit_grid = [6, 8, 10, 12, 16, 20]
    p_act, p_noise = 16, 64

    results = []

    for K_fit in K_fit_grid
        k_raw_vals   = zeros(Int, n_rep)
        k_eff_vals   = zeros(Int, n_rep)
        ari_raw_vals = zeros(Float64, n_rep)
        ari_eff_vals = zeros(Float64, n_rep)
        time_vals    = zeros(Float64, n_rep)
        u0 = 1.0 / K_fit

        Threads.@threads for rep in 1:n_rep
            seed = GLOBAL_SEED + rep
            data, true_z, _ = generate_synthetic_dataset(
                n, p_act, p_noise, K0; seed = seed, model_type = 1
            )

            t_start = time()
            res = mixClust(data, K_fit; model_setting = SFRM(),
                           u0 = u0, compact = false, max_iter = 500, tol = 1e-4)
            t_elapsed = time() - t_start

            k_raw_vals[rep] = map_cluster_order(res)
            ari_raw_vals[rep] = adjusted_rand_index(true_z, res.labels)

            pop_counts = [count(==(k), res.labels) for k in 1:K_fit]
            k_eff_vals[rep] = count(c -> c >= 0.02 * n, pop_counts)
            ari_eff_vals[rep] = adjusted_rand_index(true_z, res.labels)
            time_vals[rep] = t_elapsed
        end

        push!(results, (;
            K_fit,
            u0,
            mean_k_raw   = mean(k_raw_vals),
            std_k_raw    = std(k_raw_vals),
            mean_k_eff   = mean(k_eff_vals),
            std_k_eff    = std(k_eff_vals),
            mean_ari_raw = mean(ari_raw_vals),
            std_ari_raw  = std(ari_raw_vals),
            mean_ari_eff = mean(ari_eff_vals),
            std_ari_eff  = std(ari_eff_vals),
            mean_time    = mean(time_vals)
        ))

        @printf("K_fit = %2d (u0 = %5.3f) | K̂_raw = %4.2f ± %4.2f | K̂_eff = %4.2f ± %4.2f | ARI = %5.3f ± %5.3f | Avg Time = %4.2fs\n",
                K_fit, u0, mean(k_raw_vals), std(k_raw_vals), mean(k_eff_vals), std(k_eff_vals), mean(ari_raw_vals), std(ari_raw_vals), mean(time_vals))
        flush(stdout)
    end

    # Generate Figures
    k_fits    = [r.K_fit for r in results]
    k_raw_m   = [r.mean_k_raw for r in results]
    k_raw_s   = [r.std_k_raw for r in results]
    k_eff_m   = [r.mean_k_eff for r in results]
    ari_raw_m = [r.mean_ari_raw for r in results]
    ari_raw_s = [r.std_ari_raw for r in results]

    # Figure 1: K̂ vs K_fit
    p1 = plot(k_fits, k_raw_m; yerror = k_raw_s,
              marker = :circle, lw = 2, ms = 5,
              color = :navy, label = "Estimated K̂ (Raw MAP)",
              xlabel = "Initial Overfitted Components K_fit",
              ylabel = "Estimated Number of Clusters K̂",
              title = "Cluster Order Recovery across Overfitting Levels (True K₀ = 3)",
              legend = :topleft, ylims = (2.0, 6.0), grid = true)
    plot!(p1, k_fits, k_eff_m; marker = :diamond, lw = 2, ms = 5,
          color = :purple, label = "Effective K̂ (≥ 2% size)")
    hline!(p1, [K0]; label = "True K₀ = 3", ls = :dash, lw = 2, color = :crimson)
    savefig(p1, joinpath(FIGURES_DIR, "order_selection_k.png"))

    # Figure 2: ARI vs K_fit
    p2 = plot(k_fits, ari_raw_m; yerror = ari_raw_s,
              marker = :square, lw = 2, ms = 5,
              color = :darkgreen, label = "Clustering Accuracy (ARI)",
              xlabel = "Initial Overfitted Components K_fit",
              ylabel = "Adjusted Rand Index (ARI)",
              title = "Partition Recovery vs. Overfitting Level K_fit",
              legend = :bottomleft, ylims = (0.95, 1.005), grid = true)
    savefig(p2, joinpath(FIGURES_DIR, "order_selection_ari.png"))

    # Print LaTeX Table
    println("\n=== LaTeX Table S1 Output ===")
    println("\\begin{tabular}{cccccc}")
    println("\\toprule")
    println("\$K_{\\text{fit}}\$ & \$u_k^{(0)} = 1/K_{\\text{fit}}\$ & \$\\widehat{K}\$ (Raw MAP) & \$\\widehat{K}_{\\text{eff}}\$ (\$\\ge 2\\%\$) & ARI (Raw) & ARI (Effective) \\\\")
    println("\\midrule")
    for r in results
        @printf("%d & %.3f & \$%.2f \\pm %.2f\$ & \$%.2f \\pm %.2f\$ & \$%.4f \\pm %.4f\$ & \$%.4f \\pm %.4f\$ \\\\\n",
                r.K_fit, r.u0, r.mean_k_raw, r.std_k_raw, r.mean_k_eff, r.std_k_eff, r.mean_ari_raw, r.std_ari_raw, r.mean_ari_eff, r.std_ari_eff)
    end
    println("\\bottomrule")
    println("\\end{tabular}\n")

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_order_selection()
end
