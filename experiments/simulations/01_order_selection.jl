# ==============================================================================
# 01_order_selection.jl — In-Model Cluster Order Selection via Overfitted Mixtures
# ==============================================================================
# Evaluates in-model cluster-emptying under the sparse symmetric Dirichlet prior
# u⁽⁰⁾ ∈ {0.001, 0.01, 0.1, 0.5, 1.0, 2.0} with K_fit = 12 and True K_0 = 3.
# Reproduces Table 1 and Figures: order_selection_k.png & order_selection_ari.png.
# Also benchmarks runtime against standard BIC grid search K ∈ 1:6.
# ==============================================================================

include(joinpath(@__DIR__, "common.jl"))

function run_order_selection(; n_rep::Int = DEFAULT_N_REP)
    @info "Starting Simulation 1: In-Model Cluster Order Selection (n_rep = $n_rep)"

    u0_grid = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
    n, K0, K_fit = 140, 3, 12
    p_act, p_noise = 16, 64

    results = []

    for u0 in u0_grid
        k_hat_vals = zeros(Int, n_rep)
        ari_vals   = zeros(Float64, n_rep)
        time_vals  = zeros(Float64, n_rep)

        Threads.@threads for rep in 1:n_rep
            seed = GLOBAL_SEED + rep
            data, true_z, _ = generate_synthetic_dataset(
                n, p_act, p_noise, K0; seed = seed, model_type = 1
            )

            t_start = time()
            # In-model order selection without heuristic post-hoc pruning
            res = mixClust(data, K_fit; model_setting = SFRM(),
                           u0 = u0, compact = false, max_iter = 500, tol = 1e-4)
            t_elapsed = time() - t_start

            k_hat_vals[rep] = map_cluster_order(res)
            ari_vals[rep]   = adjusted_rand_index(true_z, res.labels)
            time_vals[rep]  = t_elapsed
        end

        push!(results, (;
            u0,
            mean_k    = mean(k_hat_vals),
            std_k     = std(k_hat_vals),
            mean_ari  = mean(ari_vals),
            std_ari   = std(ari_vals),
            mean_time = mean(time_vals),
            raw_k     = k_hat_vals,
            raw_ari   = ari_vals
        ))

        @printf("u0 = %5.3f | K̂ = %4.2f ± %4.2f | ARI = %5.3f ± %5.3f | Avg Time/rep = %4.2fs\n",
                u0, mean(k_hat_vals), std(k_hat_vals), mean(ari_vals), std(ari_vals), mean(time_vals))
        flush(stdout)
    end


    # Generate Figures
    u0_vals  = [r.u0 for r in results]
    k_means  = [r.mean_k for r in results]
    k_stds   = [r.std_k for r in results]
    ari_means = [r.mean_ari for r in results]
    ari_stds  = [r.std_ari for r in results]

    # Figure 1: K̂ vs u⁽⁰⁾
    p1 = plot(u0_vals, k_means; yerror = k_stds,
              xscale = :log10, marker = :circle, lw = 2, ms = 5,
              color = :navy, label = "Estimated K̂ (MAP)",
              xlabel = "Dirichlet hyperparameter u⁽⁰⁾ (log scale)",
              ylabel = "Estimated number of clusters K̂",
              title = "In-Model Order Selection (K_fit = 12, True K₀ = 3)",
              legend = :topleft, ylims = (2.0, 5.0), grid = true)
    hline!(p1, [K0]; label = "True K₀ = 3", ls = :dash, lw = 2, color = :crimson)
    savefig(p1, joinpath(FIGURES_DIR, "order_selection_k.png"))

    # Figure 2: ARI vs u⁽⁰⁾
    p2 = plot(u0_vals, ari_means; yerror = ari_stds,
              xscale = :log10, marker = :square, lw = 2, ms = 5,
              color = :darkgreen, label = "Clustering Accuracy (ARI)",
              xlabel = "Dirichlet hyperparameter u⁽⁰⁾ (log scale)",
              ylabel = "Adjusted Rand Index (ARI)",
              title = "Partition Recovery vs. Prior Sparsity",
              legend = :bottomleft, ylims = (0.90, 1.01), grid = true)
    savefig(p2, joinpath(FIGURES_DIR, "order_selection_ari.png"))

    # Print LaTeX Table
    println("\n=== LaTeX Table 1 Output ===")
    println("\\begin{tabular}{ccc}")
    println("\\toprule")
    println("\$u^{(0)}\$ & Estimated Order \$\\widehat{K}\$ & Partition Accuracy (ARI) \\\\")
    println("\\midrule")
    for r in results
        @printf("%.3f & \$%3.1f \\pm %4.2f\$ & \$%5.3f \\pm %5.3f\$ \\\\\n",
                r.u0, r.mean_k, r.std_k, r.mean_ari, r.std_ari)
    end
    println("\\bottomrule")
    println("\\end{tabular}\n")

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_order_selection()
end
