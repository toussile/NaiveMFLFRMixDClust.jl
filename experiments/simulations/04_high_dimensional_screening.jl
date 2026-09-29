# ==============================================================================
# 04_high_dimensional_screening.jl — High-Dimensional Feature Screening Behavior
# ==============================================================================
# Evaluates descriptive TPR and empirical FDR across dimensions J ∈ {40, 100, 200, 400}
# under two intuitive screening thresholds:
# 1. Median Probability Model rule (PIP γ̄ⱼ ≥ 0.5)
# 2. Heuristic Kent EIG thresholding (EIGⱼ ≥ 0.20)
#
# Also benchmarks ARI against an unpenalized mixture (All-features) to highlight
# resilience against the accumulation of high-dimensional noise.
# Reproduces Table 3 and Figures: fdr_control.png & tpr_control.png.
# ==============================================================================

include(joinpath(@__DIR__, "common.jl"))

function run_high_dimensional_screening(; n_rep::Int = DEFAULT_N_REP)
    @info "Starting Simulation 4: High-Dimensional Feature Screening (n_rep = $n_rep)"

    dimensions = [40, 100, 200, 400]
    n, K0, K_fit = 150, 3, 6
    p_act = 16
    u0 = 0.01

    results = []

    for J in dimensions
        p_noise = J - p_act

        pip_tpr_vals = zeros(Float64, n_rep)
        pip_fdr_vals = zeros(Float64, n_rep)
        eig_tpr_vals = zeros(Float64, n_rep)
        eig_fdr_vals = zeros(Float64, n_rep)

        ari_vi_vals   = zeros(Float64, n_rep)
        ari_nofs_vals = zeros(Float64, n_rep)

        Threads.@threads for rep in 1:n_rep
            seed = GLOBAL_SEED + rep
            data, true_z, active_mask = generate_synthetic_dataset(
                n, p_act, p_noise, K0; seed = seed, model_type = 1
            )

            # 1. Proposed MixClustVI (SFRM with feature selection)
            res = mixClust(data, K_fit; model_setting = SFRM(),
                           u0 = u0, compact = false, max_iter = 500, tol = 1e-4)

            ari_vi_vals[rep] = adjusted_rand_index(true_z, res.labels)

            # Mean PIP for each feature
            pip_j = vec(mean(res.pip, dims = 1))
            # Standardized Kent EIG
            eig_j = compute_eig(res.margins, res.w, res.pip)

            # Rule 1: Median Probability Model (PIP ≥ 0.5)
            sel_pip = pip_j .>= 0.5
            m_pip = compute_tpr_fdr(active_mask, sel_pip)
            pip_tpr_vals[rep] = m_pip.tpr
            pip_fdr_vals[rep] = m_pip.fdr

            # Rule 2: Heuristic Kent EIG filter (EIG ≥ 0.20)
            sel_eig = eig_j .>= 0.20
            m_eig = compute_tpr_fdr(active_mask, sel_eig)
            eig_tpr_vals[rep] = m_eig.tpr
            eig_fdr_vals[rep] = m_eig.fdr

            # 2. Baseline without feature selection (All-features forced active)
            res_nofs = mixClust(data, K_fit; model_setting = SFRM(),
                                u0 = u0, delta_prior = (100.0, 1.0),
                                compact = false, max_iter = 500, tol = 1e-4)
            ari_nofs_vals[rep] = adjusted_rand_index(true_z, res_nofs.labels)
        end

        push!(results, (;
            J,
            pip_tpr = mean(pip_tpr_vals),
            pip_fdr = mean(pip_fdr_vals),
            eig_tpr = mean(eig_tpr_vals),
            eig_fdr = mean(eig_fdr_vals),
            ari_vi   = mean(ari_vi_vals),
            ari_nofs = mean(ari_nofs_vals)
        ))

        @printf("J = %3d | PIP TPR = %5.3f, FDR = %5.3f | EIG TPR = %5.3f, FDR = %5.3f | ARI (VI) = %5.3f vs (NoFS) = %5.3f\n",
                J, mean(pip_tpr_vals), mean(pip_fdr_vals),
                mean(eig_tpr_vals), mean(eig_fdr_vals),
                mean(ari_vi_vals), mean(ari_nofs_vals))
    end

    # Figures
    j_vals   = [r.J for r in results]
    pip_fdr_m = [r.pip_fdr for r in results]
    eig_fdr_m = [r.eig_fdr for r in results]
    pip_tpr_m = [r.pip_tpr for r in results]
    eig_tpr_m = [r.eig_tpr for r in results]

    # Figure 1: Empirical FDR vs Dimension J
    p1 = plot(j_vals, pip_fdr_m;
              marker = :circle, lw = 2, ms = 6, color = :royalblue,
              label = "PIP selection (γ̄ⱼ ≥ 0.5)",
              xlabel = "Total feature dimension J",
              ylabel = "Empirical False Discovery Proportion",
              title = "Empirical False Discovery Rate across Dimensions",
              legend = :topleft, ylims = (-0.01, 0.25), grid = true)
    plot!(p1, j_vals, eig_fdr_m;
          marker = :square, lw = 2, ms = 6, color = :forestgreen,
          label = "Kent EIG filter (EIGⱼ ≥ 0.20)")
    savefig(p1, joinpath(FIGURES_DIR, "fdr_control.png"))

    # Figure 2: Sensitivity (TPR) vs Dimension J
    p2 = plot(j_vals, pip_tpr_m;
              marker = :circle, lw = 2, ms = 6, color = :royalblue,
              label = "PIP selection (γ̄ⱼ ≥ 0.5)",
              xlabel = "Total feature dimension J",
              ylabel = "Detection Sensitivity (TPR)",
              title = "Signal Detection Sensitivity across Dimensions",
              legend = :bottomleft, ylims = (0.50, 1.05), grid = true)
    plot!(p2, j_vals, eig_tpr_m;
          marker = :square, lw = 2, ms = 6, color = :forestgreen,
          label = "Kent EIG filter (EIGⱼ ≥ 0.20)")
    savefig(p2, joinpath(FIGURES_DIR, "tpr_control.png"))

    # Print LaTeX Table 3
    println("\n=== LaTeX Table 3 Output: Empirical Screening Diagnostics ===")
    println("\\begin{tabular}{ccccc}")
    println("\\toprule")
    println("Dimension \$J\$ & PIP TPR & PIP Empirical FDR & EIG TPR & EIG Empirical FDR \\\\")
    println("\\midrule")
    for r in results
        @printf("%3d & %5.3f & %5.3f & %5.3f & %5.3f \\\\\n",
                r.J, r.pip_tpr, r.pip_fdr, r.eig_tpr, r.eig_fdr)
    end
    println("\\bottomrule")
    println("\\end{tabular}\n")

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_high_dimensional_screening()
end
