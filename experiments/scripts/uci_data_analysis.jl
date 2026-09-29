# ==============================================================================
# uci_data_analysis.jl — UCI Heart Disease Benchmark Analysis
# ==============================================================================
# Reproduces Section 6.2 of the manuscript:
#   - LFRM clustering on 297 UCI patients with 13 mixed-type features
#   - Feature saliency via PIP and EHD
#   - Outlier detection via coordinate inactivation rate
#   - Comparison with binary angiographic disease label (ARI)
# ==============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
using NaiveMFLFRMixDClust
using CSV, DataFrames, Statistics, Random, Printf, Plots

const FIGURES_DIR = joinpath(@__DIR__, "../figures")
mkpath(FIGURES_DIR)

const C_BLUE  = "#2563EB"
const C_GREEN = "#10B981"
const C_RED   = "#EF4444"
const C_GREY  = "#9CA3AF"

# ── One-hot encoder ───────────────────────────────────────────────────────────
function onehot(val, levels)
    idx = findfirst(==(val), levels)
    isnothing(idx) && error("Value $val not found in levels $levels")
    v = zeros(Int, length(levels))
    v[idx] = 1
    return v
end

# ──────────────────────────────────────────────────────────────────────────────
function main()
    println("=" ^ 60)
    println("  UCI Heart Disease — SFRM Analysis with Outlier Detection")
    println("=" ^ 60)

    # ── Load and preprocess dataset ───────────────────────────────────────────
    # load_heart_disease() reads the bundled CSV, standardises continuous
    # features, and one-hot-encodes categoricals — no extra preprocessing needed.
    hd            = load_heart_disease()
    dataset       = hd.data
    feature_names = hd.feature_names
    n             = length(dataset[1])
    p             = length(dataset)
    println("  Loaded $n patients, $p features")

    # Ground-truth: binary disease label (0 = absent → 1, 1+ = present → 2)
    labels_true = [x == 0 ? 1 : 2 for x in hd.labels]

    K_fit   = 10
    u0      = 0.01
    tau_out = 0.55

    # ── Fit LFRM ─────────────────────────────────────────────────────────────
    println("\nFitting LFRM on UCI dataset (n=$n, K_fit=$K_fit)…")
    Random.seed!(42)
    res = mixClust(dataset, K_fit;
                   model_setting = LFRM(),
                   u0 = u0, max_iter = 2000, tol = 1e-5,
                   compact = true)
    K_active = n_active_clusters(res)
    println("  CAVI iterations: $(length(res.elbo_history))")
    println("  Active clusters K̂: $K_active")
    println("  Final ELBO: $(round(last(res.elbo_history), digits=2))")

    # ── Cluster sizes ─────────────────────────────────────────────────────────
    for k in 1:K_active
        idx_k = cluster_indices(res, k)
        @printf("    Cluster %d: n=%d (%.1f%%)\n", k, length(idx_k), 100*length(idx_k)/n)
    end

    # ── Outlier detection ─────────────────────────────────────────────────────
    rho         = coordinate_inactivation_rate(res)
    out_idx     = outlier_indices(res; threshold = tau_out)
    n_outliers  = length(out_idx)
    z_robust    = robust_cluster_assignments(res; threshold = tau_out)
    pi0         = compute_pi_0(res)

    ari_std     = adjusted_rand_index(labels_true, res.labels)
    ari_robust  = adjusted_rand_index(labels_true, z_robust)

    println("\n── Outlier Detection (τ=$tau_out) ───────────────────────────────")
    println("  Flagged as outliers: $n_outliers / $n")
    if n_outliers > 0 && n_outliers <= 20
        println("  Outlier indices:     $out_idx")
    elseif n_outliers > 20
        println("  Outlier indices:     $(out_idx[1:10])  … ($n_outliers total)")
    end
    println("  π₀ (background mass): $(round(pi0, digits=4))")
    println("  ρ̄ (mean CIR):          $(round(mean(rho), digits=4))")
    println("  ρ max:                 $(round(maximum(rho), digits=4))")

    # ── Partition accuracy ────────────────────────────────────────────────────
    println("\n── Partition Accuracy ─────────────────────────────────────────")
    println("  Standard MAP ARI vs. disease label: $(round(ari_std,    digits=4))")
    println("  Robust MAP ARI vs. disease label:   $(round(ari_robust, digits=4))")

    # ── Feature PIPs & EHD ───────────────────────────────────────────────────
    # In LFRM pip is N×J; mean over observations gives a J-vector
    pips = vec(mean(res.pip, dims = 1))
    ehd  = compute_ehd(res)

    println("\n── Feature PIPs & EHD (LFRM posterior mean) ───────────────────")
    println("  $(rpad("Feature", 12)) | PIP    | EHD    | Selected (≥0.5)")
    println("  " * "─" ^ 48)
    for j in 1:p
        sel = pips[j] >= 0.5 ? "✓" : "✗"
        @printf("  %-12s | %.4f | %.4f | %s\n", feature_names[j], pips[j], ehd[j], sel)
    end

    # ── Figures ───────────────────────────────────────────────────────────────
    function savefig_named(plt, name)
        path = joinpath(FIGURES_DIR, "real_data_$(name)_uci.png")
        savefig(plt, path)
        println("  Saved: $(basename(path))")
    end

    # PIP bar chart
    pip_colors = [pips[j] >= 0.5 ? C_GREEN : C_GREY for j in 1:p]
    p_pips = bar(1:p, pips;
                 color = pip_colors, linecolor = :match, legend = false,
                 xlabel = "Features", ylabel = "Mean PIP",
                 title = "Posterior Inclusion Probabilities — UCI Heart Disease",
                 xticks = (1:p, feature_names), xrotation = 45,
                 ylim = (0, 1.05),
                 dpi = 300, framestyle = :box, size = (780, 420), margin = 8Plots.mm)
    hline!(p_pips, [0.5]; line = (2, :dash, C_RED), label = "threshold 0.5")
    savefig_named(p_pips, "pips")

    # Cluster assignment scatter (thalach vs age, standardised)
    scatter_x = dataset[8]   # thalach (standardised)
    scatter_y = dataset[1]   # age (standardised)

    # Build colours: outliers in red, clusters in palette
    palette_cols = [:royalblue, :mediumseagreen, :darkorange, :purple, :sienna]
    pt_colors = [z_robust[i] == 0 ? C_RED :
                 palette_cols[mod1(z_robust[i], length(palette_cols))]
                 for i in 1:n]
    pt_shapes = [z_robust[i] == 0 ? :x : :circle for i in 1:n]
    pt_sizes  = [z_robust[i] == 0 ? 6 : 4 for i in 1:n]

    p_assign = scatter(scatter_x, scatter_y;
                       color = pt_colors, markershape = pt_shapes,
                       markersize = pt_sizes, markerstrokewidth = 0.5,
                       xlabel = "Max Heart Rate (standardised)",
                       ylabel = "Age (standardised)",
                       title = "Cluster Assignments (K̂=$K_active, outliers flagged ×)",
                       legend = false,
                       dpi = 300, framestyle = :box, size = (680, 460),
                       margin = 6Plots.mm)
    savefig_named(p_assign, "assignments")

    # CIR distribution plot
    p_rho = histogram(rho;
                      bins = 30, color = C_BLUE, linecolor = :white, alpha = 0.8,
                      xlabel = "Coordinate Inactivation Rate ρ̄ᵢ",
                      ylabel = "Count",
                      title = "Outlier Detection — UCI Heart Disease",
                      label = "patients", legend = :topright,
                      dpi = 300, framestyle = :box, size = (640, 380), margin = 6Plots.mm)
    vline!(p_rho, [tau_out]; line = (2, :dash, C_RED), label = "τ = $tau_out")
    if n_outliers > 0
        scatter!(p_rho, rho[out_idx], zeros(n_outliers) .+ 0.3;
                 color = C_RED, markershape = :dtriangle, markersize = 6,
                 label = "$n_outliers flagged")
    end
    savefig_named(p_rho, "outliers")

    println("\n" * "=" ^ 60)
    println("  Analysis complete. Figures saved to: $FIGURES_DIR")
    println("=" ^ 60)

    return (
        K_active    = K_active,
        ari_std     = ari_std,
        ari_robust  = ari_robust,
        n_outliers  = n_outliers,
        pi0         = pi0,
        rho_mean    = mean(rho),
        rho_max     = maximum(rho),
        pips        = pips,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
