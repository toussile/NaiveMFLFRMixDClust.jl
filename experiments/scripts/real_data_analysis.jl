# ==============================================================================
# real_data_analysis.jl — Synthetic Clinicogenomic Cohort Analysis
# ==============================================================================
# Reproduces Section 6.1 of the manuscript using the built-in
# simulate_synthetic_cohort() from NaiveMFLFRMixDClust.jl:
#   - 7 mixed-type features (Gaussian ×2, Poisson, Gamma, Multinomial, noise ×2)
#   - 3 clinical subtypes, n = 200 patients
#   - LFRM with order selection, uninformative observations, and predictive validation
# ==============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
using NaiveMFLFRMixDClust
using Statistics, Random, Printf, Plots

const FIGURES_DIR = joinpath(@__DIR__, "../figures")
mkpath(FIGURES_DIR)

const C_BLUE  = "#2563EB"
const C_GREEN = "#10B981"
const C_RED   = "#EF4444"
const C_GREY  = "#9CA3AF"

# ──────────────────────────────────────────────────────────────────────────────
function main()
    println("=" ^ 60)
    println("  Synthetic Clinicogenomic Cohort — LFRM Analysis")
    println("=" ^ 60)

    n     = 200
    K_fit = 10
    u0    = 0.005
    tau     = 0.20    # EHD selection threshold
    tau_out = 0.55    # CIR threshold for uninformative observations

    # ── Generate dataset via built-in function ─────────────────────────────────
    cohort        = simulate_synthetic_cohort(; n = n, seed = 789)
    data          = cohort.data
    true_z        = cohort.labels
    feature_names = cohort.feature_names
    p             = length(data)
    println("  Features: $p — $(join(feature_names, ", "))")

    # ── 80/20 train/test split ─────────────────────────────────────────────────
    n_train   = round(Int, 0.8 * n)
    perm      = randperm(MersenneTwister(456), n)
    train_idx = perm[1:n_train]
    test_idx  = perm[(n_train + 1):end]

    train_data = [data[j][train_idx] for j in 1:p]
    test_data  = [data[j][test_idx]  for j in 1:p]
    train_z    = true_z[train_idx]
    test_z     = true_z[test_idx]

    # ── Fit LFRM on training set ───────────────────────────────────────────────
    println("\nFitting LFRM on training set (n_train=$n_train, K_fit=$K_fit)…")
    res_train = mixClust(train_data, K_fit;
                         model_setting = LFRM(),
                         u0 = u0, max_iter = 2000, tol = 1e-5,
                         compact = true)
    K_train = n_active_clusters(res_train)
    println("  CAVI iterations: $(length(res_train.elbo_history))")
    println("  Active clusters K̂ (train): $K_train")

    # ── Uninformative observations on training set ─────────────────────────────────────
    rho_train  = coordinate_inactivation_rate(res_train)
    out_idx_tr = uninformative_indices(res_train; threshold = tau_out)
    println("  Flagged as uninformative (τ=$tau_out): $(length(out_idx_tr)) / $n_train")

    # ── Out-of-sample predictive validation ───────────────────────────────────
    println("\n── Out-of-Sample Validation (n_test=$(length(test_idx))) ──────────")
    pred_proba_test = predict_proba(res_train, test_data)
    test_pred_z     = [argmax(pred_proba_test[i, :]) for i in 1:length(test_idx)]
    test_ari        = adjusted_rand_index(test_z, test_pred_z)
    test_ll         = predictive_log_likelihood(res_train, test_data)
    println("  Out-of-sample ARI:                   $(round(test_ari, digits=4))")
    println("  Predictive log-likelihood (mixture): $(round(test_ll,  digits=2))")

    # ── Refit LFRM on full dataset for figures ─────────────────────────────────
    println("\nRefitting on full dataset (n=$n) for publication figures…")
    res_full = mixClust(data, K_fit;
                        model_setting = LFRM(),
                        u0 = u0, max_iter = 2000, tol = 1e-5,
                        compact = true)
    K_full      = n_active_clusters(res_full)
    elbo_final  = last(res_full.elbo_history)
    println("  CAVI iterations: $(length(res_full.elbo_history))")
    println("  Active clusters K̂ (full): $K_full")
    println("  Final ELBO: $(round(elbo_final, digits=2))")

    for k in 1:K_full
        idx_k = cluster_indices(res_full, k)
        @printf("    Cluster %d: n=%d (%.1f%%)\n", k, length(idx_k), 100*length(idx_k)/n)
    end

    # ── Uninformative observations on full dataset ─────────────────────────────────────
    rho_full     = coordinate_inactivation_rate(res_full)
    out_idx_full = uninformative_indices(res_full; threshold = tau_out)
    n_uninf   = length(out_idx_full)
    z_ext     = extended_cluster_assignments(res_full; mode = :inactivation_rate, threshold = tau_out)
    pi0          = compute_pi_0(res_full)

    full_ari_std    = adjusted_rand_index(true_z, res_full.labels)
    full_ari_ext = adjusted_rand_index(true_z, z_ext)

    println("\n── Uninformative Observations (τ=$tau_out) ───────────────────────────────")
    println("  Flagged as uninformative: $n_uninf / $n")
    println("  π₀ (background mass): $(round(pi0, digits=4))")
    println("  ρ̄ (mean CIR):          $(round(mean(rho_full), digits=4))")
    println("  Standard MAP ARI:      $(round(full_ari_std,    digits=4))")
    println("  Extended MAP ARI:        $(round(full_ari_ext, digits=4))")

    # ── Feature PIPs and EHD ──────────────────────────────────────────────────
    pip_means = vec(mean(res_full.pip, dims = 1))
    ehd_full  = compute_local_ehd(res_full)
    ehd_means = ndims(ehd_full) == 1 ? ehd_full : vec(mean(ehd_full, dims = 1))

    println("\n── Feature PIPs and EHD (full, n=$n) ───────────────────────────")
    println("  $(rpad("Feature", 14)) | Mean PIP | Mean EHD | Sel")
    println("  " * "─" ^ 50)
    for j in 1:p
        sel = ehd_means[j] >= tau ? "✓" : "✗"
        @printf("  %-14s | %8.4f | %8.4f | %s\n",
                feature_names[j], pip_means[j], ehd_means[j], sel)
    end

    # ── Figures ────────────────────────────────────────────────────────────────
    function savefig_named(plt, name)
        path = joinpath(FIGURES_DIR, "real_data_$(name).png")
        savefig(plt, path)
        println("  Saved: $(basename(path))")
    end

    # ELBO
    p_elbo = plot_elbo(res_full;
                       title  = "ELBO Convergence — Clinicogenomic Cohort",
                       dpi    = 300, framestyle = :box,
                       size   = (620, 400), margin = 5Plots.mm)
    savefig_named(p_elbo, "elbo")

    # Local PIPs heatmap (N×J for LFRM)
    p_pips = plot_local_pips(res_full;
                             feature_names = feature_names,
                             title = "Local Posterior Inclusion Probabilities",
                             dpi = 300, framestyle = :box,
                             size = (780, 480), margin = 5Plots.mm)
    savefig_named(p_pips, "pips")

    # Local EHD heatmap
    p_ehd = plot_local_ehd(res_full;
                           feature_names = feature_names,
                           title = "Local Expected Hellinger Distance",
                           dpi = 300, framestyle = :box,
                           size = (780, 480), margin = 5Plots.mm)
    savefig_named(p_ehd, "eig")

    # Soft assignments
    p_assign = plot_assignments(res_full, data;
                                title  = "Soft Cluster Assignments",
                                dpi    = 300, framestyle = :box,
                                size   = (640, 420), margin = 5Plots.mm)
    savefig_named(p_assign, "assignments")

    # Assignment confidence + uninformative flags
    p_conf = plot_assignment_confidence(res_full;
                                        tau_out = tau_out,
                                        title   = "Assignment Confidence & Uninformative Flags",
                                        dpi     = 300, framestyle = :box,
                                        size    = (700, 350), margin = 5Plots.mm)
    savefig_named(p_conf, "confidence")

    # Cluster profiles
    p_prof = plot_profiles(res_full, data;
                           threshold     = tau,
                           feature_names = feature_names,
                           title         = "Cluster Feature Profiles",
                           dpi           = 300, framestyle = :box,
                           size          = (840, 420), margin = 5Plots.mm)
    savefig_named(p_prof, "profiles")

    println("\n" * "=" ^ 60)
    println("  Analysis complete. Figures saved to: $FIGURES_DIR")
    println("=" ^ 60)

    return (
        K_full          = K_full,
        elbo_final      = elbo_final,
        full_ari_std    = full_ari_std,
        full_ari_ext = full_ari_ext,
        test_ari        = test_ari,
        test_ll         = test_ll,
        n_uninf      = n_uninf,
        pi0             = pi0,
        pip_means       = pip_means,
        ehd_means       = ehd_means,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
