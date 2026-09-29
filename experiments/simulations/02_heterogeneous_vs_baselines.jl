# ==============================================================================
# 02_heterogeneous_vs_baselines.jl — Advantage of Heterogeneous Margins vs. Baselines
# ==============================================================================
# Benchmarks MixClustVI against:
# 1. Li et al. (2009) Baseline (Gauss-only: restricted to continuous Gaussian variables)
# 2. Standard GMM Relaxation (Gauss-all: treating count/skewed variables as Gaussian)
# 3. Unpenalized Mixture Baseline (All-features: clustering without variable selection)
#
# Directly addresses the Associate Editor's critique regarding comparisons with Li et al. (2009).
# ==============================================================================

include(joinpath(@__DIR__, "common.jl"))

function run_heterogeneous_vs_baselines(; n_rep::Int = DEFAULT_N_REP)
    @info "Starting Simulation 2: Heterogeneous Margins vs. Baselines with Predictive Evaluation (n_rep = $n_rep)"

    n_train, n_test = 200, 100
    K0, K_fit = 3, 8
    p_act, p_noise = 16, 64
    u0 = 0.01

    # Indices of Gaussian features in the block-structured dataset
    n_gauss = p_act ÷ 4 + p_noise ÷ 4

    # Preallocate metrics
    ari_train_full       = zeros(Float64, n_rep)
    ari_train_gauss_only = zeros(Float64, n_rep)
    ari_train_gauss_all  = zeros(Float64, n_rep)
    ari_train_no_fs      = zeros(Float64, n_rep)

    ari_test_full       = zeros(Float64, n_rep)
    ari_test_gauss_only = zeros(Float64, n_rep)
    ari_test_gauss_all  = zeros(Float64, n_rep)
    ari_test_no_fs      = zeros(Float64, n_rep)

    ll_test_full       = zeros(Float64, n_rep)
    ll_test_gauss_all  = zeros(Float64, n_rep)
    ll_test_no_fs      = zeros(Float64, n_rep)

    k_full       = zeros(Int, n_rep)
    k_gauss_only = zeros(Int, n_rep)
    k_gauss_all  = zeros(Int, n_rep)
    k_no_fs      = zeros(Int, n_rep)

    Threads.@threads for rep in 1:n_rep
        seed_train = GLOBAL_SEED + rep
        seed_test  = GLOBAL_SEED + 1000 + rep

        data_train, true_z_train, active_mask = generate_synthetic_dataset(
            n_train, p_act, p_noise, K0; seed = seed_train, model_type = 1
        )
        data_test, true_z_test, _ = generate_synthetic_dataset(
            n_test, p_act, p_noise, K0; seed = seed_test, model_type = 1
        )

        # 1. Full Heterogeneous MixClustVI (Proposed)
        res_full = mixClust(data_train, K_fit; model_setting = LFRM(),
                            u0 = u0, compact = false, max_iter = 500, tol = 1e-4)
        ari_train_full[rep] = adjusted_rand_index(true_z_train, res_full.labels)
        k_full[rep]         = map_cluster_order(res_full)

        w_test_full = predict_proba(res_full, data_test)
        pred_z_full = [argmax(w_test_full[i, :]) for i in 1:n_test]
        ari_test_full[rep] = adjusted_rand_index(true_z_test, pred_z_full)
        ll_test_full[rep]  = predictive_log_likelihood(res_full, data_test) / n_test

        # 2. Gauss-only Baseline (Li et al., 2009: continuous Gaussian features with iterative M-step background)
        data_gauss_train = data_train[1:n_gauss]
        data_gauss_test  = data_test[1:n_gauss]
        res_gauss = mixClust(data_gauss_train, K_fit; model_setting = LFRM(),
                             beta_estimation = :iterative,
                             u0 = u0, compact = false, max_iter = 500, tol = 1e-4)
        ari_train_gauss_only[rep] = adjusted_rand_index(true_z_train, res_gauss.labels)
        k_gauss_only[rep]         = map_cluster_order(res_gauss)

        w_test_gauss = predict_proba(res_gauss, data_gauss_test)
        pred_z_gauss = [argmax(w_test_gauss[i, :]) for i in 1:n_test]
        ari_test_gauss_only[rep] = adjusted_rand_index(true_z_test, pred_z_gauss)

        # 3. Gauss-all Baseline (Standard GMM relaxation for all scalar features)
        ft_gaussall = [eltype(data_train[j]) <: AbstractVector ? :multinomial : :gaussian
                       for j in eachindex(data_train)]
        res_gaussall = mixClust(data_train, K_fit; model_setting = LFRM(),
                                u0 = u0, compact = false,
                                feature_types = ft_gaussall,
                                max_iter = 500, tol = 1e-4)
        ari_train_gauss_all[rep] = adjusted_rand_index(true_z_train, res_gaussall.labels)
        k_gauss_all[rep]         = map_cluster_order(res_gaussall)

        w_test_gaussall = predict_proba(res_gaussall, data_test)
        pred_z_gaussall = [argmax(w_test_gaussall[i, :]) for i in 1:n_test]
        ari_test_gauss_all[rep] = adjusted_rand_index(true_z_test, pred_z_gaussall)
        ll_test_gauss_all[rep]  = predictive_log_likelihood(res_gaussall, data_test) / n_test

        # 4. Unpenalized Mixture Baseline (All features forced active, uniform gamma prior)
        res_nofs = mixClust(data_train, K_fit; model_setting = SFRM(),
                            u0 = u0, delta_prior = (100.0, 1.0),
                            compact = false, max_iter = 500, tol = 1e-4)
        ari_train_no_fs[rep] = adjusted_rand_index(true_z_train, res_nofs.labels)
        k_no_fs[rep]         = map_cluster_order(res_nofs)

        w_test_nofs = predict_proba(res_nofs, data_test)
        pred_z_nofs = [argmax(w_test_nofs[i, :]) for i in 1:n_test]
        ari_test_no_fs[rep] = adjusted_rand_index(true_z_test, pred_z_nofs)
        ll_test_no_fs[rep]  = predictive_log_likelihood(res_nofs, data_test) / n_test
    end

    results = (;
        train_ari = (; full = ari_train_full, gauss_only = ari_train_gauss_only,
                       gauss_all = ari_train_gauss_all, no_fs = ari_train_no_fs),
        test_ari  = (; full = ari_test_full, gauss_only = ari_test_gauss_only,
                       gauss_all = ari_test_gauss_all, no_fs = ari_test_no_fs),
        test_ll   = (; full = ll_test_full, gauss_all = ll_test_gauss_all, no_fs = ll_test_no_fs),
        k_order   = (; full = k_full, gauss_only = k_gauss_only,
                       gauss_all = k_gauss_all, no_fs = k_no_fs)
    )

    # Print Summary Table
    println("\n=== Simulation 2 Results: Proposed Heterogeneous Model vs. Baselines ===")
    println("Model Setting                     | Train ARI         | Out-of-Sample Test ARI | Test Log-Lik/Obs | Estimated K̂")
    println("-"^100)
    @printf("1. MixClustVI (Full Heterogeneous) | %5.3f ± %5.3f | %5.3f ± %5.3f        | %7.2f ± %5.2f   | %4.2f ± %4.2f\n",
            mean(ari_train_full), std(ari_train_full),
            mean(ari_test_full),  std(ari_test_full),
            mean(ll_test_full),   std(ll_test_full),
            mean(k_full),         std(k_full))
    @printf("2. Li et al. (2009) (Gauss-only)   | %5.3f ± %5.3f | %5.3f ± %5.3f        |       N/A         | %4.2f ± %4.2f\n",
            mean(ari_train_gauss_only), std(ari_train_gauss_only),
            mean(ari_test_gauss_only),  std(ari_test_gauss_only),
            mean(k_gauss_only),         std(k_gauss_only))
    @printf("3. GMM Relaxation (Gauss-all)      | %5.3f ± %5.3f | %5.3f ± %5.3f        | %7.2f ± %5.2f   | %4.2f ± %4.2f\n",
            mean(ari_train_gauss_all), std(ari_train_gauss_all),
            mean(ari_test_gauss_all),  std(ari_test_gauss_all),
            mean(ll_test_gauss_all),   std(ll_test_gauss_all),
            mean(k_gauss_all),         std(k_gauss_all))
    @printf("4. Unpenalized Mixture (No FS)     | %5.3f ± %5.3f | %5.3f ± %5.3f        | %7.2f ± %5.2f   | %4.2f ± %4.2f\n",
            mean(ari_train_no_fs), std(ari_train_no_fs),
            mean(ari_test_no_fs),  std(ari_test_no_fs),
            mean(ll_test_no_fs),   std(ll_test_no_fs),
            mean(k_no_fs),         std(k_no_fs))

    # Barplot of Out-of-Sample ARI comparisons
    labels = ["MixClustVI\n(Proposed)", "Li et al. (2009)\n(Gauss-only)", "GMM Relaxation\n(Gauss-all)", "No Feature\nSelection"]
    means = [mean(ari_test_full), mean(ari_test_gauss_only), mean(ari_test_gauss_all), mean(ari_test_no_fs)]
    stds  = [std(ari_test_full), std(ari_test_gauss_only), std(ari_test_gauss_all), std(ari_test_no_fs)]
    colors = [:dodgerblue, :coral, :mediumpurple, :darkgray]

    p = bar(labels, means, yerror = stds,
            ylabel = "Out-of-Sample Test ARI",
            title = "Posterior Predictive Generalization: MixClustVI vs. Baselines",
            legend = false, ylims = (0.0, 1.05),
            color = colors, bar_width = 0.5, grid = true)
    savefig(p, joinpath(FIGURES_DIR, "heterogeneous_vs_baselines.png"))

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_heterogeneous_vs_baselines()
end
