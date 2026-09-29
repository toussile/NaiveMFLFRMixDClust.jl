# Ensure TinyTeX / TeX distributions are in PATH before loading Plots
let candidates = [
        joinpath(homedir(), "Library", "TinyTeX", "bin", "universal-darwin"),
        "/Library/TeX/texbin",
        "/usr/local/bin"
    ]
    cur_path = get(ENV, "PATH", "")
    for dir in candidates
        if isdir(dir) && !occursin(dir, cur_path)
            cur_path = dir * ":" * cur_path
        end
    end
    ENV["PATH"] = cur_path
end

using Test
using Random
using Statistics
using Plots
using LaTeXStrings
using NaiveMFLFRMixDClust

# Helper samplers for test data generation
function rand_multinomial(N, p_vec)
    C = length(p_vec)
    counts = zeros(Int, C)
    cum_p = cumsum(p_vec)
    for _ in 1:N
        u = rand()
        idx = findfirst(x -> u <= x, cum_p)
        if isnothing(idx)
            idx = C
        end
        counts[idx] += 1
    end
    return counts
end

function rand_poisson(lambda)
    L = exp(-lambda)
    k = 0
    p = 1.0
    while p > L
        k += 1
        p *= rand()
    end
    return k - 1
end

function rand_gamma_shape2(rate)
    # Gamma(2, rate) is the sum of two independent Exponentials
    u1 = rand()
    u2 = rand()
    return -log(u1 * u2) / rate
end

@testset "NaiveMFLFRMixDClust.jl Tests" begin
    Random.seed!(42)
    
    n = 120
    K_true = 3
    K_max = 8 # Overfitted mixture
    
    # Assign true cluster membership: 40 individuals per cluster
    true_z = vcat(fill(1, 40), fill(2, 40), fill(3, 40))
    
    # 1. Feature 1: Active Gaussian
    # C1: N(-2, 0.5^2), C2: N(0, 0.5^2), C3: N(2, 0.5^2)
    y1 = Float64[]
    for c in true_z
        if c == 1
            push!(y1, -2.0 + randn() * 0.5)
        elseif c == 2
            push!(y1, 0.0 + randn() * 0.5)
        else
            push!(y1, 2.0 + randn() * 0.5)
        end
    end
    
    # 2. Feature 2: Noise Gaussian
    # N(0, 1.0) for all
    y2 = randn(n)
    
    # 3. Feature 3: Active Poisson
    # C1: Poisson(1), C2: Poisson(6), C3: Poisson(12)
    y3 = Float64[]
    for c in true_z
        if c == 1
            push!(y3, Float64(rand_poisson(1.0)))
        elseif c == 2
            push!(y3, Float64(rand_poisson(6.0)))
        else
            push!(y3, Float64(rand_poisson(12.0)))
        end
    end
    
    # 4. Feature 4: Noise Poisson
    # Poisson(4) for all
    y4 = [Float64(rand_poisson(4.0)) for _ in 1:n]
    
    # 5. Feature 5: Active Multinomial (3 categories, 15 trials)
    # C1: [0.8, 0.1, 0.1], C2: [0.1, 0.8, 0.1], C3: [0.1, 0.1, 0.8]
    y5 = Vector{Int}[]
    for c in true_z
        if c == 1
            push!(y5, rand_multinomial(15, [0.8, 0.1, 0.1]))
        elseif c == 2
            push!(y5, rand_multinomial(15, [0.1, 0.8, 0.1]))
        else
            push!(y5, rand_multinomial(15, [0.1, 0.1, 0.8]))
        end
    end
    
    # 6. Feature 6: Noise Multinomial
    # [0.33, 0.33, 0.33] for all
    y6 = [rand_multinomial(15, [0.33, 0.33, 0.33]) for _ in 1:n]
    
    # 7. Feature 7: Active Gamma (shape = 2.0)
    # C1: Gamma(2, rate=1.0), C2: Gamma(2, rate=5.0), C3: Gamma(2, rate=0.2)
    y7 = Float64[]
    for c in true_z
        if c == 1
            push!(y7, rand_gamma_shape2(1.0))
        elseif c == 2
            push!(y7, rand_gamma_shape2(5.0))
        else
            push!(y7, rand_gamma_shape2(0.2))
        end
    end
    
    # 8. Feature 8: Noise Gamma
    # Gamma(2, rate=2.0) for all
    y8 = [rand_gamma_shape2(2.0) for _ in 1:n]
    
    # Combine features into the heterogeneous dataset vector
    # Order: [ActGauss, NoiseGauss, ActPoi, NoisePoi, ActMult, NoiseMult, ActGamma, NoiseGamma]
    dataset = [y1, y2, y3, y4, y5, y6, y7, y8]
    p_total = length(dataset)
    
    # Verify dataset dimensions and types
    @test length(dataset) == 8
    @test length(dataset[1]) == n
    
    @testset "CAVI Model 1 (Shared Relevance)" begin
        # Fit overfitted CAVI Model 1
        results = mixClust(dataset, K_max; model_setting=SFRM(), max_iter=80, tol=1e-5, u0=0.01, compact=false)
        
        # Check output structure sizes
        @test size(results.w) == (n, K_max)
        @test size(results.pip) == (n, p_total)
        @test length(results.u_star) == K_max
        @test length(results.margins) == p_total
        @test length(results.elbo_history) >= 2
        
        # Verify that ELBO converges and values are finite
        @test all(!isnan, results.elbo_history)
        # ELBO must be monotonically non-decreasing (CAVI guarantee)
        @test all(i -> results.elbo_history[i] >= results.elbo_history[i-1] - 1e-6,
                  2:length(results.elbo_history))
        
        # Compute Expected Information Gain (EIG) post-hoc
        eig = compute_eig(results.margins, results.w, results.pip)
        @test length(eig) == p_total
        @test all(x -> x >= 0, eig)
        
        # Perform feature selection based on EIG with threshold tau = 0.1
        tau = 0.1
        active_indices = filter_features(eig, tau)
        
        println("Model 1 EIG values:")
        for (idx, val) in enumerate(eig)
            println("Feature $idx (Active label: $(idx % 2 == 1)): EIG = ", round(val, digits=4))
        end
        
        # Check that active features (1, 3, 5, 7) have significantly higher EIG than noise features (2, 4, 6, 8)
        @test eig[1] > eig[2]
        @test eig[3] > eig[4]
        @test eig[5] > eig[6]
        @test eig[7] > eig[8]
        
        # Test visualizations
        p_elbo = plot_elbo(results)
        p_pips = plot_pips(results)
        p_eig = plot_eig(eig, tau)
        p_w = plot_assignments(results, dataset)
        p_prof = plot_profiles(results, dataset; threshold=tau)
        p_omega = plot_mixing_weights_evolution(results)
        p_omega_alias = plot_omega_history(results)
        p_loc_pip = plot_local_pips(results; only_active=true)
        p_loc_pip_all = plot_local_pips(results; only_active=false)
        p_loc_eig = plot_local_eig(results; only_active=true)
        p_loc_eig_all = plot_local_eig(results; only_active=false)
        
        # Save plots to files to verify they save correctly
        plots_dir = joinpath(@__DIR__, "plots")
        mkpath(plots_dir)
        savefig(p_elbo, joinpath(plots_dir, "elbo_m1.png"))
        savefig(p_pips, joinpath(plots_dir, "pips_m1.png"))
        savefig(p_eig, joinpath(plots_dir, "eig_m1.png"))
        savefig(p_w, joinpath(plots_dir, "assignments_m1.png"))
        savefig(p_prof, joinpath(plots_dir, "profiles_m1.png"))
        savefig(p_omega, joinpath(plots_dir, "omega_m1.png"))
        savefig(p_loc_pip, joinpath(plots_dir, "loc_pip_m1.png"))
        savefig(p_loc_eig, joinpath(plots_dir, "loc_eig_m1.png"))
        
        @test isfile(joinpath(plots_dir, "elbo_m1.png"))
        @test isfile(joinpath(plots_dir, "pips_m1.png"))
        @test isfile(joinpath(plots_dir, "eig_m1.png"))
        @test isfile(joinpath(plots_dir, "assignments_m1.png"))
        @test isfile(joinpath(plots_dir, "profiles_m1.png"))
        @test isfile(joinpath(plots_dir, "omega_m1.png"))
        @test isfile(joinpath(plots_dir, "loc_pip_m1.png"))
        @test isfile(joinpath(plots_dir, "loc_eig_m1.png"))
        @test size(results.omega_history, 2) == K_max
        @test size(results.omega_history, 1) == length(results.elbo_history)
        
        @testset "Post-hoc Prediction and Model Compaction" begin
            # 1. Test predictive density of concrete margins on test data
            for j in 1:p_total
                P_j = NaiveMFLFRMixDClust.predictive_density(results.margins[j], dataset[j][1:10])
                @test size(P_j) == (10, K_max)
                @test all(x -> x >= 0 && !isnan(x) && isfinite(x), P_j)
            end
            
            # 2. Test predict_proba and predictive_log_likelihood
            w_pred = predict_proba(results, dataset)
            @test size(w_pred) == (n, K_max)
            @test all(x -> isapprox(sum(w_pred[x, :]), 1.0, atol=1e-6), 1:n)
            
            log_lik = predictive_log_likelihood(results, dataset)
            @test isfinite(log_lik)
            @test log_lik < 0.0
            
            # 3. Test compact_clusters
            results_compact = compact_clusters(results, dataset)
            K_compact = size(results_compact.w, 2)
            @test K_compact <= K_max
            @test size(results_compact.w, 1) == n
            @test length(results_compact.margins) == p_total
            @test length(results_compact.u_star) == K_compact
            # Test direct slice without dataset
            results_sliced = compact_clusters(results)
            @test size(results_sliced.w) == size(results_compact.w)
        end
    end

    @testset "CAVI Model 2 (Cluster-Specific Relevance)" begin
        # Fit overfitted CAVI Model 2
        results = mixClust(dataset, K_max; model_setting=LFRM(), max_iter=80, tol=1e-5, u0=0.01, compact=false)
        
        # Check output structure sizes
        @test size(results.w) == (n, K_max)
        @test size(results.pip) == (n, p_total)
        @test length(results.u_star) == K_max
        @test length(results.margins) == p_total
        @test length(results.elbo_history) >= 2
        @test all(!isnan, results.elbo_history)
        # ELBO must be monotonically non-decreasing (CAVI guarantee)
        @test all(i -> results.elbo_history[i] >= results.elbo_history[i-1] - 1e-6,
                  2:length(results.elbo_history))

        # Compute EIG and verify active vs noise feature separation
        eig = compute_eig(results.margins, results.w, results.pip)

        println("Model 2 EIG values:")
        for (idx, val) in enumerate(eig)
            println("Feature $idx (Active label: $(idx % 2 == 1)): EIG = ", round(val, digits=4))
        end
        
        @test eig[1] > eig[2]
        @test eig[3] > eig[4]
        @test eig[5] > eig[6]
        @test eig[7] > eig[8]
    end

    @testset "Computed properties of MixClustResult" begin
        results = mixClust(dataset, K_max; model_setting=SFRM(), max_iter=30, tol=1e-4,
                           u0=0.01, compact=true)
        @test results.n_obs      == n
        @test results.n_features == length(dataset)
        @test results.n_clusters == size(results.w, 2)
        @test length(results.cluster_sizes) == results.n_clusters
        @test sum(results.cluster_sizes) == n
        @test length(results.inactivation_rate) == n
        @test all(0.0 .<= results.inactivation_rate .<= 1.0)
        @test 0.0 <= results.pi_0 <= 1.0
        @test isapprox(results.pi_0, compute_pi_0(results))
    end

    @testset "simulate_synthetic_cohort" begin
        cohort = simulate_synthetic_cohort()
        @test length(cohort.data) == 7
        @test length(cohort.labels) == 150
        @test all(l -> l in (1, 2, 3), cohort.labels)
        @test length(cohort.feature_names) == 7
        # Reproducibility: same seed → same data
        cohort2 = simulate_synthetic_cohort(seed=2026)
        @test cohort.labels == cohort2.labels
        @test cohort.data[1] == cohort2.data[1]
        # Custom size
        cohort3 = simulate_synthetic_cohort(n=60)
        @test length(cohort3.labels) == 60
    end

    @testset "load_heart_disease" begin
        hd = load_heart_disease()
        @test length(hd.data) == 13
        @test length(hd.labels) == 297
        @test all(l -> l in (0, 1), hd.labels)
        @test length(hd.feature_names) == 13
        # Continuous features are standardized (mean ≈ 0)
        for j in [1, 4, 5, 8, 10]
            @test abs(mean(hd.data[j])) < 1e-10
        end
        # Categorical features are one-hot vectors
        for j in [2, 3, 6, 7, 9, 11, 12, 13]
            @test all(v -> sum(v) == 1, hd.data[j])
        end
    end

    @testset "CAVI with Iterative Background Estimation" begin
        # 1. SFRM + :iterative
        res_iter1 = mixClust(dataset, K_max; model_setting=SFRM(), max_iter=40, tol=1e-4, u0=0.01, compact=false, beta_estimation=:iterative)
        @test size(res_iter1.w) == (n, K_max)
        @test all(!isnan, res_iter1.elbo_history)
        # Note: :iterative uses heuristic background updates (not proper coord ascent),
        # so strict monotonicity is not guaranteed — we only check finiteness.

        # 2. LFRM + :iterative
        res_iter2 = mixClust(dataset, K_max; model_setting=LFRM(), max_iter=40, tol=1e-4, u0=0.01, compact=false, beta_estimation=:iterative)
        @test size(res_iter2.w) == (n, K_max)
        @test all(!isnan, res_iter2.elbo_history)
    end

    @testset "Outlier Detection and Robust Assignments" begin
        # Synthetic PIP matrix: 3 individuals, 4 features
        # Obs 1: inlier (all features active)
        # Obs 2: partial/noise
        # Obs 3: outlier (all features inactive)
        pip_test = [0.9 0.95 0.85 0.9;
                    0.6 0.4  0.7  0.3;
                    0.05 0.1 0.02 0.08]
        
        rho = coordinate_inactivation_rate(pip_test)
        @test length(rho) == 3
        @test rho[1] < 0.2
        @test rho[3] > 0.9
        
        # 1. Inactivation rate threshold
        out_rho = detect_outliers(pip_test; threshold=0.7, mode=:inactivation_rate)
        @test out_rho == Bool[false, false, true]
        
        # 2. Strict MAP (all phi < 0.5)
        out_strict = detect_outliers(pip_test; mode=:map_strict)
        @test out_strict == Bool[false, false, true]
        
        # 3. Relaxed MAP (proportion < 0.5 >= 0.75)
        out_relaxed = detect_outliers(pip_test; threshold=0.75, mode=:map_relaxed)
        @test out_relaxed == Bool[false, false, true]
        
        # Test robust_cluster_assignments with MixClustResult
        dummy_margins = [GaussianMargin(randn(20), 2) for _ in 1:4]
        res_dummy = MixClustResult(
            [0.9 0.1; 0.2 0.8; 0.6 0.4], # w
            [1, 2, 1],                   # labels
            pip_test,                    # pip
            [10.0, 10.0],                # u_star
            fill(1.0, 4, 2),             # delta_star
            dummy_margins,               # margins
            [0.0]                        # elbo_history
        )
        
        @test coordinate_inactivation_rate(res_dummy) == rho
        @test res_dummy.inactivation_rate == rho
        z_rob = robust_cluster_assignments(res_dummy; threshold=0.7, mode=:inactivation_rate)
        @test z_rob == [1, 2, 0] # Obs 3 rejected to class 0

        # Test outlier_indices and cluster_indices helpers
        @test outlier_indices(res_dummy; threshold=0.7) == [3]
        @test cluster_indices(res_dummy, 1; robust=false) == [1, 3]
        @test cluster_indices(res_dummy, 1; threshold=0.7, robust=true) == [1]
        @test cluster_indices(res_dummy, 2; robust=true) == [2]

        # Test compute_pi_0 and res_dummy.pi_0
        pi_0_val = compute_pi_0(res_dummy)
        @test 0.0 <= pi_0_val <= 1.0
        @test res_dummy.pi_0 == pi_0_val
    end

    @testset "Robust Background Calibration & New Margins" begin
        # 1. Gaussian robust calibration (immunity to severe outliers)
        clean_g = randn(100)
        corrupted_g = vcat(clean_g, [1000.0, 2000.0]) # Severe outliers
        g_robust = GaussianMargin(corrupted_g, 2; robust=true)
        g_naive = GaussianMargin(corrupted_g, 2; robust=false)
        @test abs(g_robust.mu_bg - median(clean_g)) < 0.5
        @test g_robust.mu_bg < 5.0 # median is immune
        @test g_naive.mu_bg > 25.0 # sample mean is corrupted
        @test g_robust.tau_bg > 0.1 # reasonable precision

        # 2. LogNormal Margin
        pos_clean = exp.(randn(100))
        pos_corrupted = vcat(pos_clean, [1e6, 2e6])
        ln_robust = LogNormalMargin(pos_corrupted, 2; robust=true)
        @test ln_robust.mu_bg < 5.0
        @test ln_robust.tau_bg > 0.1
        eld_ln = NaiveMFLFRMixDClust.expected_log_density(ln_robust, pos_clean[1:5])
        @test size(eld_ln) == (5, 2)
        @test all(isfinite, eld_ln)
        bld_ln = NaiveMFLFRMixDClust.background_log_density(ln_robust, pos_clean[1:5])
        @test length(bld_ln) == 5
        @test all(isfinite, bld_ln)

        # 3. Exponential Margin
        exp_clean = rand(100) .+ 0.1
        exp_corrupted = vcat(exp_clean, [1000.0, 2000.0])
        exp_robust = ExponentialMargin(exp_corrupted, 2; robust=true)
        @test exp_robust.lambda_bg > 0.1 # robust rate matching median
        eld_exp = NaiveMFLFRMixDClust.expected_log_density(exp_robust, exp_clean[1:5])
        @test size(eld_exp) == (5, 2)
        @test all(isfinite, eld_exp)

        # 4. Gamma robust calibration
        gam_clean = rand(100) .+ 1.0
        gam_corrupted = vcat(gam_clean, [500.0, 1000.0])
        gam_robust = GammaMargin(gam_corrupted, 2; robust=true)
        @test gam_robust.a_bg > 0.01
        @test gam_robust.b_bg > 0.01

        # 5. Poisson robust calibration
        poi_clean = [rand_poisson(3.0) for _ in 1:100]
        poi_corrupted = Float64.(vcat(poi_clean, [500, 1000]))
        poi_robust = PoissonMargin(poi_corrupted, 2; robust=true)
        @test poi_robust.lambda_bg < 10.0 # median is immune to 2 massive outliers

        # Poisson with zeros (testing trimmed mean fallback)
        poi_zeros = zeros(Float64, 100)
        poi_zeros[95:100] .= 10.0 # mostly zeros, few non-zeros
        poi_z_margin = PoissonMargin(poi_zeros, 2; robust=true)
        @test poi_z_margin.lambda_bg >= 0.1

        # 6. Bernoulli Laplace smoothing
        bern_data = [0.0, 0.0, 0.0, 0.0]
        bern_margin = BernoulliMargin(bern_data, 2; robust=true)
        @test bern_margin.p_bg == (1.0 + 0.0) / (4.0 + 2.0) # 1/6 != 0

        # 7. Multinomial symmetric Dirichlet smoothing
        mult_data = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
        mult_margin = MultinomialMargin(mult_data, 2; robust=true)
        # Category 2 and 3 should have strictly positive background probability
        @test all(mult_margin.phi_bg .> 0.0)
        @test isapprox(sum(mult_margin.phi_bg), 1.0)

        # 8. End-to-end CAVI with mixed dataset including LogNormal and Exponential
        n_sub = 60
        y_ln = exp.(randn(n_sub))
        y_exp = rand(n_sub) .+ 0.2
        sub_dataset = [y_ln, y_exp]
        res_mixed = mixClust(sub_dataset, 3; 
                             feature_types=[:lognormal, :exponential], 
                             model_setting=LFRM(), 
                             max_iter=30, tol=1e-4, compact=false)
        @test size(res_mixed.w) == (n_sub, 3)
        @test length(res_mixed.margins) == 2
        @test isa(res_mixed.margins[1], LogNormalMargin)
        @test isa(res_mixed.margins[2], ExponentialMargin)
        @test all(!isnan, res_mixed.elbo_history)
        eig_mixed = compute_eig(res_mixed.margins, res_mixed.w, res_mixed.pip)
        @test length(eig_mixed) == 2
        @test all(x -> 0.0 <= x < 1.0, eig_mixed)
        
        # Test compact_clusters on mixed dataset
        res_compact = compact_clusters(res_mixed, sub_dataset)
        @test size(res_compact.w, 1) == n_sub
        @test size(res_compact.w, 2) <= 3
        @test isa(res_compact.margins[1], LogNormalMargin)
        @test isa(res_compact.margins[2], ExponentialMargin)
    end

    @testset "Hellinger Divergence & Expected Hellinger Distance (EHD)" begin
        # 1. Gaussian margin
        gauss_m = GaussianMargin(randn(100), 2)
        h2_gauss = hellinger_divergence(gauss_m)
        @test length(h2_gauss) == 2
        @test all(0.0 .<= h2_gauss .<= 1.0)

        # 2. LogNormal margin
        ln_m = LogNormalMargin(exp.(randn(100)), 2)
        h2_ln = hellinger_divergence(ln_m)
        @test length(h2_ln) == 2
        @test all(0.0 .<= h2_ln .<= 1.0)

        # 3. Exponential margin
        exp_m = ExponentialMargin(rand(100) .+ 0.1, 2)
        h2_exp = hellinger_divergence(exp_m)
        @test length(h2_exp) == 2
        @test all(0.0 .<= h2_exp .<= 1.0)

        # 4. Bernoulli margin
        bern_m = BernoulliMargin(rand(0:1, 100), 2)
        h2_bern = hellinger_divergence(bern_m)
        @test length(h2_bern) == 2
        @test all(0.0 .<= h2_bern .<= 1.0)

        # 5. Multinomial margin
        mult_m = MultinomialMargin([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 2)
        h2_mult = hellinger_divergence(mult_m)
        @test length(h2_mult) == 2
        @test all(0.0 .<= h2_mult .<= 1.0)

        # 6. Poisson margin
        poi_m = PoissonMargin(Float64.([1, 2, 3, 4, 5]), 2)
        h2_poi = hellinger_divergence(poi_m)
        @test length(h2_poi) == 2
        @test all(0.0 .<= h2_poi .<= 1.0)

        # 7. Gamma margin
        gam_m = GammaMargin(rand(100) .+ 0.5, 2)
        h2_gam = hellinger_divergence(gam_m)
        @test length(h2_gam) == 2
        @test all(0.0 .<= h2_gam .<= 1.0)

        # 8. EHD vs EIG equivalence / aliases and bounds on full model
        w_fake = [0.6 0.4; 0.7 0.3; 0.2 0.8]
        pip_fake = [0.9, 0.1]
        margins_fake = [gauss_m, exp_m]

        ehd_loc = compute_local_ehd(margins_fake, w_fake, pip_fake)
        @test size(ehd_loc) == (2, 2)
        @test all(0.0 .<= ehd_loc .<= 1.0)
        @test compute_local_eig(margins_fake, w_fake, pip_fake) == ehd_loc

        ehd_glob = compute_ehd(margins_fake, w_fake, pip_fake)
        @test length(ehd_glob) == 2
        @test all(0.0 .<= ehd_glob .<= 1.0)
        @test compute_eig(margins_fake, w_fake, pip_fake) == ehd_glob

        # 9. Visualization tests for EHD
        p_bar = plot_ehd(ehd_glob, 0.5)
        @test p_bar isa Plots.Plot
        p_bar_alias = plot_eig(ehd_glob, 0.5)
        @test p_bar_alias isa Plots.Plot
    end
end



