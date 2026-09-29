# Simulation study: Outlier Detection and Robust Clustering on Mixed-Type Data
# NaiveMFLFRMixDClust.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using NaiveMFLFRMixDClust
using Random
using Statistics
using Printf

println("="^75)
println("  MONTE CARLO SIMULATION: OUTLIER DETECTION & ORDER ROBUSTNESS (MIXED DATA)")
println("="^75)

function rand_multinomial(rng, N, p_vec)
    C = length(p_vec); counts = zeros(Int, C); cum_p = cumsum(p_vec)
    for _ in 1:N
        u = rand(rng); idx = findfirst(x -> u <= x, cum_p)
        counts[isnothing(idx) ? C : idx] += 1
    end
    return counts
end

function rand_poisson(rng, lambda)
    L = exp(-lambda); k = 0; p = 1.0
    while p > L
        k += 1; p *= rand(rng)
    end
    return k - 1
end

rand_gamma(rng, shape, rate) = -sum(log(rand(rng)) for _ in 1:round(Int, shape)) / rate

R = 20
n_in = 120
n_out = 15 # ~11% contamination
n_total = n_in + n_out
p = 8
tau_eval = 0.58

k_selected = Int[]
ari_standard = Float64[]
ari_robust = Float64[]
tpr_tau = Float64[]
fpr_tau = Float64[]
tpr_strict = Float64[]

inlier_idx = 1:n_in
outlier_idx = (n_in + 1):n_total
labels_true = vcat(fill(1, 40), fill(2, 40), fill(3, 40), fill(0, n_out))

println("Configuration: R = $R replications, n = $n_total ($n_in inliers across 3 clusters, $n_out outliers), p = 8 mixed features.")
println("Features: 4 active (Gaussian, Poisson, Multinomial, Gamma) + 4 noise (Gaussian, Poisson, Multinomial, Gamma).")
println("-"^75)

for rep in 1:R
    rng = Random.MersenneTwister(100 + rep)
    true_z_in = vcat(fill(1, 40), fill(2, 40), fill(3, 40))
    
    # 1. Active Gaussian
    y1_in = [c == 1 ? -2.0 + 0.5randn(rng) : c == 2 ? 0.0 + 0.5randn(rng) : 2.0 + 0.5randn(rng) for c in true_z_in]
    # 2. Noise Gaussian
    y2_in = randn(rng, n_in)
    # 3. Active Poisson
    y3_in = Float64[c == 1 ? rand_poisson(rng, 1.0) : c == 2 ? rand_poisson(rng, 6.0) : rand_poisson(rng, 14.0) for c in true_z_in]
    # 4. Noise Poisson
    y4_in = Float64[rand_poisson(rng, 4.0) for _ in 1:n_in]
    # 5. Active Multinomial
    y5_in = [c == 1 ? rand_multinomial(rng, 15, [0.85, 0.1, 0.05]) :
             c == 2 ? rand_multinomial(rng, 15, [0.1, 0.85, 0.05]) :
                      rand_multinomial(rng, 15, [0.05, 0.1, 0.85]) for c in true_z_in]
    # 6. Noise Multinomial
    y6_in = [rand_multinomial(rng, 15, [0.33, 0.33, 0.34]) for _ in 1:n_in]
    # 7. Active Gamma
    y7_in = [c == 1 ? rand_gamma(rng, 2, 1.0) : c == 2 ? rand_gamma(rng, 2, 5.0) : rand_gamma(rng, 2, 0.2) for c in true_z_in]
    # 8. Noise Gamma
    y8_in = [rand_gamma(rng, 2, 2.0) for _ in 1:n_in]
    
    # Outliers: background noise across all 8 features
    y1_out = randn(rng, n_out)
    y2_out = randn(rng, n_out)
    y3_out = Float64[rand_poisson(rng, 6.0) for _ in 1:n_out]
    y4_out = Float64[rand_poisson(rng, 4.0) for _ in 1:n_out]
    y5_out = [rand_multinomial(rng, 15, [0.33, 0.33, 0.34]) for _ in 1:n_out]
    y6_out = [rand_multinomial(rng, 15, [0.33, 0.33, 0.34]) for _ in 1:n_out]
    y7_out = [rand_gamma(rng, 2, 2.0) for _ in 1:n_out]
    y8_out = [rand_gamma(rng, 2, 2.0) for _ in 1:n_out]
    
    data = [
        vcat(y1_in, y1_out), vcat(y2_in, y2_out),
        vcat(y3_in, y3_out), vcat(y4_in, y4_out),
        vcat(y5_in, y5_out), vcat(y6_in, y6_out),
        vcat(y7_in, y7_out), vcat(y8_in, y8_out)
    ]
    
    res = mixClust(data, 8;
                   model_setting = LFRM(),
                   u0            = 0.01,
                   max_iter      = 400,
                   tol           = 1e-4,
                   compact       = true,
                   n_init        = 3,
                   max_iter_init = 8)
                   
    k_est = res.n_clusters
    push!(k_selected, k_est)
    
    # Baseline ARI (standard MAP ignoring outliers)
    push!(ari_standard, adjusted_rand_index(labels_true, res.labels))
    
    # Outlier detection with inactivation rate
    d_tau = detect_outliers(res; threshold=tau_eval, mode=:inactivation_rate)
    push!(tpr_tau, mean(d_tau[outlier_idx]))
    push!(fpr_tau, mean(d_tau[inlier_idx]))
    
    # Strict MAP rule
    d_strict = detect_outliers(res; mode=:map_strict)
    push!(tpr_strict, mean(d_strict[outlier_idx]))
    
    # Robust assignment ARI
    z_rob = robust_cluster_assignments(res; threshold=tau_eval, mode=:inactivation_rate)
    push!(ari_robust, adjusted_rand_index(labels_true, z_rob))
    
    @printf("Rep %2d: K_hat = %d | TPR(tau=%.2f) = %5.1f%% | FPR = %4.1f%% | ARI: %.3f -> %.3f\n",
            rep, k_est, tau_eval, 100*tpr_tau[end], 100*fpr_tau[end], ari_standard[end], ari_robust[end])
end

println("-"^75)
println("  MONTE CARLO SYNTHESIS ($R REPLICATIONS)")
println("-"^75)
println("Order selection: K* = 3 selected in $(count(==(3), k_selected)) / $R reps ($(round(100*count(==(3), k_selected)/R, digits=1))%)")
println("Mean estimated K: $(round(mean(k_selected), digits=2)) ± $(round(std(k_selected), digits=2))")
println("Outlier Detection (tau = $tau_eval):")
println("  Mean Detection Rate (TPR): $(round(100*mean(tpr_tau), digits=1))% ± $(round(100*std(tpr_tau), digits=1))%")
println("  Mean False Alarm Rate (FPR): $(round(100*mean(fpr_tau), digits=1))% ± $(round(100*std(fpr_tau), digits=1))%")
println("Strict MAP rule (all phi < 0.5):")
println("  Mean TPR:                  $(round(100*mean(tpr_strict), digits=1))%")
println("Partition Accuracy (Global ARI vs true partition including class 0):")
println("  Standard MAP:              $(round(mean(ari_standard), digits=4)) ± $(round(std(ari_standard), digits=4))")
println("  Robust MAP (with rejection):$(round(mean(ari_robust), digits=4)) ± $(round(std(ari_robust), digits=4))")
println("  Net ARI Gain:              +$(round(mean(ari_robust) - mean(ari_standard), digits=4))")
println("="^75)
