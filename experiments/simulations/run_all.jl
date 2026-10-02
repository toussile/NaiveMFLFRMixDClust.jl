# ==============================================================================
# run_all.jl — Master Simulation Suite for NaiveMFLFRMixDClust
# ==============================================================================
# Executes all simulation studies in sequence:
# 1. In-Model Cluster Order Selection (u⁽⁰⁾ sweep, MAP order K̂, ARI)
# 2. Heterogeneous Margins vs. Baselines (Li et al. 2009 & GMM relaxation)
# 3. Shared vs. Local Feature Saliency (SFRM vs. LFRM with Kent EIG)
# 4. High-Dimensional Screening Behavior (TPR, empirical FDR, vs. No-FS)
# ==============================================================================

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "01_order_selection.jl"))
include(joinpath(@__DIR__, "02_heterogeneous_vs_baselines.jl"))
include(joinpath(@__DIR__, "03_shared_vs_local_saliency.jl"))
include(joinpath(@__DIR__, "04_high_dimensional_screening.jl"))
include(joinpath(@__DIR__, "05_uninformative_absorption.jl"))

function run_all(; n_rep::Int = 10)
    println("==================================================================")
    println("  NaiveMFLFRMixDClust: Reproducible Simulation Suite Execution")
    println("  Replications per experiment: $n_rep")
    println("==================================================================\n")

    t0 = time()

    println("\n>>> [1/5] Running Simulation 1: In-Model Cluster Order Selection...")
    res1 = run_order_selection(; n_rep = n_rep)

    println("\n>>> [2/5] Running Simulation 2: Heterogeneous Margins vs. Baselines...")
    res2 = run_heterogeneous_vs_baselines(; n_rep = n_rep)

    println("\n>>> [3/5] Running Simulation 3: Shared vs. Local Saliency...")
    res3 = run_shared_vs_local(; n_rep = n_rep)

    println("\n>>> [4/5] Running Simulation 4: High-Dimensional Screening Behavior...")
    res4 = run_high_dimensional_screening(; n_rep = n_rep)

    println("\n>>> [5/5] Running Simulation 5: Absorption of Uninformative Observations...")
    res5 = run_uninformative_absorption(; n_rep = 20)

    total_time = time() - t0
    @printf("\nAll simulations completed successfully in %4.2f minutes.\n", total_time / 60.0)
    println("Generated figures saved to: $FIGURES_DIR")
    println("Simulation data saved to:   $DATA_DIR")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
