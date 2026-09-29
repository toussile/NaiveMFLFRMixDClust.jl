# ==============================================================================
# common.jl — Unified Environment and Utilities for NaiveMFLFRMixDClust Simulations
# ==============================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using NaiveMFLFRMixDClust
using Statistics, Random, Printf, DelimitedFiles
using Plots

# Include unified data generator
include(joinpath(@__DIR__, "data_generator.jl"))

# Paths for figures and simulation data
const SIM_DIR     = @__DIR__
const FIGURES_DIR = normpath(joinpath(SIM_DIR, "../figures"))
const DATA_DIR    = normpath(joinpath(SIM_DIR, "../data"))

mkpath(FIGURES_DIR)
mkpath(DATA_DIR)

# Global simulation constants
const GLOBAL_SEED = 2026
const DEFAULT_N_REP = 10

"""
    compute_tpr_fdr(active_mask::BitVector, selected_mask::BitVector)

Compute descriptive True Positive Rate (TPR) and empirical False Discovery Rate (FDR).
"""
function compute_tpr_fdr(active_mask, selected_mask)
    tp = sum(active_mask .& selected_mask)
    fp = sum(.!active_mask .& selected_mask)
    n_active = sum(active_mask)
    n_selected = sum(selected_mask)
    
    tpr = n_active > 0 ? tp / n_active : 0.0
    fdr = n_selected > 0 ? fp / n_selected : 0.0
    return (; tpr, fdr, tp, fp)
end

"""
    map_cluster_order(res::MixClustResult) -> Int

Determine the number of clusters K_hat populated under MAP hard assignment.
"""
function map_cluster_order(res::MixClustResult)
    return length(unique(res.labels))
end
