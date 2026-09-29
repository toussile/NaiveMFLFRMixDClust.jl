module NaiveMFLFRMixDClust

# Import dependencies
using DelimitedFiles
using SpecialFunctions
using Statistics
using LinearAlgebra
using Random
using Plots
using LaTeXStrings

export LaTeXString, @L_str

# Include files
include("types.jl")
include("margins.jl")
include("margins/gaussian.jl")
include("margins/lognormal.jl")
include("margins/exponential.jl")
include("margins/bernoulli.jl")
include("margins/multinomial.jl")
include("margins/poisson.jl")
include("margins/gamma.jl")
include("cavi.jl")
include("post_hoc.jl")
include("metrics.jl")
include("visualization.jl")
include("datasets.jl")

# Export types
export AbstractMargin, ModelSetting, SFRM, LFRM, MixClustResult
export GaussianMargin, LogNormalMargin, ExponentialMargin, BernoulliMargin, MultinomialMargin, PoissonMargin, GammaMargin

# Export functions
export mixClust, hellinger_divergence
export compute_ehd, compute_local_ehd, compute_eig, compute_local_eig
export filter_features, adjusted_rand_index
export predict_proba, predictive_log_likelihood, compact_clusters
export n_active_clusters
export coordinate_inactivation_rate, detect_outliers, robust_cluster_assignments
export compute_pi_0, outlier_indices, cluster_indices
export plot_elbo, plot_pips, plot_ehd, plot_eig, plot_assignments, plot_profiles
export plot_cluster_sizes, plot_assignment_confidence, plot_local_pips, plot_local_ehd, plot_local_eig
export plot_mixing_weights_evolution, plot_omega_history


# Export bundled datasets
export load_heart_disease, simulate_synthetic_cohort

function __init__()
    _ensure_latex_path!()
end

end # module NaiveMFLFRMixDClust
