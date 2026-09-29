using Plots
using LaTeXStrings

# Ensure standard LaTeX paths (TinyTeX, MacTeX, TeXLive) are in PATH for GR rendering
function _ensure_latex_path!()
    candidates = [
        joinpath(homedir(), "Library", "TinyTeX", "bin", "universal-darwin"),
        "/Library/TeX/texbin",
        "/usr/local/bin",
    ]
    current_path = get(ENV, "PATH", "")
    for dir in candidates
        if isdir(dir) && !occursin(dir, current_path)
            current_path = dir * ":" * current_path
        end
    end
    ENV["PATH"] = current_path
end
_ensure_latex_path!()

# ── Internal defaults ────────────────────────────────────────────────────────
const _THEME = (
    dpi        = 300,
    framestyle = :box,
    tickfont   = font(10),
    guidefont  = font(11),
    legendfont = font(9),
    titlefont  = font(12, :bold),
    size       = (700, 420),
    margin     = 5Plots.mm,
)

const _C_BLUE   = "#2563EB"
const _C_GREEN  = "#10B981"
const _C_RED    = "#EF4444"
const _C_ORANGE = "#F59E0B"
const _C_GREY   = "#9CA3AF"

const _IMPORTANCE_COLORMAP = :viridis

const _CLUSTER_PALETTE = [
    "#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
    "#EC4899", "#14B8A6", "#F97316", "#6366F1", "#84CC16",
]

function _cluster_colors(K::Int)
    [_CLUSTER_PALETTE[mod1(k, length(_CLUSTER_PALETTE))] for k in 1:K]
end

function _feature_labels(feature_names, p::Int)
    isnothing(feature_names) ? [L"x_{%$j}" for j in 1:p] : collect(feature_names)
end

# ── Public API ───────────────────────────────────────────────────────────────

"""
    plot_pips(results; feature_names=nothing, kwargs...)

Bar chart of mean Posterior Inclusion Probabilities per feature.
Bars above 0.5 are highlighted in blue; others in grey.
"""
function plot_pips(results::MixClustResult;
                   feature_names=nothing, kwargs...)
    pips = vec(mean(results.pip, dims=1))
    p    = length(pips)
    lbls = _feature_labels(feature_names, p)
    colors = [v >= 0.5 ? _C_BLUE : _C_GREY for v in pips]

    bar(1:p, pips;
        xticks     = (1:p, lbls),
        xrotation  = 45,
        ylabel     = L"\bar{\gamma}_j",
        legend     = false,
        color      = colors,
        linecolor  = :match,
        ylim       = (0, 1.05),
        title      = "Posterior Inclusion Probabilities (" * L"\bar{\gamma}_j" * ")",
        _THEME...,
        kwargs...)
end

"""
    plot_ehd(ehd, threshold; feature_names=nothing, kwargs...)

Bar chart of Expected Hellinger Distance values with a threshold line.
Active features (EHD ≥ threshold) are shown in green; others in grey.
"""
function plot_ehd(ehd::Vector{Float64}, threshold::Real;
                  feature_names=nothing, kwargs...)
    p    = length(ehd)
    lbls = _feature_labels(feature_names, p)
    colors = [v >= threshold ? _C_GREEN : _C_GREY for v in ehd]

    p_plot = bar(1:p, ehd;
                 xticks    = (1:p, lbls),
                 xrotation = 45,
                 ylabel    = L"\mathbf{EHD}_j",
                 legend    = false,
                 color     = colors,
                 linecolor = :match,
                 title     = "Expected Hellinger Distance (" * L"\tau = %$threshold" * ")",
                 _THEME...,
                 kwargs...)

    hline!(p_plot, [threshold];
           line  = (2, :dash, _C_RED),
           label = "threshold " * L"\tau = %$threshold")
    return p_plot
end

# Backward compatibility alias
const plot_eig = plot_ehd

"""
    plot_profiles(results, data; feature_names=nothing, threshold=0.1, only_active=true, kwargs...)

Heatmap of cluster-specific feature profiles (z-score vs. background) for
features whose EHD exceeds `threshold`. Rows = clusters, columns = active features.
A positive (red) cell means the cluster's expected value is above the population
background; negative (blue) means below.
By default, only active clusters are displayed (`only_active=true`).
"""
function plot_profiles(results::MixClustResult, data::AbstractVector;
                       feature_names=nothing, threshold=0.1, only_active=true, kwargs...)
    p   = length(results.margins)
    K   = size(results.w, 2)
    ehd = compute_ehd(results.margins, results.w, results.pip)

    active_idx = findall(x -> x >= threshold, ehd)
    isempty(active_idx) && (active_idx = 1:p)


    lbls = _feature_labels(feature_names, p)
    active_lbls = lbls[active_idx]

    profile_matrix = Matrix{Float64}(undef, K, length(active_idx))

    for (col, j) in enumerate(active_idx)
        margin = results.margins[j]

        if margin isa GaussianMargin
            sd_bg = 1.0 / sqrt(max(margin.tau_bg, 1e-10))
            profile_matrix[:, col] = (margin.mu_star .- margin.mu_bg) ./ sd_bg

        elseif margin isa LogNormalMargin
            sd_bg = 1.0 / sqrt(max(margin.tau_bg, 1e-10))
            profile_matrix[:, col] = (margin.mu_star .- margin.mu_bg) ./ sd_bg

        elseif margin isa ExponentialMargin
            sd_bg = 1.0 / max(margin.lambda_bg, 1e-10)
            expected = margin.alpha_star ./ margin.beta_star
            profile_matrix[:, col] = (expected .- (1.0 / margin.lambda_bg)) ./ sd_bg

        elseif margin isa BernoulliMargin
            sd_bg = sqrt(max(margin.p_bg * (1.0 - margin.p_bg), 1e-10))
            expected = margin.alpha_star ./ (margin.alpha_star .+ margin.beta_star)
            profile_matrix[:, col] = (expected .- margin.p_bg) ./ sd_bg

        elseif margin isa PoissonMargin
            sd_bg = sqrt(max(margin.lambda_bg, 1e-10))
            expected = margin.a_star ./ margin.b_star
            profile_matrix[:, col] = (expected .- margin.lambda_bg) ./ sd_bg

        elseif margin isa GammaMargin
            mu_bg = margin.a_bg / margin.b_bg
            sd_bg = sqrt(margin.a_bg) / margin.b_bg
            alpha_m1 = [max(margin.alpha_star[k] - 1.0, 1e-5) for k in 1:K]
            expected  = margin.a_cl .* margin.beta_star ./ alpha_m1
            profile_matrix[:, col] = (expected .- mu_bg) ./ sd_bg

        elseif margin isa MultinomialMargin
            vsum     = [sum(margin.varphi_star[k, :]) for k in 1:K]
            exp_prob = margin.varphi_star ./ vsum
            dev      = exp_prob .- margin.phi_bg'
            best_cat = argmax(vec(sum(abs.(dev), dims=1)))
            denom    = sqrt(max(margin.phi_bg[best_cat] * (1.0 - margin.phi_bg[best_cat]), 1e-10))
            profile_matrix[:, col] = dev[:, best_cat] ./ denom
        else
            profile_matrix[:, col] = zeros(K)
        end
    end

    active_k = only_active ? sort(unique(results.labels)) : collect(1:K)
    isempty(active_k) && (active_k = collect(1:K))
    K_act = length(active_k)
    profile_act = profile_matrix[active_k, :]
    cluster_labels = [L"C_{%$k}" for k in active_k]

    heatmap(active_lbls, 1:K_act, profile_act;
            yticks = (1:K_act, cluster_labels),
            xlabel = "Active Features",
            ylabel = "Clusters",
            title  = "Subtype Feature Profiles (z-score vs. background)",
            color  = :coolwarm,
            clim   = (-3.0, 3.0),
            _THEME...,
            kwargs...)
end

"""
    plot_assignments(results, data; x_feature=1, y_feature=2, feature_names=nothing, kwargs...)

Scatter plot of individuals projected onto two features, coloured by their
hard cluster assignment (argmax of posterior responsibility).
"""
function plot_assignments(results::MixClustResult, data::AbstractVector;
                          x_feature::Int=1, y_feature::Int=2,
                          feature_names=nothing, kwargs...)
    K      = size(results.w, 2)
    labels = results.labels
    lbls   = _feature_labels(feature_names, length(data))
    colors = _cluster_colors(K)

    plt = plot(; title  = "Cluster Assignments (" * L"\hat{K} = %$K" * ")",
                 xlabel = lbls[x_feature],
                 ylabel = lbls[y_feature],
                 legend = :outertopright,
                 _THEME...,
                 kwargs...)

    x_vals = data[x_feature]
    y_vals = data[y_feature]
    # Handle Multinomial features: use most-likely category index as numeric proxy
    if x_vals isa Vector{<:AbstractVector}
        x_vals = Float64[argmax(v) for v in x_vals]
    end
    if y_vals isa Vector{<:AbstractVector}
        y_vals = Float64[argmax(v) for v in y_vals]
    end

    for k in 1:K
        idx = findall(==(k), labels)
        isempty(idx) && continue
        scatter!(plt, x_vals[idx], y_vals[idx];
                 label  = "Cluster $k  (n=$(length(idx)))",
                 color  = colors[k],
                 markersize = 4,
                 markerstrokewidth = 0)
    end
    return plt
end

"""
    plot_elbo(results; kwargs...)

Line plot of the ELBO history over CAVI iterations.
Useful for diagnosing convergence.
"""
function plot_elbo(results::MixClustResult; kwargs...)
    plot(results.elbo_history;
         xlabel    = "Iteration",
         ylabel    = "ELBO",
         title     = "ELBO Convergence",
         legend    = false,
         linewidth = 2,
         color     = _C_BLUE,
         _THEME...,
         kwargs...)
end

"""
    plot_cluster_sizes(results; kwargs...)

Bar chart of the number of individuals assigned to each active cluster
(hard assignment = argmax of posterior responsibility).
"""
function plot_cluster_sizes(results::MixClustResult; kwargs...)
    n, K   = size(results.w)
    labels = results.labels
    sizes  = [count(==(k), labels) for k in 1:K]
    pcts   = round.(100 .* sizes ./ n, digits=1)
    colors = _cluster_colors(K)

    bar(1:K, sizes;
        xticks    = (1:K, ["C$k\n$(pcts[k])%" for k in 1:K]),
        ylabel    = "Number of individuals",
        legend    = false,
        color     = colors,
        linecolor = :match,
        title     = "Cluster sizes (" * L"\hat{K} = %$K" * ", n = $n)",
        _THEME...,
        kwargs...)
end

"""
    plot_assignment_confidence(results; threshold=0.9, kwargs...)

Histogram of each individual's maximum posterior responsibility max_k w_{ik},
a measure of how crisply each individual is assigned to their cluster.
A vertical line marks `threshold` (default 0.9); the mean confidence and the
fraction of individuals above the threshold are printed in the title.
"""
function plot_assignment_confidence(results::MixClustResult;
                                    threshold::Real=0.9, kwargs...)
    conf     = [maximum(results.w[i, :]) for i in 1:size(results.w, 1)]
    mu       = round(mean(conf), digits=3)
    frac     = round(mean(conf .>= threshold) * 100, digits=1)

    histogram(conf;
              xlabel  = L"\max_k w_{ik}",
              ylabel  = "Count",
              bins    = 30,
              legend  = false,
              color   = _C_BLUE,
              title   = "Assignment confidence (mean=$mu, $(frac)% ≥ $threshold)",
              xlim    = (0, 1),
              _THEME...,
              kwargs...)
    vline!([threshold]; line=(2, :dash, _C_RED), label="threshold " * L"\tau = %$threshold")
    vline!([mu];        line=(2, :dot,  _C_ORANGE), label="mean = $mu")
end

"""
    plot_mixing_weights_evolution(results; kwargs...)
    plot_omega_history(results; kwargs...)

Line plot of the evolution of expected mixture weights 𝔼[ω_k] across CAVI iterations.
Visualizes how empty components are automatically extinguished (weights driven toward zero)
under the sparse Dirichlet prior. Active components at convergence are displayed with solid
lines; extinguished components are shown with dashed lines.
"""
function plot_mixing_weights_evolution(results::MixClustResult; kwargs...)
    omega_hist = results.omega_history
    if isempty(omega_hist)
        K = length(results.u_star)
        w_final = results.u_star ./ sum(results.u_star)
        return bar(1:K, w_final;
                   xlabel = "Cluster",
                   ylabel = L"\mathbb{E}[\omega_k]",
                   title  = "Final Mixture Weights (" * L"\mathbb{E}[\omega_k]" * ")",
                   legend = false,
                   _THEME..., kwargs...)
    end

    T, K = size(omega_hist)
    colors = _cluster_colors(K)
    active_k = Set(unique(results.labels))

    plt = plot(;
        xlabel    = "Iteration",
        ylabel    = L"\mathbb{E}_{q^{(t)}}[\omega_k]",
        title     = "Evolution of Mixture Proportions (K = $K)",
        legend    = :outertopright,
        ylim      = (0.0, 1.05),
        xlim      = (1, max(T, 2)),
        _THEME...,
        kwargs...)

    for k in 1:K
        is_active = k in active_k
        linestyle = is_active ? :solid : :dash
        label_str = is_active ? "Cluster $k" : "Cluster $k (extinct)"
        alpha_val = is_active ? 1.0 : 0.6
        plot!(plt, 1:T, omega_hist[:, k];
              label     = label_str,
              color     = colors[k],
              linestyle = linestyle,
              linewidth = is_active ? 2.5 : 1.5,
              alpha     = alpha_val)
    end

    return plt
end

const plot_omega_history = plot_mixing_weights_evolution

"""
    plot_local_pips(results; feature_names=nothing, only_active=true,
                    color=_IMPORTANCE_COLORMAP, clim=(0.0, 1.0), kwargs...)

Heatmap of per-cluster mean Posterior Inclusion Probabilities (active clusters × p matrix).
Each cell (k, j) shows the soft-assignment-weighted mean PIP of feature j in cluster k.
By default, only active clusters are displayed (`only_active=true`), and the color
palette and scale match `plot_local_eig` (`color=_IMPORTANCE_COLORMAP`, `clim=(0.0, 1.0)`)
to ensure direct, commensurable comparison between relevance metrics.
"""
function plot_local_pips(results::MixClustResult;
                         feature_names=nothing, only_active=true,
                         color=_IMPORTANCE_COLORMAP, clim=(0.0, 1.0), kwargs...)
    n, K  = size(results.w)
    p     = size(results.pip, 2)
    lbls  = _feature_labels(feature_names, p)

    # Weighted mean PIP per cluster: (K × p) matrix
    local_pip = Matrix{Float64}(undef, K, p)
    for k in 1:K
        wk = results.w[:, k]
        total = sum(wk)
        for j in 1:p
            local_pip[k, j] = total > 0 ? dot(wk, results.pip[:, j]) / total : 0.0
        end
    end

    active_k = only_active ? sort(unique(results.labels)) : collect(1:K)
    isempty(active_k) && (active_k = collect(1:K))
    K_act = length(active_k)
    local_pip_act = local_pip[active_k, :]
    cluster_labels = ["C$k" for k in active_k]

    heatmap(lbls, 1:K_act, local_pip_act;
            yticks    = (1:K_act, cluster_labels),
            xlabel    = "Features",
            ylabel    = "Clusters",
            title     = "Per-cluster mean PIPs (" * L"\bar{\gamma}_{k,j}" * ")",
            color     = color,
            clim      = clim,
            _THEME...,
            kwargs...)
end

"""
    plot_local_ehd(results; feature_names=nothing, only_active=true,
                   color=_IMPORTANCE_COLORMAP, clim=(0.0, 1.0), kwargs...)

Heatmap of per-cluster Expected Hellinger Distance (EHD_{k,j} ∈ [0, 1]).
Each cell (k, j) shows the local squared Hellinger distance for feature j in cluster k:
    \\mathbf{EHD}_{k,j} = \\bar{\\gamma}_{k,j} * H^2(f_{k,j}, f_{0,j})
By default, only active clusters are displayed (`only_active=true`), and the color
palette and scale match `plot_local_pips` (`color=_IMPORTANCE_COLORMAP`, `clim=(0.0, 1.0)`)
for direct, commensurable comparison.
"""
function plot_local_ehd(results::MixClustResult;
                        feature_names=nothing, only_active=true,
                        color=_IMPORTANCE_COLORMAP, clim=(0.0, 1.0), kwargs...)
    n, K  = size(results.w)
    p     = length(results.margins)
    lbls  = _feature_labels(feature_names, p)

    local_ehd = compute_local_ehd(results.margins, results.w, results.pip)

    active_k = only_active ? sort(unique(results.labels)) : collect(1:K)
    isempty(active_k) && (active_k = collect(1:K))
    K_act = length(active_k)
    local_ehd_act = local_ehd[active_k, :]
    cluster_labels = ["C$k" for k in active_k]

    heatmap(lbls, 1:K_act, local_ehd_act;
            yticks    = (1:K_act, cluster_labels),
            xlabel    = "Features",
            ylabel    = "Clusters",
            title     = "Per-cluster local EHD (" * L"\mathbf{EHD}_{k,j}" * ")",
            color     = color,
            clim      = clim,
            _THEME...,
            kwargs...)
end

# Backward compatibility alias
const plot_local_eig = plot_local_ehd

