"""
    load_heart_disease() -> NamedTuple

Load the UCI Heart Disease (Cleveland) dataset bundled with the package.

Returns a named tuple `(data, labels, feature_names)`:
- `data::Vector{Any}` — 13-feature dataset ready to pass directly to `mixClust`
- `labels::Vector{Int}` — binary disease status: 0 = healthy, 1 = disease (any degree)
- `feature_names::Vector{String}` — human-readable feature names

The 13 features, in order:

| # | Name | Type | Encoding |
|:--|:-----|:-----|:---------|
| 1 | `age` | continuous | standardized |
| 2 | `sex` | binary | one-hot (2 categories) |
| 3 | `cp` | chest pain type | one-hot (4 categories) |
| 4 | `trestbps` | resting blood pressure | standardized |
| 5 | `chol` | serum cholesterol | standardized |
| 6 | `fbs` | fasting blood sugar > 120 mg/dl | one-hot (2 categories) |
| 7 | `restecg` | resting ECG result | one-hot (3 categories) |
| 8 | `thalach` | maximum heart rate achieved | standardized |
| 9 | `exang` | exercise-induced angina | one-hot (2 categories) |
| 10 | `oldpeak` | ST depression induced by exercise | standardized |
| 11 | `slope` | slope of peak exercise ST segment | one-hot (3 categories) |
| 12 | `ca` | number of major vessels coloured by fluoroscopy | one-hot (4 categories) |
| 13 | `thal` | thalassemia | one-hot (3 categories) |

# Example

```julia
hd = load_heart_disease()
results = mixClust(hd.data, 10;
                   model_setting = SFRM(),
                   u0            = 0.01,
                   max_iter      = 2000,
                   tol           = 1e-5)
eig    = compute_eig(results.margins, results.w, results.pip)
active = filter_features(eig, 0.10)
println("Active features: ", hd.feature_names[active])
```

# Source

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988).
*Heart Disease* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C52P4X>
"""
function load_heart_disease()
    path = joinpath(pkgdir(NaiveMFLFRMixDClust), "data", "uci_heart_disease.csv")
    if !isfile(path)
        path = joinpath(pkgdir(NaiveMFLFRMixDClust), "experiments", "data", "uci_heart_disease.csv")
    end
    raw, _ = readdlm(path, ',', header = true)
    n = size(raw, 1)

    # Parse column j as a Float64 vector, handling both numeric and string cells
    fcol(j) = Float64[isa(raw[i, j], Number) ? Float64(raw[i, j]) :
                      parse(Float64, string(raw[i, j])) for i in 1:n]

    standardize(v) = (v .- mean(v)) ./ std(v)

    function onehot(val::Float64, levels::Vector{Float64})
        arr = zeros(Int, length(levels))
        idx = findfirst(==(val), levels)
        isnothing(idx) || (arr[idx] = 1)
        return arr
    end

    data = Vector{Any}(undef, 13)
    data[1]  = standardize(fcol(1))                                        # age
    data[2]  = [onehot(x, [0.0, 1.0])             for x in fcol(2)]       # sex
    data[3]  = [onehot(x, [1.0, 2.0, 3.0, 4.0])  for x in fcol(3)]       # cp
    data[4]  = standardize(fcol(4))                                        # trestbps
    data[5]  = standardize(fcol(5))                                        # chol
    data[6]  = [onehot(x, [0.0, 1.0])             for x in fcol(6)]       # fbs
    data[7]  = [onehot(x, [0.0, 1.0, 2.0])        for x in fcol(7)]       # restecg
    data[8]  = standardize(fcol(8))                                        # thalach
    data[9]  = [onehot(x, [0.0, 1.0])             for x in fcol(9)]       # exang
    data[10] = standardize(fcol(10))                                       # oldpeak
    data[11] = [onehot(x, [1.0, 2.0, 3.0])        for x in fcol(11)]      # slope
    data[12] = [onehot(x, [0.0, 1.0, 2.0, 3.0])  for x in fcol(12)]      # ca
    data[13] = [onehot(x, [3.0, 6.0, 7.0])        for x in fcol(13)]      # thal

    labels = Int[x == 0 ? 0 : 1 for x in fcol(14)]

    feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                     "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

    return (data = data, labels = labels, feature_names = feature_names)
end

"""
    simulate_synthetic_cohort(; seed=2026, n=150) -> NamedTuple

Generate the synthetic clinicogenomic patient cohort used in the paper.

Returns a named tuple `(data, labels, feature_names)`:
- `data::Vector{Any}` — 7-feature dataset ready to pass directly to `mixClust`
- `labels::Vector{Int}` — true cluster assignment (1, 2, or 3)
- `feature_names::Vector{String}` — human-readable feature names

The simulation is fully reproducible: the same `seed` always produces the same dataset.
With the default `seed=2026` and `n=150`, the dataset matches the example in the paper
(50 individuals per subtype, 3 informative features + 1 categorical + 2 noise features).

# The 7 features

| # | Name | Type | Signal |
|:--|:-----|:-----|:-------|
| 1 | `age` | Gaussian (standardized) | mean differs across all 3 subtypes |
| 2 | `bmi` | Gaussian (standardized) | subtype 3 has elevated BMI |
| 3 | `mutations` | Poisson | subtype 2 is hypermutated (λ=18 vs 2/6) |
| 4 | `tumour_size` | Gamma | rate differs across subtypes |
| 5 | `histology` | Multinomial (3 categories) | distinct profile per subtype |
| 6 | `noise_cont` | Gaussian (standardized) | uninformative |
| 7 | `noise_count` | Poisson (λ=4) | uninformative |

# Example

```julia
cohort = simulate_synthetic_cohort()
results = mixClust(cohort.data, 10;
                   model_setting = LFRM(),
                   u0            = 0.01,
                   max_iter      = 500,
                   tol           = 1e-4,
                   n_init        = 10,
                   max_iter_init = 10)

println("Estimated clusters: ", size(results.w, 2))   # → 3
eig    = compute_eig(results.margins, results.w, results.pip)
active = filter_features(eig, 0.10)
println("Active features: ", cohort.feature_names[active])
# → ["age", "mutations", "tumour_size", "histology"]
```
"""
function simulate_synthetic_cohort(; seed::Integer = 2026, n::Integer = 150)
    rng = Random.MersenneTwister(seed)

    n_per = n ÷ 3
    labels = vcat(fill(1, n_per), fill(2, n_per), fill(3, n - 2 * n_per))

    # Poisson variate via Knuth algorithm
    function _rpois(λ)
        L = exp(-λ); k = 0; p = 1.0
        while p > L
            k += 1
            p *= rand(rng)
        end
        return k - 1
    end

    # Categorical draw
    function _rand_cat(probs)
        u = rand(rng); cum = 0.0
        for (i, p) in enumerate(probs)
            cum += p
            u <= cum && return i
        end
        return length(probs)
    end

    # Gamma(shape, rate) via sum of exponentials — exact for integer shape
    _rand_gamma(shape, rate) =
        -sum(log(rand(rng)) for _ in 1:round(Int, shape)) / rate

    # Feature 1 – Age (Gaussian, standardized)
    age_raw = [labels[i] == 1 ? 45.0 + 8randn(rng) :
               labels[i] == 2 ? 62.0 + 8randn(rng) : 55.0 + 8randn(rng) for i in 1:n]
    age = (age_raw .- mean(age_raw)) ./ std(age_raw)

    # Feature 2 – BMI (Gaussian, standardized)
    bmi_raw = [labels[i] == 3 ? 31.0 + 4randn(rng) : 24.0 + 4randn(rng) for i in 1:n]
    bmi = (bmi_raw .- mean(bmi_raw)) ./ std(bmi_raw)

    # Feature 3 – Mutation count (Poisson)
    mutations = Float64[labels[i] == 1 ? _rpois(2.0) :
                        labels[i] == 2 ? _rpois(18.0) : _rpois(6.0) for i in 1:n]

    # Feature 4 – Tumour size (Gamma, shape=3; rate differs by subtype)
    tumour_size = [labels[i] == 1 ? _rand_gamma(3, 0.3) :
                   labels[i] == 2 ? _rand_gamma(3, 0.1) : _rand_gamma(3, 0.2) for i in 1:n]

    # Feature 5 – Histological subtype (Multinomial, 3 categories)
    cat_probs = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]
    histology = [[_rand_cat(cat_probs[labels[i]]) == c ? 1 : 0 for c in 1:3] for i in 1:n]

    # Features 6-7 – Noise (uninformative)
    noise_cont  = randn(rng, n)
    noise_count = Float64[_rpois(4.0) for _ in 1:n]

    data = Any[age, bmi, mutations, tumour_size, histology, noise_cont, noise_count]
    feature_names = ["age", "bmi", "mutations", "tumour_size",
                     "histology", "noise_cont", "noise_count"]

    return (data = data, labels = labels, feature_names = feature_names)
end
