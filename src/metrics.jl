"""
    adjusted_rand_index(labels_true, labels_pred) -> Float64

Adjusted Rand Index between two label vectors. Returns 1.0 for perfect agreement,
0.0 in expectation for random labellings, and can be negative.
"""
function adjusted_rand_index(labels_true::Vector{Int}, labels_pred::Vector{Int})
    n = length(labels_true)
    classes  = unique(labels_true)
    clusters = unique(labels_pred)

    contingency = zeros(Int, length(classes), length(clusters))
    for i in 1:n
        c = findfirst(==(labels_true[i]),  classes)
        k = findfirst(==(labels_pred[i]), clusters)
        contingency[c, k] += 1
    end

    sum_ij = sum(x -> x * (x - 1) / 2, contingency)
    sum_i  = sum(x -> x * (x - 1) / 2, sum(contingency, dims=2))
    sum_j  = sum(x -> x * (x - 1) / 2, sum(contingency, dims=1))
    total  = n * (n - 1) / 2

    expected = (sum_i * sum_j) / total
    maximum  = 0.5 * (sum_i + sum_j)

    maximum == expected && return 1.0
    return (sum_ij - expected) / (maximum - expected)
end
