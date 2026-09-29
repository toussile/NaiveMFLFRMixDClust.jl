using Downloads, CSV, DataFrames, Statistics

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
dest = "experiments/data/uci_heart_disease.csv"

# Download the file
raw_data = String(take!(Downloads.download(url, IOBuffer())))
lines = split(raw_data, "\n", keepempty=false)

# Filter out rows with '?'
clean_lines = filter(l -> !occursin("?", l), lines)

open(dest, "w") do io
    # Write header
    println(io, "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num")
    for l in clean_lines
        println(io, l)
    end
end

df = CSV.read(dest, DataFrame)
println("Saved processed UCI dataset to $dest")
println("Shape: ", size(df))
