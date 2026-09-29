using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
include("real_data_analysis.jl")

function export_data()
    n = 300
    data, true_z = generate_clinicogenomic_dataset(n; seed=789)
    
    feature_names = [
        "SBP", "Cholesterol", "BMI", "Age",
        "SomaticMut", "TIL", "Hosp",
        "Stage", "HistType", "Genotype",
        "SurvivalTime", "TimeToFailure", "Ventilation"
    ]
    
    out_dir = joinpath(@__DIR__, "../data")
    mkpath(out_dir)
    
    open(joinpath(out_dir, "clinicogenomic_cohort.csv"), "w") do io
        write(io, join(["True_Subtype"; feature_names], ",") * "\n")
        for i in 1:n
            row = [string(true_z[i])]
            for j in 1:13
                push!(row, string(data[j][i]))
            end
            write(io, join(row, ",") * "\n")
        end
    end
    println("Dataset exported to experiments/data/clinicogenomic_cohort.csv")
end

export_data()
