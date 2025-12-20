using StatsBase
using DataFrames
using MLJModelInterface

struct TargetEncoder <: MLJModelInterface.Deterministic
    features::Vector{Symbol}
end

TargetEncoder(features::Symbol...) = TargetEncoder(collect(features))

function MLJModelInterface.fit(model::TargetEncoder, verbosity::Int, X, y)
    mappings = Dict{Symbol, Dict{Any, Float64}}()
    
    for feature in model.features
        col_data = X[!, feature]
        mask = .!ismissing.(col_data)
        filtered_feature = col_data[mask]
        filtered_y = y[mask]

        unique_labels = unique(filtered_feature)
        mappings[feature] = Dict(
            label => mean(filtered_y[filtered_feature .== label])
            for label in unique_labels
        )
    end

    cache = Dict(f => mean(values(mappings[f])) for f in model.features)
    fitresult = (mappings = mappings, cache = cache)
    report = (
        n_features = length(model.features),
        n_labels = Dict(f => length(mappings[f]) for f in model.features)
    )

    return fitresult, nothing, report
end

function MLJModelInterface.transform(model::TargetEncoder, fitresult, X)
    Xnew = copy(X)
    
    for feature in model.features
        mapping = fitresult.mappings[feature]
        default = fitresult.cache[feature]
        
        Xnew[!, feature] = map(Xnew[!, feature]) do v 
            ismissing(v) ? default : get(mapping, v, default) 
        end
    end
    
    return Xnew
end