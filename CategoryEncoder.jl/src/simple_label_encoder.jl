using MLJModelInterface

struct SimpleLabelEncoder <: MLJModelInterface.Unsupervised
    features::Vector{Symbol}
    missing_value::Union{Int, Nothing}
end

SimpleLabelEncoder(features::Symbol...; missing_value::Union{Int, Nothing} = nothing) = SimpleLabelEncoder(collect(features), missing_value)

function MLJModelInterface.fit(model::SimpleLabelEncoder, verbosity::Int, X)
    mappings = Dict{Symbol, Dict{Any, Int}}()

    for feature in model.features
        col = X[!, feature]
        labels = unique(skipmissing(col))

        mappings[feature] = Dict(label => i for (i, label) in enumerate(labels))
    end

    fitresult = mappings
    cache = nothing
    report = (
        n_features = length(model.features),
        n_labels = Dict(f => length(mappings[f]) for f in model.features)
    )

    return fitresult, cache, report
end

function MLJModelInterface.transform(model::SimpleLabelEncoder, fitresult, X)
    Xnew = deepcopy(X)

    for feature in model.features
        col = Xnew[!, feature]
        mapping = fitresult[feature]

        if model.missing_value === nothing
            Xnew[!, feature] = [ismissing(v) ? missing : get(mapping, v, 0) for v in col]
        else
            mv = model.missing_value
            Xnew[!, feature] = [ismissing(v) ? mv : get(mapping, v, 0) for v in col]
        end
    end

    return Xnew
end