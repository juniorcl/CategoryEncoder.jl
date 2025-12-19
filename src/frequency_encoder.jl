using MLJModelInterface
using StatsBase

struct FrequencyEncoder <: MLJModelInterface.Unsupervised
    features::Vector{Symbol}
    missing_value::Union{Float64, Nothing}
end

FrequencyEncoder(features::Symbol...; missing_value::Union{Float64, Nothing}=nothing) =
    FrequencyEncoder(collect(features), missing_value)

function MLJModelInterface.fit(model::FrequencyEncoder, verbosity::Int, X)
    mappings = Dict{Symbol, Dict{Any, Float64}}()

    for feature in model.features
        col = X[!, feature]

        counts = countmap(skipmissing(col))
        isempty(counts) && continue

        total = sum(values(counts))
        LabelType = eltype(skipmissing(col))

        mappings[feature] = Dict{LabelType, Float64}(
            k => v / total for (k, v) in counts
        )
    end

    fitresult = mappings
    cache = nothing
    report = (
        n_features = length(model.features),
        n_labels = Dict(
            f => length(get(mappings, f, Dict{Any, Float64}()))
            for f in model.features
        )
    )

    return fitresult, cache, report
end

function MLJModelInterface.transform(model::FrequencyEncoder, fitresult, X)
    Xnew = copy(X)
    default = model.missing_value === nothing ? missing : model.missing_value

    for feature in model.features
        col = Xnew[!, feature]
        mapping = fitresult[feature]

        Xnew[!, feature] = [
            ismissing(v) ? default : get(mapping, v, 0.0)
            for v in col
        ]
    end

    return Xnew
end