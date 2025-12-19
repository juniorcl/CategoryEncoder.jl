module CategoryEncoder

using MLJModelInterface

include("simple_label_encoder.jl")

include("frequency_encoder.jl")

export SimpleLabelEncoder

export FrequencyEncoder

end