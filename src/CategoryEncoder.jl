module CategoryEncoder

using MLJModelInterface

include("simple_label_encoder.jl")
include("frequency_encoder.jl")
include("target_encoder.jl")

export SimpleLabelEncoder
export FrequencyEncoder
export TargetEncoder

end