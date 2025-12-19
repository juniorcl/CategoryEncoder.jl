include("../src/CategoryEncoder.jl")

using MLJ
using DataFrames
using .CategoryEncoder

X = DataFrame(
    gender  = ["M", "F", "M", missing],
    Country = ["BR", "US", "BR", "FR"],
    age     = [23, 35, 29, 41]
)

encoder = CategoryEncoder.FrequencyEncoder(:gender, :Country)
mach = machine(encoder, X) |> fit!

X_enc = CategoryEncoder.transform(mach, X)

first(X_enc, 1)

last(X_enc, 1)