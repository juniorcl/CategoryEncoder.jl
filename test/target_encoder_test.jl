include("../src/CategoryEncoder.jl")

using MLJ
using DataFrames
using .CategoryEncoder

X = DataFrame(
    gender  = ["M", "F", "M", missing],
    Country = ["BR", "US", "BR", "FR"],
    age     = [23, 35, 29, 41]
)

y = Vector{Int64}([1, 0, 1, 0])

encoder = CategoryEncoder.TargetEncoder(:gender, :Country)
mach = machine(encoder, X, y) |> fit!

X_enc = MLJ.transform(mach, X)