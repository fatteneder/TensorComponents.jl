module TensorComponents

using LinearAlgebra
using SymEngine
using TensorOperations
import TensorOperations: Strided

const TO = TensorOperations

include("index.jl")
include("symbolictensor.jl")
include("components.jl")

export @components, @expand

end # module TensorComponents
