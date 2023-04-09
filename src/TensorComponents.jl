module TensorComponents


using LinearAlgebra
using MacroTools
using RowEchelon
using SymEngine
using TensorOperations
import TensorOperations: Strided

const TO = TensorOperations


export @components
include("utils.jl")
include("symbolictensor.jl")
include("index.jl")
include("components.jl")


end
