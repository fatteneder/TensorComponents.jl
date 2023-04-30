module TensorComponents


using Dates
using LinearAlgebra
using MacroTools
using Printf
using Random
using RowEchelon
using SymEngine


export @components, @generate_code, @test_code, adjugate


include("utils.jl")
include("symbolictensor.jl")
include("index.jl")
include("meinsum.jl")
include("components.jl")
include("codegen.jl")


end
