module TensorComponents

using Reexport

using Dates
@reexport using LinearAlgebra
using MacroTools
using Printf
using Random
using RowEchelon
@reexport using SymEngine


export @components, @generate_code, @test_code, adjugate


include("utils.jl")
include("symbolictensor.jl")
include("index.jl")
include("meinsum.jl")
include("components.jl")
include("codegen.jl")


end
