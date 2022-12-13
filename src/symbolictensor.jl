struct SymbolicTensor{N} <: AbstractArray{Basic,N}
    t::Array{Basic,N}
end

function SymbolicTensor(head::Symbol, idxrange::AbstractRange{Int}...)
    dims = length.(idxrange)
    all(l -> l > 0, dims) || error("Encountered empty index ranges for SymbolicTensor $head")
    cidxs = CartesianIndices(idxrange)
    t = [ symbols("$(head)$(join(Tuple(c)))") for c in cidxs ]
    return SymbolicTensor(reshape(t, dims))
end

function SymbolicTensor(head::Symbol, idxs...)
    return SymbolicTensor(head, to_range.(idxs)...)
end

to_range(i::AbstractRange) = i
to_range(i::Int) = 1:i

Base.size(a::SymbolicTensor) = size(a.t)
Base.getindex(a::SymbolicTensor, i::Int) = getindex(a.t, i)
Base.getindex(a::SymbolicTensor{N}, I::Vararg{Int,N}) where N = getindex(a.t, I...)
Base.setindex!(a::SymbolicTensor, v, i::Int) = setindex!(a.t, v, i)
Base.setindex!(a::SymbolicTensor{N}, v, I::Vararg{Int,N}) where N = setindex!(a.t, v, I...)
# Base.similar(a::SymbolicTensor{N}) where N = SymbolicTensor{N}(zeros(SymEngine.Basic, size(a)))
# Base.similar(a::SymbolicTensor{N}, ::Type{S}) where {N,S} = SymbolicTensor{N}(zeros(SymEngine.Basic, size(a)))
# Base.similar(a::SymbolicTensor{N}, dims::Dims) where N = SymbolicTensor{N}(zeros(SymEngine.Basic, dims))
# Base.similar(a::SymbolicTensor{N}, ::Type{S}, dims::Dims) where {N,S} = SymbolicTensor{N}(zeros(SymEngine.Basic, dims))
Base.promote_rule(::Type{Bool}, ::Type{SymEngine.Basic}) = SymEngine.Basic

Strided.StridedView(t::TensorComponents.SymbolicTensor{N}) where N = Strided.StridedView(t.t)

function SymbolicTensor(head::Symbol, idxs::Index...)
    rngs = [ i.rng for i in idxs ]
    return SymbolicTensor(head, rngs...)
end

Base.view(t::SymbolicTensor, idxs::Index...) = view(t.t, (i.rng for i in idxs)...)

LinearAlgebra.det(t::SymbolicTensor{2}) = det(t.t)

components(t::SymbolicTensor) = t.t[:]
