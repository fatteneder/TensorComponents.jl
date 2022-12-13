struct SymbolicTensor{N} <: AbstractArray{Basic,N}
    t::Array{Basic,N}
end

function SymbolicTensor(::Type{T}, idxrange::AbstractRange{Int}...) where T
    dims = length.(idxrange)
    # all(l -> l > 0, dims) || error("Encountered empty index ranges")
    # cidxs = CartesianIndices(idxrange)
    # t = [ symbols("$(head)$(join(Tuple(c)))") for c in cidxs ]
    # return SymbolicTensor(reshape(t, dims))
    t = zeros(T, dims)
    return SymbolicTensor(t)
end

function SymbolicTensor(idxrange::AbstractRange{Int}...)
    return SymbolicTensor(Int, idxrange...)
end

function SymbolicTensor(idxs::Int...)
    return SymbolicTensor(to_range.(idxs)...)
end

function SymbolicTensor(a::AbstractArray{T,N}) where {T,N}
    return SymbolicTensor{N}(a)
end

function SymbolicTensor(idxs::Index...)
    rngs = [ i.rng for i in idxs ]
    return SymbolicTensor(rngs...)
end

to_range(i::AbstractRange) = i
to_range(i::Int) = 1:i

# implement Array Interface
Base.size(a::SymbolicTensor) = size(a.t)
Base.getindex(a::SymbolicTensor, i::Int) = getindex(a.t, i)
Base.getindex(a::SymbolicTensor{N}, I::Vararg{Int,N}) where N = getindex(a.t, I...)
Base.setindex!(a::SymbolicTensor, v, i::Int) = setindex!(a.t, v, i)
Base.setindex!(a::SymbolicTensor{N}, v, I::Vararg{Int,N}) where N = setindex!(a.t, v, I...)

# needed to get @tensor to work
Strided.StridedView(t::TensorComponents.SymbolicTensor{N}) where N = Strided.StridedView(t.t)

Base.view(t::SymbolicTensor, idxs::Index...) = view(t.t, (i.rng for i in idxs)...)

components(t::SymbolicTensor) = t.t[:]

# extend math operations
LinearAlgebra.det(t::SymbolicTensor{2}) = det(t.t)

# overrides so that operations involving SymbolicTensors all return a SymbolicTensor again
Base.similar(a::SymbolicTensor) = SymbolicTensor(zeros(SymEngine.Basic, size(a)))
Base.similar(a::SymbolicTensor, ::Type{S}) where S = SymbolicTensor(zeros(SymEngine.Basic, size(a)))
Base.similar(a::SymbolicTensor, dims::Dims) = SymbolicTensor(zeros(SymEngine.Basic, dims))
Base.similar(a::SymbolicTensor, ::Type{S}, dims::Dims) where S = SymbolicTensor(zeros(SymEngine.Basic, dims))
Base.promote_rule(::Type{Bool}, ::Type{SymEngine.Basic}) = SymEngine.Basic
Base.promote_rule(::Type{T}, ::Type{SymbolicTensor}) where T<:AbstractArray = SymbolicTensor
Base.:(+)(s::SymbolicTensor{N}, m::AbstractArray) where N = SymbolicTensor(s.t + m)
Base.:(-)(s::SymbolicTensor{N}, m::AbstractArray) where N = SymbolicTensor(s.t - m)
Base.:(*)(s::SymbolicTensor{N}, m::AbstractMatrix) where N = SymbolicTensor(s.t * m)
Base.:(+)(m::AbstractArray, s::SymbolicTensor{N}) where N = SymbolicTensor(m + s.t)
Base.:(-)(m::AbstractArray, s::SymbolicTensor{N}) where N = SymbolicTensor(m - s.t)
Base.:(*)(m::AbstractMatrix, s::SymbolicTensor{N}) where N = SymbolicTensor(m * s.t)
