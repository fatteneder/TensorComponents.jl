struct Rule
    sumedidx::Vector{Int64}
    freeidx::Vector{Int64}
    r::Function
end

function impose_rule!(a::AbstractArray, rule::Rule)
    cidxs = CartesianIndices(a)
    for i = 1:length(a)
        idxs = cidxs[i]
        prefactor, r_idxs = rule.r(Tuple(idxs)...)
        a[r_idxs...] = prefactor * a[Tuple(idxs)...]
    end
end

function isvalidslot(s, a::AbstractArray)
    return 1 <= s <= ndims(a)
end

function symmetrize!(a::AbstractArray{T,2}, slot1::Int64, slot2::Int64) where T
    if !isvalidslot(slot1, a) || !isvalidslot(slot2, a)
        error("Slot indices '$slot1,$slot1' exceed rank of 2-array")
    end
    s1, s2 = size(a)
    for i1 = 1:s1, i2 = i1+1:s2
        a[i2,i1] = a[i1,i2]
    end
end

function antisymmetrize!(a::AbstractArray{T,2}, slot1::Int64, slot2::Int64) where T
    if !isvalidslot(slot1, a) || !isvalidslot(slot2, a)
        error("Slot indices '$slot1,$slot1' exceed rank of 2-array")
    end
    s1, s2 = size(a)
    for i1 = 1:s1, i2 = i1:s2
        a[i2,i1] = i1 == i2 ? 0 : -a[i1,i2]
    end
end

function permute_slots_to_start(a::AbstractArray, slot1, slot2)
    # construct permutation indices such that slot indices appear first
    slots = [ i for i = 1:ndims(a) ]
    slots[slot1], slots[1] = slots[1], slots[slot1]
    slots[slot2], slots[2] = slots[2], slots[slot2]
    # permutation view
    pa = PermutedDimsArray(a, slots)
    return pa
end

function permute_slots_to_start(a::AbstractArray, slot1::NTuple{N}, slot2::NTuple{N}) where N
    # construct permutation indices such that slot1 indices appear first, then slot2 indidces
    # and then any remaining indices
    slots  = [ i for i = 1:ndims(a) ]
    vslot1 = [ s1 for s1 in slot1 ]
    vslot2 = [ s2 for s2 in slot2 ]
    diffslots = setdiff(slots, vslot1, vslot2)
    perm = vcat(vslot1, vslot2, diffslots)
    pa = PermutedDimsArray(a, perm)
    return pa
end

function symmetrize!(a::AbstractArray, slot1::Int64, slot2::Int64)

    if !isvalidslot(slot1, a) || !isvalidslot(slot2, a)
        error("Slot indices '$slot1,$slot1' exceed rank of $(ndims(a))-array")
    end

    pa = permute_slots_to_start(a, slot1, slot2)

    sz = size(a)
    s1, s2 = sz[slot1], sz[slot2]
    nresdims = ndims(a) - 2
    cols = (Colon() for _ = 1:nresdims)
    for i1 = 1:s1, i2 = i1+1:s2
        pa[i2,i1,cols...] .= pa[i1,i2,cols...]
    end

    return
end

function symmetrize!(a::AbstractArray, slot1::NTuple{N,Int64}, slot2::NTuple{N,Int64}) where N

    if any(s -> !isvalidslot(s, a), slot1) || any(s -> !isvalidslot(s, a), slot2)
        error("Slot indices '$slot1,$slot1' exceed rank of $(ndims(a))-array")
    end

    pa = permute_slots_to_start(a, slot1, slot2)

    sz = size(a)
    cidxs1 = CartesianIndices(Tuple(sz[s] for s in slot1))
    cidxs2 = CartesianIndices(Tuple(sz[s] for s in slot2))
    nresdims = ndims(a) - length(slot1) - length(slot2)
    if nresdims == 0
        for c1 in cidxs1, c2 in cidxs2
            tc1, tc2 = Tuple(c1), Tuple(c2)
            if all(tc1 .<= tc2)
                eqs = tc1 .== tc2
                all(eqs) && continue
                es = [ e for e in eqs ]
                !all(es .= sort(es)) && continue
                pa[tc2...,tc1...] = pa[tc1...,tc2...]
            end
        end
    else
        cols = (Colon() for _ = 1:nresdims)
        for c1 in cidxs1, c2 in cidxs2
            tc1, tc2 = Tuple(c1), Tuple(c2)
            if all(tc1 .<= tc2) && !all(tc1 .== tc2)
                pa[tc2...,tc1...,cols...] .= pa[tc1...,tc2...,cols...]
            end
        end
    end

    return
end

function antisymmetrize!(a::AbstractArray, slot1::Int64, slot2::Int64)

    if !isvalidslot(slot1, a) || !isvalidslot(slot2, a)
        error("Slot indices '$slot1,$slot1' exceed rank of $(ndims(a))-array")
    end

    pa = permute_slots_to_start(a, slot1, slot2)

    sz = size(a)
    s1, s2 = sz[slot1], sz[slot2]
    nresdims = ndims(a) - 2
    cols = ( Colon() for _ = 1:nresdims)
    for i1 = 1:s1, i2 = i1:s2
        pa[i2,i1,cols...] .= i1 == i2 ? 0 : -pa[i1,i2,cols...]
    end

    return
end

struct SymbolicTensor{N}
    idxs::NTuple{N,UnitRange{Int64}}
    rules::Vector{Rule}
end

function SymbolicTensor(idxs::Int64...)
    all(i -> i > 0, idxs) || error("Index ranges must be positive")
    return SymbolicTensor(to_range.(idxs), Rule[])
end

function SymbolicTensor(idxs::Index...)
    rngs = [ i.rng for i in idxs ]
    return SymbolicTensor(rngs)
end

to_range(i::AbstractRange) = i
to_range(i::Int) = 1:i

Base.size(a::SymbolicTensor) = length.(a.idxs)

function Base.collect(a::SymbolicTensor, head::Symbol)
    dims = size(a)
    cidxs = CartesianIndices(a.idxs)
    rawt = [ symbols("$(head)$(join(Tuple(c)))") for c in cidxs ]
    t = reshape(rawt, dims)
    # for r in a.rules
    #     impose_rule!(t, r)
    # end
    return t
end

macro collect(expr...)
    return esc(_collect(expr))
end

function _collect(expr)
    row = [ :(collect($ex,Symbol($(string(ex))))) for ex in expr]
    return quote $(row...) end
end

Base.push!(a::SymbolicTensor, r::Rule) = push!(a.rules, r)
