macro meinsum(expr)
    return esc(meinsum(expr))
end


# examples:
#
# @meinsum B[i,j] = A[i,j]
# ->
# for i = 1:size(A,1), j = 1:size(A,2)
#   B[i,j] = A[i,j]
# end
#
# @meinsum B[i,j] = a[i] * a[j]
# ->
# for i = 1:size(a,1), j = 1:size(a,2)
#   B[i,j] = a[i] * a[j]
# end
#
# @meinsum B[i,j] = A[i,j] + a[i] * a[j]
# ->
# @assert axes(A,1) == axes(a,1)
# @assert axes(A,2) == axes(a,1)
# for i = 1:size(a,1), j = 1:size(a,2)
#   B[i,j] = A[i,j] + a[i] * a[j]
# end
#
# @meinsum B[i,j] = A[i,j,k,k]
# @assert axes(A,3) == axes(A,4)
# for i = 1:size(A,1), j = 1:size(A,2), k = size(A,3)
#   B[i,j] = A[i,j,k,k]
# end


function meinsum(expr)
    # 1. setup
    # - get all open indices LHS
    # - get all open indices RHS
    # getopenindices should already check for consistency in a tensor expression
    lhs_idxs = getopenindices(lhs)
    rhs_idxs = getopenindices(rhs)

    # 2. verify
    # - make sure there are no contracted indices on the LHS
    # - make sure there are the same open indices on both sides
    @assert isempty(getcontractedindices(lhs))
    @assert isperm(toperm(lhs_idxs,lhs_idxs), toperm(rhs_idxs,lhs_idxs))
    rhs_cidxs = getcontractedindices(rhs)

    # 3. constract
end


# examples
#
# A[i,j]
# ->
# [i,j]
#
# (a *) A[i,j] * B[j]
# ->
# [i]
#
# a * A[i,j] * B[j] + C[i]
# ->
# [i]
#
# A[i,j] + C[i]
# ->
# error invalid contraction pattern
#
# A[i,j] + c
# ->
# [i,j]


function getopenindices(ex)

    isscalarexpr(ex) && return Any[]
    !istensorexpr(ex) && throw(ArgumentError("not a tensor expression: $ex"))

    idxs = Any[]
    idxs = if istensor(ex)
        getindices(ex)
    elseif ex.head === :call && ex.args[1] in (:+,:-)
        all_subidxs = [ getopenindices(a) for a in ex.args[2:end] ]
        catalog = permutation_catalog(first(all_subidxs))
        if !all(sidxs -> ispermutation(sidxs, catalog), all_subidxs)
            throw(ArgumentError("inconsistent contraction pattern: $ex"))
        end
        append!(idxs, first(all_subidxs))
    elseif ex.head === :call && ex.args[1] === :*
        append!(idxs,
                reduce(vcat, [ istensor(a) ? getindices(a) : [] for a in ex.args[2:end] ])
               )
    elseif ex.head === :call && ex.args[1] === :/
        append!(idxs, getindices(ex.args[2]))
    # elseif ex.head === :call && ex.args[1] === :\
    #     append!(idxs, getindices(ex.args[3]))
    else
        throw(ex)
    end

    # filter contracted indices (=^= appear exactly twice)
    # TODO Enforce exactly twice somewhere ...
    dups = [ i for i in idxs if count(j -> i == j, idxs) > 1 ]
    foreach(d -> filter!(i -> i != d, idxs), dups)

    return idxs
end




# # Taken from TensorOperations.jl
# isassignment(ex) = false
# isassignment(ex::Expr) = (ex.head == :(=) || ex.head == :(+=) || ex.head == :(-=))
#
# # Taken from TensorOperations.jl
# function getlhs(ex::Expr)
#     if isassignment(ex) && length(ex.args) == 2
#         return ex.args[1]
#     else
#         throw(ArgumentError("invalid assignment or definition $ex"))
#     end
# end
# function getrhs(ex::Expr)
#     if isassignment(ex) && length(ex.args) == 2
#         return ex.args[2]
#     else
#         throw(ArgumentError("invalid assignment or definition $ex"))
#     end
# end
#
#


# function decomposetensor(ex)
#     if istensor(ex)
#         return (ex.args[1], ex.args[2:end])
#     else
#         throw(ArgumentError("not a vlid tensor: $ex"))
#     end
# end
#
#
# Remaining couplings to TensorOperations
# istensor
# isgeneraltensor
# decomposetensor
# istensorexpr : a*A[i] + b*B[j] is true, a*A[i] + b is false, a * A[i] * B[i] is true
# isscalarexpr : anything without brackets is a scalar expression
# gettensorobjects
# getallindices
# getindices
# gettensors
# @tensor



# tensor =^= a single array
istensor(ex::Symbol) = false
istensor(ex::Number) = false
istensor(ex) = ex.head === :ref && length(ex.args) >= 2

# general tensor =^= a single array with at most scalar coefficients
function isgeneraltensor(ex)
    istensor(ex) && return true
    ex.head === :call && length(ex.args) >= 3 && ex.args[1] == :* &&
        count(a -> istensor(a), ex.args[2:end]) == 1 &&
        count(a -> isscalarexpr(a), ex.args[2:end]) == length(ex.args)-2 && return true
    ex.head === :call && length(ex.args) == 3 && ex.args[1] == :/ && istensor(ex.args[2]) && !istensor(ex.args[3]) && return true
    return false
end
isgeneraltensor(ex::Symbol) = false
isgeneraltensor(ex::Number) = false


# any expression which involves (general)tensors combined with any of the +,-,*,/ operators
function istensorexpr(ex)
    isgeneraltensor(ex) && return true
    ex.head === :call && length(ex.args) >= 3 && ex.args[1] === :* &&
        all(a -> istensorexpr(a) || isscalarexpr(a), ex.args[2:end]) &&
        count(a -> istensorexpr(a), ex.args[2:end]) >= 1 && return true
    ex.head === :call && length(ex.args) == 3 && ex.args[1] === :/ &&
        isscalarexpr(ex.args[3]) && return istensorexpr(ex.args[2])
    ex.head === :call && length(ex.args) >= 3 && ex.args[1] in (:+,:-) &&
        all(a -> istensorexpr(a), ex.args[2:end]) && return true
    return false
end
istensorexpr(ex::Symbol) = false
istensorexpr(ex::Number) = false


# anything with indices is not a scalar expression
isscalarexpr(ex) = !hasindices(ex)


# Any expression brackets ([ ] =^= ex.head === :rea) is considered to have indices, even :(A[])
function hasindices(ex)
    ex.head === :ref && return true
    ex.head === :call && length(ex.args) >= 3 && any(a -> hasindices(a), ex.args[2:end]) && return true
    return false
end
hasindices(ex::Symbol) = false
hasindices(ex::Number) = false



function getindices(ex::Expr)
    if istensor(ex)
        return ex.args[2:end]
    elseif isgeneraltensor(ex)
        idxs = reduce(vcat, [ getindices(a) for a in ex.args[2:end] ] )
    else
        throw(ArgumentError("not a tensor: $ex"))
    end
end
getindices(ex::Symbol) = [ ex ]
getindices(ex::Int) = [ ex ]


# variable =^= any Symbol that is used in an expression with head===:call
function getscalars(ex)
    vars = Symbol[]
    _getscalars!(ex, vars)
    return vars
end


_getscalars!(s::Symbol, vars) = push!(vars, s)
_getscalars!(s::Number, vars) = nothing
function _getscalars!(ex::Expr, vars)
    if ex.head === :call
        foreach(ex.args[2:end]) do a
            _getscalars!(a, vars)
        end
    else
        error("Failed to extract variables from '$ex'")
    end
    return
end
