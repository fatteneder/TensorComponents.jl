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


# function getallindices2(ex)
#
#     isscalarexpr(ex) && return Any[], Any[]
#     !istensorexpr(ex) && throw(ArgumentError("not a tensor expression: $ex"))
#
#     idxs, contracted = Any[], Any[]
#     idxs = if istensor(ex)
#         getallindices(ex)
#     elseif ex.head === :call && ex.args[1] in (:+,:-)
#         allsubidxs = [ getallindices2(a) for a in ex.args[2:end] ]
#         allopensubidxs = [ idxs[1] for idxs in allsubidxs ]
#         catalog = permutation_catalog(first(allopensubidxs))
#         if !all(sidxs -> ispermutation(sidxs, catalog), allopensubidxs)
#             throw(ArgumentError("inconsistent contraction pattern: $ex"))
#         end
#         append!(contracted, reduce(vcat, a[2] for a in allsubidxs))
#         append!(idxs, first(allopensubidxs))
#     elseif ex.head === :call && ex.args[1] === :*
#         append!(idxs,
#                 reduce(vcat, [ istensorexpr(a) ? getallindices(a) : [] for a in ex.args[2:end] ])
#                )
#     elseif ex.head === :call && ex.args[1] === :/
#         append!(idxs, getallindices(ex.args[2]))
#     # elseif ex.head === :call && ex.args[1] === :\
#     #     append!(idxs, getindices(ex.args[3]))
#     else
#         throw(ex)
#     end
#
#     # @show ex
#     # display(idxs)
#     # display(contracted)
#
#     # filter contracted indices (=^= appear exactly twice)
#     # TODO Enforce exactly twice somewhere ...
#     # contracted = [ i for i in idxs if count(j -> i == j, idxs) > 1 ]
#     append!(contracted, i for i in idxs if count(j -> i == j, idxs) > 1)
#     foreach(d -> filter!(i -> i != d, idxs), contracted)
#     unique!(contracted)
#     # display(idxs)
#     # display(contracted)
#
#     return idxs, contracted
# end


# TODO Figure out what these should do
# - getallindices ... all indices, but unique
# - getindices ... only open indices
# - getopenindices
# - getcontractedindices

# getopenindices(ex) = getallindices(ex)[1]
# getcontractedindices(ex) = getallindices(ex)[2]


getallindices(ex) = unique(_getallindices(ex))
function _getallindices(ex::Expr)
    if isscalarexpr(ex)
        return []
    elseif istensor(ex)
        return ex.args[2:end]
    elseif isgeneraltensor(ex)
        return reduce(vcat, [ _getallindices(a) for a in ex.args[2:end] ])
    elseif istensorexpr(ex)
        return reduce(vcat, [ _getallindices(a) for a in ex.args[2:end] if istensorexpr(a) ])
    else
        throw(ArgumentError("not a tensor expression: $ex"))
    end
end
_getallindices(ex::Symbol) = []
_getallindices(ex::Int) = []

# function getopenindices(ex)
#     allidxs = _getallindices(ex)
#     openidxs = filter(i -> count(j -> i == j, allidxs) == 1, allidxs)
#     unique!(openidxs)
#     return openidxs
# end
getindices(ex) = unique(_getindices(ex))
function _getindices(ex::Expr)
    if isscalarexpr(ex)
        return []
    elseif istensor(ex)
        return ex.args[2:end]
    elseif isgeneraltensor(ex)
        allidxs = reduce(vcat, _getindices(a) for a in ex.args[2:end]; init=[])
        idxs = filter(i -> count(j -> j == i, allidxs) == 1, allidxs)
        return idxs
    elseif istensorexpr(ex)
        # display(ex)
        idxs = if ex.args[1] === :*
            # println("*")
            allidxs = reduce(vcat, _getindices(a) for a in ex.args[2:end]; init=[])
            filter(i -> count(j -> j == i, allidxs) == 1, allidxs)
        elseif ex.args[1] in (:+,:-)
            # println("+,-")
            allidxs = [ _getindices(a) for a in ex.args[2:end] ]
            allidxs = map(allidxs) do idxs
                filter(i -> count(j -> j == i, idxs) == 1, idxs)
            end
            allidxs = unique(reduce(vcat, allidxs; init=[]))
            allidxs
        elseif ex.args[1] === :/
            _getindices(ex.args[2])
        else
            throw(ErrorException("this should not have happened!"))
        end
        # display(idxs)
        return idxs
    else
        throw(ArgumentError("not a tensor expression: $ex"))
    end
end
_getindices(ex::Symbol) = []
_getindices(ex::Int) = []

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


# variable =^= any Symbol that is used in an expression with head===:call
function getscalars(ex)
    vars = Symbol[]
    _getscalars!(vars, ex)
    return vars
end


_getscalars!(vars, s::Symbol) = push!(vars, s)
_getscalars!(vars, s::Number) = nothing
function _getscalars!(vars, ex::Expr)
    if ex.head === :call
        foreach(ex.args[2:end]) do a
            _getscalars!(vars, a)
        end
    else
        error("Failed to extract variables from '$ex'")
    end
    return
end
