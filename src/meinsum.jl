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

    expr = MacroTools.postwalk(rmlines, expr)
    expr = MacroTools.flatten(expr)

    @assert isassignment(expr)
    lhs, rhs = getlhs(expr), getrhs(expr)

    # 1. setup
    # - get all open indices LHS
    # - get all open indices RHS
    # getopenindices should already check for consistency in a tensor expression
    lhs_idxs = getindices(lhs)
    rhs_idxs = getindices(rhs)

    # 2. verify
    # - make sure there are no contracted indices on the LHS
    # - make sure there are the same open indices on both sides
    @assert isempty(getcontractedindices(lhs))
    @assert ispermutation(rhs_idxs, permutation_catalog(lhs_idxs))
    rhs_conidxs = getcontractedindices(rhs)

    # 3. unroll and contract
    # loop_idxs      = [ :($i = TensorComponents.start($i):TensorComponents.stop($i)) for i in rhs_idxs ]
    # innerloop_idxs = [ :($i = TensorComponents.start($i):TensorComponents.stop($i)) for i in rhs_cidxs ]

    # we want to loop over the expr as it is (don't replace any symbols or indices) to keep
    # debugging simple
    # but since we sum over all indices (duplicated ones for contraction, open ones to
    # fill the resulting array), we need to locally rename each index's range, e.g. consider
    # ```
    # i, j = Index(4), Index(4)
    # A, B = TC.SymbolicTensor(:A,4,4), TC.SymbolicTensor(:B,4,4)
    # @meinsum A[i,j] = B[i,j]
    # ```
    # the macro should then expand to
    # ```
    # let A = zeros(eltype(A),A), B = B, var"#i#123" = i, var"#j#124" = j, i, j
    #   for i = var"#i#123", j = var"#j#124"
    #       A[i,j] += B[i,j]
    #   end
    #   A
    # end
    # ```

    # gensym all indices
    idx_dict    = Dict(idx => gensym(idx) for idx in rhs_idxs)
    conidx_dict = Dict(idx => gensym(idx) for idx in rhs_conidxs)
    allidx_dict = merge(idx_dict, conidx_dict)

    # setup let args
    let_args = Any[]
    lhs_head = if istensor(lhs)
        lhs_head = first(decomposetensor.(gettensors(lhs)))[1]
        push!(let_args, :($lhs_head = zeros(eltype($lhs_head),size($lhs_head))) )
        lhs_head
    else
        lhs_head = lhs
        push!(let_args, :($lhs_head = 0)) # don't need a copy
        lhs_head
    end
    rhs_tensors = gettensors(rhs)
    unique!(rhs_tensors)
    rhs_heads = [ decomposetensor(t)[1] for t in rhs_tensors ]
    append!(let_args, :($h = $h) for h in rhs_heads )
    append!(let_args, :($gi = $i) for (i,gi) in pairs(idx_dict) )
    append!(let_args, :($gi = $i) for (i,gi) in pairs(conidx_dict) )
    append!(let_args, i for i in rhs_idxs )
    append!(let_args, i for i in rhs_conidxs )
    unique!(let_args)

    # construct the let body by decomposing the expression into a contraction stack
    computestack = make_compute_stack(expr)
    @assert length(computestack) >= 1
    loops = Expr[]
    # last stack element always computes the initial lhs
    _, tmpexpr = pop!(computestack)
    tmpexpr.head = :(+=)
    conidxs = getallindices(getrhs(tmpexpr))
    init_tmpvars = Any[]
    thebody = Expr(:block, tmpexpr)
    theloop = Expr(:for, Expr(:block, [ :($i = $(allidx_dict[i])) for i in conidxs ]...), thebody)
    for (tmplhs, tmprhs) in reverse(computestack)

        openidxs = getindices(tmplhs)
        conidxs = getcontractedindices(tmprhs)
        # loop = Expr(:block)
        # if !isempty(openidxs)
        #     lengths = [ :(length(i)) for i in openidxs ]
        #     push!(loop.args, :($tmplhs = zeros($(lengths...))))
        # end
        # push!(loop.args, Expr(:for,
        #                       Expr(:block, [ :($i = $(allidx_dict[i])) for i in conidxs ]...),
        #                       :($tmplhs += $tmprhs)) )
        loop = Expr(:for, Expr(:block, [ :($i = $(allidx_dict[i])) for i in conidxs ]...),
                          :($tmplhs += $tmprhs) )
        pushfirst!(thebody.args, loop)

        # also need to initialize the temporary variables
        if !isempty(openidxs)
            lengths = [ :(length($(allidx_dict[i]))) for i in openidxs ]
            head, _ = decomposetensor(tmplhs)
            push!(init_tmpvars, :($head = zeros(Basic, $(lengths...))))
        else
            push!(init_tmpvars, :($tmplhs = 0))
        end
    end
    # display(theloop)

    # # setup loop arguments
    # loop_idxs      = [ :($i = $gi) for (i,gi) in zip(rhs_idxs,gen_idxs) ]
    # innerloop_idxs = [ :($i = $gi) for (i,gi) in zip(rhs_cidxs,gen_cidxs) ]

    # # setup for loops
    # # only here we manipulate expr by converting it from = to + to accumulate result in LHS
    # addup_expr = Expr(:(+=), expr.args...)
    # innerfor_loop = Expr(:for, Expr(:block, innerloop_idxs...), addup_expr)
    # for_loop = if istensor(lhs)
    #     Expr(:for, Expr(:block, loop_idxs...), innerfor_loop)
    # else
    #     innerfor_loop
    # end
    #
    # code = Expr(:let, Expr(:block, let_args...), Expr(:block, for_loop, lhs_head))

    code = Expr(:let, Expr(:block, let_args...), Expr(:block, init_tmpvars..., theloop, lhs_head))

    return code

end


function make_compute_stack(expr)

    stack = Tuple{Any,Expr}[]
    istensor(expr) && iscontraction(expr) && return stack

    # !iscontraction(expr) && return stack
    new_ex = MacroTools.postwalk(expr) do ex

        !iscontraction(ex) && return ex

        # extract contraction into separate expression
        contracted, rest = decomposecontraction(ex)
        contraction = Expr(:call, :*)
        append!(contraction.args, contracted)
        # generate a tensor head for the contraction so we can refer to it later
        openidxs = getindices(contraction)
        gend_head = gensym()
        contracted_tensor = if isempty(openidxs)
            gend_head
        else
            Expr(:ref, gend_head, openidxs...)
        end
        # and push it to the compute stack
        push!(stack, (contracted_tensor, contraction))

        # rewrite the current expression by replacing the contraction with the generated tensor
        new_ex = if isempty(rest)
            contracted_tensor
        else
            new_ex = Expr(:call, :*)
            append!(new_ex.args, rest)
            push!(new_ex.args, contracted_tensor)
        end

        return new_ex
    end

    push!(stack, (nothing, new_ex))
    return stack
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
        hasindices(ex) && return reduce(vcat, [ _getallindices(a) for a in ex.args[2:end] ])
        return Any[]
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

# =^= getopenindices
getindices(ex) = unique(_getindices(ex))
function _getindices(ex::Expr)
    if isscalarexpr(ex)
        return []
    elseif istensor(ex)
        idxs = ex.args[2:end]
        uidxs = unique(ex.args[2:end])
        is_notdup = findall(i -> count(j -> j == i, idxs) == 1, idxs)
        return isempty(is_notdup) ? Any[] : idxs[is_notdup]
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


function getcontractedindices(ex)
    allidxs = getallindices(ex)
    openidxs = getindices(ex)
    return [ i for i in allidxs if !(i in openidxs) ]
end


# Taken from TensorOperations.jl
isassignment(ex) = false
isassignment(ex::Expr) = (ex.head == :(=) || ex.head == :(+=) || ex.head == :(-=))


# Taken from TensorOperations.jl
function getlhs(ex)
    if isassignment(ex) && length(ex.args) == 2
        return ex.args[1]
    else
        throw(ArgumentError("invalid assignment or definition $ex"))
    end
end
function getrhs(ex)
    if isassignment(ex) && length(ex.args) == 2
        return ex.args[2]
    else
        throw(ArgumentError("invalid assignment or definition $ex"))
    end
end


function gettensors(ex)
    ts = []
    _gettensors!(ts, ex)
    return ts
end


function _gettensors!(vars, ex)
    if isscalarexpr(ex)
        nothing
    elseif istensor(ex)
        push!(vars, ex)
    elseif isgeneraltensor(ex)
        foreach(ex.args) do a
            istensor(a) && push!(vars, a)
        end
    elseif ex.head === :call && length(ex.args) >= 3 && ex.args[1] in (:*,:+,:-)
        foreach(ex.args[2:end]) do a
            _gettensors!(vars, a)
        end
    elseif ex.head === :call && length(ex.args) == 3 && ex.args[1] === :/
        _gettensors!(vars, ex.args[2])
    else
        error("Failed to extract variables from '$ex'")
    end
    return
end
_gettensors!(vars, ex::Symbol) = nothing
_gettensors!(vars, ex::Number) = nothing


function decomposetensor(ex)
    if istensor(ex)
        return (ex.args[1], ex.args[2:end])
    elseif isgeneraltensor(ex)
        for a in ex.args[2:end]
            istensor(a) && return (a.args[1], args[2:end])
        end
    else
        throw(ArgumentError("not a vlid tensor: $ex"))
    end
end


function decomposecontraction(ex)
    !iscontraction(ex) && return [], []
    istensor(ex) && return [ex], []
    cidxs = getcontractedindices(ex)
    is = findall(ex.args) do a
        !istensor(a) && return false
        idxs = getallindices(a)
        return any(ci -> ci in cidxs, idxs)
    end
    contracted = ex.args[is]
    rest = [ ex.args[i] for i = 1:length(ex.args)
            if i != 1 #= ex.args[1] === :* =# && !(i in is) ]
    return contracted, rest
end


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


# anything with open indices is not a scalar expression
# TODO Fix isscalarexpr(:(D[k] * D[k] + b)) == true
# isscalarexpr(ex) = !hasindices(ex)
isscalarexpr(ex::Symbol) = true
isscalarexpr(ex::Number) = true
function isscalarexpr(ex)
    !hasindices(ex) && return true
    ex.head === :call && length(ex.args) >= 3 && ex.args[1] in (:+,:-) &&
        return all(a -> isscalarexpr(a) || isempty(getindices(a)), ex.args[2:end])
    return false
end


iscontraction(ex::Symbol) = false
iscontraction(ex::Number) = false
function iscontraction(ex)
    istensor(ex) && !isempty(getcontractedindices(ex)) && return true
    isgeneraltensor(ex) && return any(a -> iscontraction(a), ex.args[2:end])
    !istensorexpr(ex) && return false
    !(ex.head === :call && length(ex.args) >= 3 && ex.args[1] === :* &&
      count(a -> istensor(a), ex.args) >= 2) && return false
    cidxs = getcontractedindices(ex)
    isempty(cidxs) && return false
    return true
end


# Any expression brackets ([ ] =^= ex.head === :ref) is considered to have indices, even :(A[])
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
