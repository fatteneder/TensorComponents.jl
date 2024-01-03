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

    # 0. preprocess
    # We support three kinds of functions
    #   - scalar to scalar, e.g. logα[i]   = log(α[i])
    #   - tensor to scalar, e.g. detg      = det(g[a,b])
    #   - tensor to tensor, e.g. invg[a,b] = adjugate(g[a,b]) / detg
    # the supported functions are declared with
    #   - scalar_to_scalar_funcs(),
    #   - tensor_to_scalar__funcs(),
    #   - tensor_to_tensor_funcs().
    precomputes_dummies = Symbol[]
    precomputes = Expr[]
    expr = MacroTools.postwalk(expr) do node
        !isfunctioncall(node) && return node
        fn = node.args[1]
        if fn in scalar_to_scalar_funcs()
            # we apply function elementwise, so nothing to do
            return node
        elseif fn in tensor_to_scalar__funcs()
            # replace the function's argument with a dummy scalar
            # and precompute dummy variables before unrolling
            arg = node.args[2]
            if !istensor(arg)
                throw(ArgumentError("@meinsum: tensor-to-scalar function '$fn' can't be applied to '$arg'"))
            end
            head, idxs = decomposetensor(arg)
            dummy = gensym(Symbol(fn,:_,head))
            push!(precomputes, :($dummy = $fn($head)))
            push!(precomputes_dummies, dummy)
            new_node = dummy
            return new_node
        elseif fn in tensor_to_tensor_funcs()
            # replace the function's argument with a dummy tensor
            # and precompute dummy variables before unrolling
            arg = node.args[2]
            if !istensor(arg)
                throw(ArgumentError("@meinsum: tensor-to-tensor function '$fn' can't be applied to '$arg'"))
            end
            head, idxs = decomposetensor(arg)
            dummy = gensym(Symbol(fn,:_,head))
            push!(precomputes, :($dummy = $fn($head)))
            push!(precomputes_dummies, dummy)
            new_node = :($dummy[$(idxs...)])
            return new_node
        else
            error("@meinsum: found unexpected function call $fn in node $node in expression $expr")
        end
    end

    @assert isassignment(expr)
    if !isassignment(expr)
        throw(ArgumentError("@meinsum: expected assignment like 'LHS = RHS', found $expr"))
    end
    lhs, rhs = getlhs(expr), getrhs(expr)

    # 1. setup
    # - get all open indices LHS
    # - get all open indices RHS
    lhs_idxs = getindices(lhs)
    rhs_idxs = getindices(rhs)

    if sort(lhs_idxs) != sort(rhs_idxs)
        throw(ArgumentError("@meinsum: unbalanced open indices between lhs and rhs: $expr"))
    end
    if rhs isa Expr && rhs.head === :call && rhs.args[1] in (:+,:-)
        rhs_idxs_list = getindices.(rhs.args[2:end])
        sorted_rhs_idxs = sort(rhs_idxs)
        if any(idxs -> sort!(idxs) != sorted_rhs_idxs, rhs_idxs_list)
            throw(ArgumentError("@meinsum: inconsistent open indices on rhs: $rhs"))
        end
    end

    # 2. verify
    # - make sure there are no contracted indices on the LHS
    # - make sure there are the same open indices on both sides
    if !isempty(getcontractedindices(lhs))
        throw(ArgumentError("@meinsum: lhs cannot have contracted indices: $lhs"))
    end
    if !ispermutation(rhs_idxs, permutation_catalog(lhs_idxs))
        throw(ArgumentError("@meinsum: mismatch between open indices between lhs and rhs: $expr"))
    end
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
    append!(let_args, :($h = $h) for h in rhs_heads if !(h in precomputes_dummies) )
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
    parentloop = thebody
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

        if openidxs != lhs_idxs && istraced(tmprhs)
            tmplhs_head, _ = decomposetensor(tmplhs)
            parentloop.args[1].args[2] = Expr(:block, :(fill!($tmplhs_head, 0)), loop, parentloop.args[1].args[2])
            parentloop = loop
        else
            # push!(nestedloop.args, loop)
            pushfirst!(thebody.args, loop)
        end

        # also need to initialize the temporary variables
        if !isempty(openidxs)
            lengths = [ :(length($(allidx_dict[i]))) for i in openidxs ]
            head, _ = decomposetensor(tmplhs)
            push!(init_tmpvars, :($head = zeros(Basic, $(lengths...))))
        else
            push!(init_tmpvars, :($tmplhs = 0))
        end
    end

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

    # TODO Add asserts for tensor sizes
    code = quote
        let $(let_args...)
            $(init_tmpvars...)
            $(precomputes...)
            $theloop
            $lhs_head
        end
    end
    code = MacroTools.postwalk(rmlines, code)

    return code
end


function make_compute_stack(expr)

    stack = Tuple{Any,Expr}[]
    istensor(expr) && iscontraction(expr) && return stack

    # @info expr

    # !iscontraction(expr) && return stack
    new_ex = MacroTools.postwalk(expr) do ex

        # @show ex

        !iscontraction(ex) && return ex

        # extract contraction into separate expression
        contracted, rest = decomposecontraction(ex)
        contraction = if length(contracted) == 1
            contracted[1]
        else
            Expr(:call, :*, contracted...)
            # contraction = Expr(:call, :*)
            # append!(contraction.args, contracted)
            # contraction
        end
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
            new_ex
        end

        # @show contracted, rest, contracted_tensor
        # @show new_ex

        return new_ex
    end

    push!(stack, (nothing, new_ex))
    return stack
end


# TODO Figure out what these should do
# - getallindices ... all indices, but unique
# - getindices ... only open indices
# - getopenindices
# - getcontractedindices

# getopenindices(ex) = getallindices(ex)[1]
# getcontractedindices(ex) = getallindices(ex)[2]


### getters


getallindices(expr) = Any[]
function getallindices(expr::Expr)
    idxs = Any[]
    MacroTools.postwalk(expr) do ex
        istensor(ex) && append!(idxs, ex.args[2:end])
        ex
    end
    unique!(idxs)
    return idxs
end

istraced(expr) = false
function istraced(expr::Expr)
    istensor(expr) || return false
    return !isempty(getcontractedindices(expr))
end


# =^= getopenindices
getindices(ex) = symsort!(unique(_getindices(ex)))
function _getindices(ex::Expr)
    if isfunctioncall(ex)
        return _getindices(ex.args[2])
    elseif istensor(ex)
        idxs = ex.args[2:end]
        uidxs = unique(idxs)
        is_notdup = findall(i -> count(j -> j == i, idxs) == 1, idxs)
        return isempty(is_notdup) ? Any[] : idxs[is_notdup]
    elseif istensorexpr(ex)
        idxs = if ex.args[1] === :*
            allidxs = reduce(vcat, _getindices(a) for a in ex.args[2:end]; init=[])
            filter(i -> count(j -> j == i, allidxs) == 1, allidxs)
        elseif ex.args[1] in (:+,:-)
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
        return idxs
    else # its a scalar expression
        return []
    end
end
_getindices(ex::Symbol) = []
_getindices(ex::Int) = []


function getcontractedindices(ex)
    allidxs = getallindices(ex)
    openidxs = getindices(ex)
    return filter(allidxs) do i
        !(i in openidxs)
    end
end


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


gettensors(ex) = []
function gettensors(expr::Expr)
    tensors = []
    MacroTools.postwalk(expr) do ex
        istensor(ex) && push!(tensors, ex)
        ex
    end
    unique!(tensors)
    return tensors
end


gettensorheads(ex) = []
gettensorheads(expr::Expr) = [ tensor.args[1] for tensor in gettensors(expr) ]


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


_splitprod!(t, s, ex::Number) = push!(s, ex)
_splitprod!(t, s, ex::Symbol) = push!(s, ex)
function _splitprod!(t, s, ex::Expr)
    if istensor(ex)
        push!(t, ex)
        return
    elseif istensorprod(ex)
        @assert ex.head === :call
        op = ex.args[1]
        if op === :/
            push!(s, :(1/$(ex.args[3])))
            _splitprod!(t, s, ex.args[2])
        elseif op === :*
            for a in ex.args[2:end]
                _splitprod!(t, s, a)
            end
        else
            error("Unhandled case $ex")
        end
    elseif isscalarexpr(ex)
        push!(s, ex)
        return
    else
        error("Unhandled case $ex")
    end
end
# split scalars coefficients (but not scalar contractions) from a tensor product
function splitprod(ex)
    t, s = Any[], Any[]
    _splitprod!(t, s, ex)
    symsort!(s)
    return if length(t) == 0
        nothing, s
    elseif length(t) == 1
        t[1], s
    else
        tp = Expr(:call, :*)
        append!(tp.args, t)
        tp, s
    end
end



# variable =^= any Symbol that does not have indices
function getscalars(ex)
    vars = Any[]
    _getscalars!(vars, ex)
    unique!(vars)
    return vars
end
_getscalars!(vars, s::Symbol) = push!(vars, s)
_getscalars!(vars, s::Number) = nothing
function _getscalars!(vars, ex::Expr)
    if istensor(ex)
        return
    elseif ex.head === :call
        if ex.args[1] === :/
            push!(vars, ex)
        else
            foreach(ex.args[2:end]) do a
                _getscalars!(vars, a)
            end
        end
    else
        error("Failed to extract variables from '$ex'")
    end
    return
end







### atomic analyzers


isassignment(ex) = false
isassignment(ex::Expr) = (ex.head == :(=) || ex.head == :(+=) || ex.head == :(-=))


# tensor =^= a single array
istensor(ex::Symbol) = false
istensor(ex::Number) = false
istensor(ex) = false
istensor(ex::Expr) = ex.head === :ref && length(ex.args) >= 2


# for now only restrict to elementary functions with 1 arg
# if we allow functions with more than 1 arg we need to change all probably need to
# update all callsites of isfunctioncall too
function isfunctioncall(ex::Expr)
    ex.head === :call || return false
    length(ex.args) == 2 || return false
    ex.args[1] !== (:ref,:*,:-,:/,:\,:+)
end
isfunctioncall(ex) = false


scalar_to_scalar_funcs()  = (:tan,:sin,:cos,:atan,:asin,:acos,
                             :tanh,:sinh,:cosh,:atanh,:asinh,:acosh,
                             :cot,:sec,:acot,:asec,:coth,:sech,
                             :log,:log10,:log2,:log1p,
                             :exp,:exp10,:exp2,:expm1,
                             :sinpi,:cospi, :abs)
tensor_to_tensor_funcs()  = (:adjugate,)
tensor_to_scalar__funcs() = (:det,)


### non-atomic analyzers

# general tensor =^= a single array with at most scalar coefficients
function isgeneraltensor(ex::Expr)
    istensor(ex) && return true
    ex.head === :call || return false
    length(ex.args) >= 3 && ex.args[1] == :* &&
        count(a -> isgeneraltensor(a), ex.args[2:end]) == 1 &&
        count(a -> isscalarexpr(a), ex.args[2:end]) == length(ex.args)-2 && return true
    length(ex.args) == 3 && ex.args[1] == :/ && isgeneraltensor(ex.args[2]) &&
        !isgeneraltensor(ex.args[3]) && return true
    return false
end
isgeneraltensor(ex) = false


# anything with open indices is not a scalar expression
function isscalarexpr(ex::Expr)
    ex.head === :ref && all(i -> i isa Integer, getindices(ex)) && return true
    ex.head === :call || return false
    isfunctioncall(ex) && return isscalarexpr(ex.args[2])
    ex.args[1] in (:+,:-) && return all(a -> isscalarexpr(a), ex.args[2:end])
    ex.args[1] === :* && return isempty(getindices(ex))
    ex.args[1] === :/ && length(ex.args) == 3 && !istensor(ex.args[end]) &&
        return isempty(getindices(ex))
    ex.args[1] === :^ && length(ex.args) == 3 && isscalarexpr(ex.args[2]) &&
        isscalarexpr(ex.args[3]) && return true
    return false
end
isscalarexpr(ex) = false
isscalarexpr(ex::Symbol) = true
isscalarexpr(ex::Number) = true


function iscontraction(ex::Expr)
    istensorprod(ex) && !isempty(getcontractedindices(ex)) && return true
    isgeneraltensor(ex) && return any(a -> iscontraction(a), ex.args[2:end])
    !istensorexpr(ex) && return false
    all(a -> isscalarexpr(a), ex.args[2:end]) && return true
    !(ex.head === :call && length(ex.args) >= 3 && ex.args[1] === :* &&
      count(a -> istensor(a), ex.args) >= 2) && return false
    cidxs = getcontractedindices(ex)
    isempty(cidxs) && return false
    return true
end
iscontraction(ex) = false


function istensorprod(ex::Expr)
    istensor(ex) && return true
    !istensorexpr(ex) && return false
    ex.head !== :call && return false
    ex.args[1] === :/ && any(a -> istensorprod(a), ex.args[2:end]) && return true
    ex.args[1] === :*
end
istensorprod(ex) = false


# any expression which involves (general)tensors combined with any of the +,-,*,/ operators
function istensorexpr(ex::Expr)
    isgeneraltensor(ex) && return true
    ex.head === :call || return false
    ex.args[1] === :* && length(ex.args) >= 3 &&
        all(a -> istensorexpr(a) || isscalarexpr(a), ex.args[2:end]) &&
        count(a -> istensorexpr(a), ex.args[2:end]) >= 1 && return true
    ex.args[1] === :/ && length(ex.args) == 3 &&
        isscalarexpr(ex.args[3]) && return istensorexpr(ex.args[2])
    ex.args[1] in (:+,:-) && length(ex.args) >= 3 &&
        all(a -> istensorexpr(a), ex.args[2:end]) && return true
    return false
end
istensorexpr(ex) = false


# Any expression brackets ([ ] =^= ex.head === :ref) is considered to have indices, even :(A[])
function hasindices(ex::Expr)
    isfunctioncall(ex) && return hasindices(ex.args[2])
    ex.head === :ref && return true
    ex.head === :call && length(ex.args) >= 3 && any(a -> hasindices(a), ex.args[2:end]) && return true
    return false
end
hasindices(ex) = false
