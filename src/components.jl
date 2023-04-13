macro components(expr)
    return esc(components(expr))
end


function components(expr)
    expr = MacroTools.prewalk(MacroTools.rmlines, expr)

    @assert expr.head === :block "Expected :block, found $(expr.head)"
    exprs = expr.args

    ex_idxs, ex_eqs, ex_sym = decompose_expressions(exprs)
    def_idxs, def_idx_dims = parse_index_definitions(ex_idxs)
    eq_heads, eq_idxpairs = parse_heads_idxpairs_equations(ex_eqs)
    sym_heads, sym_idxpairs = parse_heads_idxpairs_symmetries(ex_sym)

    verify_tensors_indices(eq_heads, eq_idxpairs, sym_heads, sym_idxpairs, def_idxs)

    # determine independent tensors; relies on tensors having consistent ranks
    idep_heads = determine_independents(ex_eqs)

    # for each tensor determine all indices used in every slot
    uheads, grouped_idxs = group_indices_by_slot(eq_heads, eq_idxpairs)

    ### generate code

    # define indices
    code_idxs = [ :($name = TensorComponents.Index($dim)) for (dim,name) in zip(def_idx_dims, def_idxs) ]

    # define tensors and scalars
    code_tensors = Expr[]
    for (head, gidxs) in zip(uheads, grouped_idxs)
        if length(gidxs) == 0
            push!(code_tensors, :($head = SymEngine.symbols($(QuoteNode(head)))))
        else
            slotdims = [ :(TensorComponents.slotdim($(idxs...))) for idxs in gidxs ]
            push!(code_tensors, :($head = TensorComponents.SymbolicTensor($(QuoteNode(head)), $(slotdims...))))
        end
    end

    # define components:
    # 1. setup views of tensors
    # 2. forward contraction to TensorOperations.@tensor macro
    # 3. gather output
    code_components = [ generate_components_code(eq) for eq in ex_eqs ]

    code = quote
        $(code_idxs...)
        $(code_tensors...)

        components = Tuple{Basic,Basic}[]
        $([ :(comps = $def; append!(components, comps)) for def in code_components ]...)

        return components
    end

    return code
end


slotdim(i::Int) = 1
slotdim(i::Index) = length(i)
slotdim(is...) = maximum(slotdim(i) for i in is)


# We recognize the following lines
# - lines starting with @index
# - lines starting with @symmetry
# - lines with an assignment, e.g. A[i,j] = B[i,j] or a = b + 1
function decompose_expressions(exprs)

    idxs, eqs, syms = Expr[], Expr[], Expr[]

    for ex in exprs
        matched = MacroTools.@capture(ex, @index __)
        matched && (push!(idxs, ex); true) && continue
        matched = TO.isassignment(ex)
        matched && (push!(eqs, ex); true) && continue
        matched = MacroTools.@capture(ex, @symmetry sym__)
        matched && (push!(syms, ex); true) && continue

        @warn "@components: found unknown expression '$ex', skipping ..."
    end

    return idxs, eqs, syms
end


# Gather index names and their dimensions from expressions like
# ```julia
#     @index a, b, c, d = 4
# ```
# Index names must be valid Julia `Symbols`, e.g. `a, b, i123`.
# An index can only be declared once.
function parse_index_definitions(exprs)

    dims, indices = Int[], Symbol[]
    for ex in exprs

        # assuming all exprs start with @index
        MacroTools.@capture(ex, @index args__)

        if length(args) == 0
            error("@components: @index: expected something like '@index a b c = 1:4', found '$ex'")
        end

        matched_tpl = MacroTools.@capture(args[1], (idxs__,) = range_)
        matched_sngl = MacroTools.@capture(args[1], idx_ = range_)
        if !matched_tpl && !matched_sngl
            error("@components: @index: expected something like '@index a b c = 1:4', found '$ex'")
        end

        matched_range = MacroTools.@capture(range, start_:stop_)
        dim = matched_range ? (start:stop) : range

        if !((dim isa Int && dim > 0) || (dim isa UnitRange && dim.start > 0 && length(dim) > 0))
            error("@components: @index: index range must be positive integer or a positive unit range (e.g. 1:4, 2:4), found '$dim'")
        end

        idxlist = matched_tpl ? idxs : [idx]
        for idx in idxlist
            !(idx isa Symbol) && error("@components: @index: names must be symbols like 'a, b, i123', found '$idx' in '$ex'")
            push!(dims, dim isa Int ? dim : length(dim))
            push!(indices, idx)
        end
    end

    if !allunique(indices)
        dups = [ u for u in unique(indices) if count(i -> i === u, indices) > 1 ]
        error("@components: @index: an index can only be defined once, found multiple definitions for '$(join(dups,' '))'")
    end

    return indices, dims
end


# Determine the tensor and all occuring index permutations used in the
# provided symmetry relation which like look
# ```julia
#     @symmetry A[i,j] = A[j,i]
# ```
# A symmetry relation can only involve one tensor and must be linear in it.
#
# We verify here that tensors are all used with consistent rank.
function parse_heads_idxpairs_symmetries(exprs)

    heads, idxpairs = Symbol[], Vector{Any}[]
    for ex in exprs

        # assuming all exprs start with @index
        MacroTools.@capture(ex, @symmetry args__)

        if length(args) != 1 || !TO.isassignment(args[1])
            error("@components: @symmtery: expected something like '@symmetry A[i,j] = A[i,j]', found '$ex'")
        end

        lhs, rhs = TO.getlhs(args[1]), TO.getrhs(args[1])

        # TODO Can we support something like T[i,j] = k?
        if !TO.istensorexpr(lhs) || !TO.istensorexpr(rhs)
            error("@components: @symmtery: expected symmetry relation for a tensor's components, e.g. omething like '@symmetry A[i,j] = A[i,j]', found '$ex'")
        end

        ts_lhs, ts_rhs = TO.gettensorobjects(lhs), TO.gettensorobjects(rhs)
        ts = unique!(reduce(vcat, (ts_lhs, ts_rhs)))
        if length(ts) != 1
            error("@components: @symmetry: a relation can only involve one tensor, found multiple ones '$(join(ts,','))' in '$ex'")
        end
        head = ts[1]

        allidxs_lhs, allidxs_rhs = TO.getallindices(lhs), TO.getallindices(rhs)
        idxs_lhs, idxs_rhs       = TO.getindices(lhs), TO.getindices(rhs)

        if length(allidxs_lhs) != length(allidxs_rhs) != length(idxs_lhs) != length(idxs_rhs)
            error("@components: @symmetry: inconsistent index pattern found in '$ex'")
        end

        heads_idxs_lhs = TO.decomposetensor.(TO.gettensors(lhs))
        heads_idxs_rhs = TO.decomposetensor.(TO.gettensors(rhs))

        start = length(length(idxpairs))
        for heads_idxs in (heads_idxs_lhs, heads_idxs_rhs)
            for (_,idxs,_) in heads_idxs
                push!(heads, head)
                push!(idxpairs, idxs)
            end
        end
        stop = length(length(idxpairs))

        if !allequal(length.(view(idxpairs, start:stop)))
            error("@components: @symmetry: inconsistent index pattern found in '$ex'")
        end

    end

    return heads, idxpairs
end


# Return list of all tensor heads and a list of the indices they are used with.
# We verify here that tensors are all used with consistent rank.
function parse_heads_idxpairs_equations(eqs)

    # record linenrs for debugging
    tensorheads, tensoridxs, linenrs = Symbol[], Vector{Any}[], Int[]
    for (nr, eq) in enumerate(eqs)

        lhs, rhs = TO.getlhs(eq), TO.getrhs(eq)

        lhs_heads_idxs = if TO.istensorexpr(lhs)
            TO.decomposetensor.(TO.gettensors(lhs))
        elseif TO.isscalarexpr(lhs)
            [ (lhs, [], []) ]
        else
            error("@components: This should not have happened!")
        end
        rhs_heads_idxs = if TO.istensorexpr(rhs)
            heads_idxs = TO.decomposetensor.(TO.gettensors(rhs))
            scalars = unique(reduce(vcat, getscalars.(getcoeffs(rhs)), init=Symbol[]))
            append!(heads_idxs, (s,[],[]) for s in scalars)
            heads_idxs
        elseif TO.isscalarexpr(rhs)
            coeffs = getcoeffs(rhs)
            scalars = unique(reduce(vcat, getscalars.(coeffs)))
            [ (s, [], []) for s in scalars ]
        else
            error("@components: This should not have happened!")
        end

        for heads_idxs in (lhs_heads_idxs, rhs_heads_idxs)
            for (head, idxs, _) in heads_idxs

                # check if we already encountered this tensor
                # and verify that its rank did not change
                i = findfirst(h -> h===head, tensorheads)
                if !isnothing(i)
                    prev_idxs = tensoridxs[i]
                    if length(prev_idxs) != length(idxs)
                        prev_eq = eqs[linenrs[i]]
                        # TODO What if inconsistency appears on the same line?
                        error("""@components: found tensor '$head' with inconsistent rank, compare
                                $prev_eq
                                  vs.
                                $eq
                              """)
                    end
                end

                push!(tensorheads, head)
                push!(tensoridxs, idxs)
                push!(linenrs, nr)
            end
        end
    end

    uheads = unique!(tensorheads)
    uidxs = unique(reduce(vcat, tensoridxs, init=Symbol[]))
    if any(i -> i in uheads, uidxs)
        dups = [ idx for idx in uidxs if idx in uheads ]
        error(rstrip(msg[1:end-length(", ")]))
        error("@components: found duplicated symbols between indices and tensors '$(join(dups,','))'")
    end

    return tensorheads, tensoridxs
end


function verify_tensors_indices(eq_heads, eq_idxpairs, sym_heads, sym_idxpairs, defined_idxs)

    for idxpairs in (eq_idxpairs, sym_idxpairs), idxs in idxpairs, i in idxs
        if !(i in defined_idxs)
            error("@components: undefined indices '$undefined_idxs'; you need to declare them with @index first")
        end
    end

end



# Walk through equations and determine tensors which we have not been computed before
# they are used. E.g. returns all tensors that did not appear in a lhs before they first
# appeared in a rhs.
function determine_independents(eqs)

    ideps = Symbol[]
    defined = Symbol[]

    N = length(eqs)

    N == 0 && return ideps

    LHSs, RHSs = TO.getlhs.(eqs), TO.getrhs.(eqs)

    # everyting in rhs of first equation is independent
    rhs_ts = TO.gettensorobjects(RHSs[1])
    append!(ideps, rhs_ts)

    N == 1 && return ideps

    lhs_ts = TO.gettensorobjects(LHSs[1])
    length(lhs_ts) != 1 && error("@components: found invalid LHS in '$(eqs[1])'")
    push!(defined, lhs_ts[1])

    for i in 2:N
        li, ri = LHSs[i], RHSs[i]
        rhs_ts = TO.gettensorobjects(ri)
        for t in rhs_ts
            if !(t in defined)
                push!(ideps, t)
            end
        end
        lhs_ts = TO.gettensorobjects(li)
        length(lhs_ts) != 1 && error("@components: found invalid LHS in '$(eqs[i])'")
        push!(defined, lhs_ts[1])
    end

    unique!(ideps)

    return ideps
end


# Given some `heads` and a list of indices `idxpairs` each head is used with,
# find for each head the unique indices used for each of its slot.
#
# Assumes that all `idxpairs` have consistent rank wrt their heads.
#
# # Example
#
# ```julia
# heads = [ :A, :B, :A ]
# idxpairs = [ [:i,:j], [:k], [:i,:a] ]
# group_indices_by_slot(heads, idxpairs)
# [ :A, :B ], [ [[:i],[:j,:a]], [:k] ]
# ```
function group_indices_by_slot(heads, idxpairs)

    uheads = unique(heads)
    idxcombos = Vector{Vector{Any}}[]
    for head in uheads
        is = findall(h -> h === head, heads)
        idxs = view(idxpairs, is)
        combo = [ unique(idx[i] for idx in idxs) for i = 1:length(first(idxs)) ]
        push!(idxcombos, combo)
    end

    return uheads, idxcombos
end


function generate_components_code(eq)

    lhs, rhs = TO.getlhs(eq), TO.getrhs(eq)

    # only extract tensors, because we don't need take views of scalars
    lhs_head, lhs_idxs = if TO.istensorexpr(lhs)
        TO.decomposetensor(TO.gettensors(lhs)[1])
    else
        lhs, []
    end
    rhs_heads_idxs = if TO.istensorexpr(rhs)
        TO.decomposetensor.(TO.gettensors(rhs))
    elseif TO.isscalarexpr(rhs)
        coeffs = getcoeffs(rhs)
        scalars = unique(reduce(vcat, getscalars.(coeffs)))
        [ (s, [], []) for s in scalars ]
    else
        error("@components: This should not have happened!")
    end

    # define views of lhs and rhs tensor, if any
    vlhs_tensor = if TO.istensorexpr(lhs)
        :($(Symbol(:v_,lhs_head)) = view($lhs_head, $(lhs_idxs...)))
    else
        nothing
    end
    vrhs_tensors = [ :($(Symbol(:v_,head)) = view($head, $(idxs...)))
                    for (head,idxs,_) in rhs_heads_idxs if length(idxs) > 0 ]

    # we carry out the contraction using TO.@tensor
    # we do so by interpolating/capturing the views of the rhs tensors into a let block
    # for the lhs tensor we capture a deepcopy of its view and then return it
    #
    # E.g.
    #   A[i,j] = B[i,j,k] * C[k]
    # translates to
    #   v_A = view(...)
    #   ...
    #   ret_A = let A = deepcopy(v_A), B = v_B, C = v_C
    #       @tensor A[i,j] = B[i,j,k] * C[k]
    #       A
    #   end
    #
    # using a let block here has sevaral advantages:
    # - we can forward the equation directly to @tensor, so we get the full stack trace
    # from TensorOperations in case there is a problem with index contractions
    # - we don't have to replace tensor heads with their view'd symbols; @tensor will
    # see the equation that was actually entered
    # - in fact, one can not even (easily) use MacroTools.pre/postwalk to replace the
    # tensor heads alone and ignore indices, which is needed if there are name clashes
    # between heads and indices, e.g. A[A,B]; avoiding name clashes between heads and indices
    # might be doable, but its not so easy to also avoid if for scalar variables, e.g.
    # A[A,B] * B, where B is a scalar and index.
    #
    # TODO Generated code is inefficient in terms of number of operations,
    # because atm any scalar coefficients are always immediately multiplied
    # with all elements.
    # Idea: Can we add a field 'coeffs' to SymbolicTensor and then dispatch
    # Base.:*(coeff, symtensor) to update only the coeff but not immediately
    # multiply everything? Or is something like a * A done using broadcasting?
    letargs = [ :($head = $(Symbol(:v_,head))) for (head,idxs,_) in rhs_heads_idxs
                if length(idxs) > 0 ] # only views
    if !isnothing(vlhs_tensor)
        insert!(letargs, 1, :($lhs_head = deepcopy($(Symbol(:v_,lhs_head)))))
    end
    ret_lhs = Symbol(:ret_, lhs_head)
    let_tensor_expr = :($ret_lhs = let $(letargs...); @tensor $eq; $lhs_head end)

    # unpack the results
    # TODO only unpack the independent components of the lhs_tensor
    components_expr = isempty(lhs_idxs) ? :(zip($lhs_head, $ret_lhs)) : :(zip($lhs_head[:], $ret_lhs[:]))

    code = quote
        $vlhs_tensor
        $(vrhs_tensors...)
        $let_tensor_expr
        $components_expr
    end

    return code
end
