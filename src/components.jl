macro components(expr)
    return esc(components(expr))
end


function components(expr)
    expr = MacroTools.prewalk(MacroTools.rmlines, expr)

    @assert expr.head === :block "Expected :block, found $(expr.head)"
    exprs = expr.args

    ex_idxs, ex_eqs, ex_sym = decompose_expressions(exprs)
    def_idxs, def_idx_dims  = parse_index_definitions(ex_idxs)
    eq_heads, eq_idxpairs   = parse_heads_idxpairs_equations(ex_eqs)
    sym_heads, sym_idxpairs = parse_heads_idxpairs_symmetries(ex_sym)

    allheads, allidxpairs = vcat(eq_heads, sym_heads), vcat(eq_idxpairs, sym_idxpairs)

    verify_tensors_indices(allheads, allidxpairs, def_idxs)

    # determine independent tensors; relies on tensors having consistent ranks
    idep_heads = determine_independents(ex_eqs)

    # for each tensor determine all indices used in every slot
    uheads, grouped_idxs = group_indices_by_slot(allheads, allidxpairs)

    ### generate code

    # define indices
    code_def_idxs = [ :($name = TensorComponents.Index($dim)) for (dim,name) in zip(def_idx_dims, def_idxs) ]

    # define tensors and scalars
    code_def_tensors = Expr[]
    for (head, gidxs) in zip(uheads, grouped_idxs)
        if length(gidxs) == 0
            push!(code_def_tensors, :($head = SymEngine.symbols($(QuoteNode(head)))))
        else
            slotdims = [ :(TensorComponents.slotdim($(idxs...))) for idxs in gidxs ]
            push!(code_def_tensors, :($head = TensorComponents.SymbolicTensor($(QuoteNode(head)), $(slotdims...))))
        end
    end

    # impose any symmetry relations
    code_resolve_syms = [ generate_code_resolve_symmetries(sym) for sym in ex_sym ]

    # define components:
    # 1. setup views of tensors
    # 2. forward contraction to @meinsum macro
    # 3. gather output
    code_unroll_eqs = [ generate_code_unroll_equations(eq) for eq in ex_eqs ]

    code = quote
        $(code_def_idxs...)
        $(code_def_tensors...)
        $(code_resolve_syms...)
        components = Tuple{Basic,Basic}[]
        $([ quote
               comps = $def
               append!(components, comps)
            end for def in code_unroll_eqs ]...)
        components
    end

    code = MacroTools.prewalk(MacroTools.rmlines, code)
    code = MacroTools.flatten(code)

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
        matched = isassignment(ex)
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
# provided symmetry relation which looks like
# ```julia
#     @symmetry A[i,j] = A[j,i]
# ```
# A symmetry relation can only involve one tensor and must be linear in it.
#
# We verify here that tensors are all used with consistent rank.
function parse_heads_idxpairs_symmetries(exprs)

    heads, idxpairs = Symbol[], Vector{Any}[]
    for ex in exprs

        # TODO Do this capturing only in decompose_expressions and emit a warning of an
        # @index or @symmetry statement is empty
        # assuming all exprs start with @index
        MacroTools.@capture(ex, @symmetry args__)

        if length(args) != 1 || !isassignment(args[1])
            error("@components: @symmtery: expected something like '@symmetry A[i,j] = A[i,j]', found '$ex'")
        end

        lhs, rhs = getlhs(args[1]), getrhs(args[1])

        # TODO Can we support something like T[i,j] = k?
        if !istensorexpr(lhs) || !istensorexpr(rhs)
            error("@components: @symmtery: expected symmetry relation for a tensor's components, e.g. something like '@symmetry A[i,j] = A[i,j]', found '$ex'")
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

        lhs, rhs = getlhs(eq), getrhs(eq)

        if isfunctioncall(lhs)
            error("@components: LHS cannot involve function calls, found '$lhs'!")
        end

        lhs_heads_idxs = if istensorexpr(lhs)
            TO.decomposetensor.(TO.gettensors(lhs))
        elseif isscalarexpr(lhs)
            [ (lhs, [], []) ]
        else
            error("@components: This should not have happened; don't know how to handle: '$lhs'!")
        end

        # unwrap any function calls
        if isfunctioncall(rhs)
            rhs = rhs.args[2]
        end

        rhs_heads_idxs = if istensorexpr(rhs)
            heads_idxs = TO.decomposetensor.(TO.gettensors(rhs))
            scalars = unique(reduce(vcat, getscalars.(getcoeffs(rhs)), init=Symbol[]))
            append!(heads_idxs, (s,[],[]) for s in scalars)
            heads_idxs
        elseif isscalarexpr(rhs)
            coeffs = getcoeffs(rhs)
            scalars = unique(reduce(vcat, getscalars.(coeffs)))
            [ (s, [], []) for s in scalars ]
        else
            error("@components: This should not have happened; don't know how to handle: '$rhs'!")
        end

        for heads_idxs in (lhs_heads_idxs, rhs_heads_idxs), (head, idxs, _) in heads_idxs

            # check if we already encountered this tensor
            # and verify that its rank did not change
            i = findfirst(h -> h===head, tensorheads)
            if !isnothing(i)
                prev_idxs = tensoridxs[i]
                if length(prev_idxs) != length(idxs)
                    prev_eq = eqs[linenrs[i]]
                    if nr == i # inconsistenchy occured on the same line
                        error("@components: found tensor '$head' with inconsistent rank, in '$eq'")
                    else
                        error("""@components: found tensor '$head' with inconsistent rank, compare
                                $prev_eq
                                  vs.
                                $eq
                              """)
                    end
                end
            end

            push!(tensorheads, head)
            push!(tensoridxs, idxs)
            push!(linenrs, nr)
        end
    end

    uheads = unique(tensorheads)
    uidxs = unique(reduce(vcat, tensoridxs, init=Symbol[]))
    if any(i -> i in uheads, uidxs)
        dups = [ idx for idx in uidxs if idx in uheads ]
        error("@components: found duplicated symbols between indices and tensors/scalars '$(join(dups,','))'")
    end

    return tensorheads, tensoridxs
end


function verify_tensors_indices(tensorheads, idxpairs, defined_idxs)
    allidxs = unique(reduce(vcat,idxpairs,init=Symbol[]))
    undefined_idxs = [ idx for idx in allidxs if (!(idx isa Integer) && !(idx in defined_idxs)) ]
    filter!(i -> i isa Symbol, undefined_idxs)
    if !isempty(undefined_idxs)
        error("@components: undefined indices '$(join(undefined_idxs,','))'; you need to declare them with @index first")
    end
    allheads = unique(tensorheads)
    unused_idxs_which_clash = [ idx for idx in defined_idxs if idx in allheads ]
    if !isempty(unused_idxs_which_clash)
        error("@components: found duplicated symbols between unused indices and tensors/scalars '$(join(unused_idxs_which_clash,','))'")
    end
    for head in allheads
        is = findall(h -> h === head, tensorheads)
        ipairs = idxpairs[is]
        !allequal(length.(ipairs)) && error("@components: found tensor '$head' being used with inconsistent rank")
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

    LHSs, RHSs = getlhs.(eqs), getrhs.(eqs)

    # everyting in rhs of first equation is independent
    rhs_ts = TO.gettensorobjects(RHSs[1])
    append!(ideps, rhs_ts)

    N == 1 && return ideps

    lhs_ts = TO.gettensorobjects(LHSs[1])
    if isscalarexpr(LHSs[1])
        push!(defined, LHSs[1])
    elseif length(lhs_ts) == 1
        push!(defined, lhs_ts[1])
    else
        length(lhs_ts) != 1 && error("@components: found invalid LHS in '$(eqs[1])'")
    end

    for i in 2:N
        li, ri = LHSs[i], RHSs[i]
        rhs_ts = TO.gettensorobjects(ri)
        for t in rhs_ts
            if !(t in defined)
                push!(ideps, t)
            end
        end
        lhs_ts = TO.gettensorobjects(li)
        if isscalarexpr(li)
            push!(defined, li)
        elseif length(lhs_ts) == 1
            push!(defined, lhs_ts[1])
        else
            length(lhs_ts) != 1 && error("@components: found invalid LHS in '$(eqs[1])'")
        end
        # push!(defined, lhs_ts[1])
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


# we carry out the contraction using @meinsum
# we do so by interpolating/capturing the views of the rhs tensors into a let block
# for the lhs tensor we capture a deepcopy of its view and then return it
#
# E.g.
#   A[i,j] = B[i,j,k] * C[k]
# translates to
#   v_A = view(...)
#   ...
#   ret_A = let A = deepcopy(v_A), B = v_B, C = v_C
#       A = @meinsum A[i,j] = B[i,j,k] * C[k]
#   end
#
# using a let block here has sevaral advantages:
# - we can forward the equation directly to @meinsum, so we get the full stack trace
# from TensorOperations in case there is a problem with index contractions
# - we don't have to replace tensor heads with their view'd symbols; @meinsum will
# see the equation that was actually entered
# - in fact, one can not even (easily) use MacroTools.pre/postwalk to replace the
# tensor heads alone and ignore indices, which is needed if there are name clashes
# between heads and indices, e.g. A[A,B]; avoiding name clashes between heads and indices
# might be doable, but its not so easy to also avoid if for scalar variables, e.g.
# A[A,B] * B, where B is a scalar and index.
function generate_code_unroll_equations(eq)

    lhs, rhs = getlhs(eq), getrhs(eq)

    # replace all tensor heads with generated symbols, because we use views for each
    # indexed tensor, and some tensors might appear twice (although only on RHSs)
    gend_lhs_heads_idxs = []
    lhs = MacroTools.postwalk(lhs) do node
        # can't capture scalars here (easily), because postwalk will also visit indices (if
        # there are any) and classify those as scalars ...
        !istensor(node) && return node
        heads_idxs = TO.decomposetensor(node)
        head, idxs = heads_idxs[1], heads_idxs[2]
        gend_head = gensym(head)
        push!(gend_lhs_heads_idxs, (head, gend_head, idxs, heads_idxs[2:end]...))
        newnode = MacroTools.postwalk(node) do h
            h === head ? gend_head : h
        end
        filter!(i -> i isa Symbol, newnode.args[2:end])
        if length(newnode.args) == 1
            newnode = newnode.args[1]
        end
        return newnode
    end
    if isempty(gend_lhs_heads_idxs) # lhs must be a scalar
        push!(gend_lhs_heads_idxs, (lhs, lhs, [], [], []))
    end

    gend_rhs_heads_idxs = []
    rhs = MacroTools.postwalk(rhs) do node
        !istensor(node) && return node
        heads_idxs = TO.decomposetensor(node)
        head, idxs = heads_idxs[1], heads_idxs[2]
        gend_head = gensym(head)
        push!(gend_rhs_heads_idxs, (head, gend_head, idxs, heads_idxs[2:end]...))
        newnode = MacroTools.postwalk(node) do h
            h === head ? gend_head : h
        end
        filter!(i -> i isa Symbol, newnode.args)
        if length(newnode.args) == 1
            newnode = newnode.args[1]
        end
        return newnode
    end

    # define views of lhs and rhs tensor, if any
    vlhs_tensors = [ :($ghead = view($head, $(idxs...)))
                    for (head,ghead,idxs,_) in gend_lhs_heads_idxs if length(idxs) > 0 ]
    vrhs_tensors = [ :($ghead = view($head, $(idxs...)))
                    for (head,ghead,idxs,_) in gend_rhs_heads_idxs if length(idxs) > 0 ]

    lhs_head, lhs_ghead = first(gend_lhs_heads_idxs)[1:2]
    ret_lhs = Symbol(:ret_, lhs_ghead)
    let_tensor_expr = quote
        $ret_lhs = let $lhs_ghead = deepcopy($lhs_ghead)
            TensorComponents.@meinsum $lhs = $rhs
        end
    end

    # extract independent tensor components, if any
    extract_indeps = if !isscalarexpr(lhs)
        ulhs_ghead = Symbol(:u_,lhs_ghead)
        quote
            $ulhs_ghead = unique($lhs_ghead)
            idx_indeps = [ findfirst(h -> h == u, $lhs_ghead) for u in $ulhs_ghead ]
            zip(view($lhs_ghead, idx_indeps), view($ret_lhs, idx_indeps))
        end
    else
        :(zip($lhs_ghead, $ret_lhs))
    end

    code = quote
        let
            $(vlhs_tensors...)
            $(vrhs_tensors...)
            $let_tensor_expr
            $extract_indeps
        end
    end

    return code
end


# we carry out the contraction similar to generate_code_unroll_equations above
# the difference here is that the expression transforming looks like
#   @symmetry A[i,j] = A[j,i]
# becomes
#   var"#A##0123" = view(A,i,j)
#   var"#A##0567" = view(A,j,i)
#   eqs = let
#       @meinsum eqs[i,j] = var"#A##0123"[i,j] - var"#A##0123"
#   end
#   A = TensorComponents.resolve_dependents(A, eqs)
function generate_code_resolve_symmetries(ex_sym)

    MacroTools.@capture(ex_sym, @symmetry sym_)

    lhs, rhs = getlhs(sym), getrhs(sym)

    # extract the tensor for which we resolve symmetries
    # we already checked that there is exactly one
    ts = TO.gettensorobjects(lhs)
    head = !isempty(ts) ? ts[1] : TO.gettensorobjects(rhs)[1]

    # replace all tensor heads with generated symbols, because we use views for each
    # indexed tensor, and some tensors might appear twice (although on ly on RHSs)
    gend_lhs_heads_idxs = []
    gend_lhs = MacroTools.postwalk(lhs) do node
        !istensor(node) && return node
        heads_idxs = TO.decomposetensor(node)
        head = heads_idxs[1]
        gend_head = gensym(head)
        push!(gend_lhs_heads_idxs, (head, gend_head, heads_idxs[2:end]...))
        newnode = MacroTools.postwalk(node) do h
            h === head ? gend_head : h
        end
        return newnode
    end
    gend_rhs_heads_idxs = []
    gend_rhs = MacroTools.postwalk(rhs) do node
        !istensor(node) && return node
        heads_idxs = TO.decomposetensor(node)
        head = heads_idxs[1]
        gend_head = gensym(head)
        push!(gend_rhs_heads_idxs, (head, gend_head, heads_idxs[2:end]...))
        newnode = MacroTools.postwalk(node) do h
            h === head ? gend_head : h
        end
        return newnode
    end

    # define views of lhs and rhs tensors, if any
    vlhs_tensors = [ :($ghead = view($head, $(idxs...)))
                    for (head,ghead,idxs,_) in gend_lhs_heads_idxs if length(idxs) > 0 ]
    vrhs_tensors = [ :($ghead = view($head, $(idxs...)))
                    for (head,ghead,idxs,_) in gend_rhs_heads_idxs if length(idxs) > 0 ]

    eqvar = gensym(:eq)
    eqs_idxs = first(gend_lhs_heads_idxs)[3]
    eqs_init = if isempty(eqs_idxs)
        0
    else
        slotdims = [ :(TensorComponents.slotdim($idxs)) for idxs in eqs_idxs ]
        :(TensorComponents.SymbolicTensor(:eqs, $(slotdims...)))
    end
    # let_tensor_expr = :(let; TensorComponents.@meinsum eqs[$(eqs_idxs...)] = $gend_lhs - $gend_rhs; end)
    let_tensor_expr = quote
        let
            eqs = $eqs_init
            TensorComponents.@meinsum eqs[$(eqs_idxs...)] = $gend_lhs - $gend_rhs
        end
    end

    # # TODO Add symmetry relation as comment with a linenode, because we had
    # # to obfuscate it; wait, obfuscation already occurs because of the @meinsum macro,
    # # so its needed anyways
    # # comment = Expr(:line, string(ex_sym))
    # comment1 = LineNumberNode(0, Symbol(repeat('=',100)*"\n sers oida"))
    # comment2 = LineNumberNode(0, Symbol(ex_sym))
    # comment3 = LineNumberNode(0, Symbol(repeat('=',100)))
    code = quote
        $head = let
            $(vlhs_tensors...)
            $(vrhs_tensors...)
            $eqvar = $let_tensor_expr
            TensorComponents.resolve_dependents($head, $eqvar)
        end
    end

    return code
end
