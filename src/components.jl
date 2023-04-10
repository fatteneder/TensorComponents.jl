macro components(expr)
    return esc(components(expr))
end


function components(expr)
    expr = MacroTools.prewalk(MacroTools.rmlines, expr)

    @assert expr.head === :block "Expected :block, found $(expr.head)"
    exprs = expr.args

    # checks that indices are *Symbols* and their index range is valid
    uidxs, idx_dims = gather_index_definitions(exprs)

    # TODO Implement
    # sym_tensor_heads, sym_lhs, sym_rhs = gather_symmetry_definitions(exprs)

    # gather tensor statements =^= every ex for which TO.isassignment(ex) = true
    eqs = gather_tensor_equations_definitions(exprs)

    # gather tensor heads and all index pairs that appear with them (for each equation)
    # In this step we also
    # - verify that tensors have consistent rank in all eqs,
    # - verify that all indices appearing in tensor statements have been defined with @index.
    heads, idxpairs = gather_tensor_heads_idxpairs(eqs, uidxs)

    # gather all variables that appear in scalar factors (for each equation)
    scalarvars = gather_scalar_variables(eqs)
    uscalarvars = unique!(reduce(vcat, scalarvars, init=Symbol[]))

    # determine independent tensors; relies on tensors having consistent ranks
    idep_heads = determine_independents(eqs)

    # for each tensor determine all indices used in every slot
    uheads, grouped_idxs = group_indices_by_slot(heads, idxpairs)

    resolve_name_clashes!(uidxs, uheads, uscalarvars)

    ### generate code

    # define indices
    def_idxs = [ :($name = TensorComponents.Index($dim)) for (dim,name) in zip(idx_dims, uidxs) ]

    # define symbolic tensors
    aux_slotdims = [ [ :(TensorComponents.slotdim($(idxs...))) for idxs in gidxs ] for gidxs in grouped_idxs ]
    def_tensors  = [ :($head = TensorComponents.SymbolicTensor($(QuoteNode(head)), $(slotdims...)))
                     for (head,slotdims) in zip(uheads, aux_slotdims) ]

    # define variables
    def_vars = [ :($var = SymEngine.symbols($(QuoteNode(var)))) for var in uscalarvars ]

    # define components:
    # 1. setup views of tensors
    # 2. forward contraction to TensorOperations.@tensor macro
    # 3. gather output
    def_components = [ generate_components_code(eq) for eq in eqs ]

    code = quote
        $(def_idxs...)
        $(def_tensors...)
        $(def_vars...)

        components = Tuple{Basic,Basic}[]
        $([ :(comps = $def; append!(components, comps)) for def in def_components ]...)

        return components
    end

    return code
end


slotdim(i::Int) = 1
slotdim(i::Index) = length(i)
slotdim(is...) = maximum(slotdim(i) for i in is)


"""
    function gather_index_definitions(exprs)

Gather lines from `exprs` which look like
```julia
    @index a, b, c, d = 4
```

Index names must be valid Julia `Symbols`, e.g. `a, b, i123`.

An index can only be declared once in an `exprs` block.
"""
function gather_index_definitions(exprs)

    dims, indices = Int[], Symbol[]
    for (nr, ex) in enumerate(exprs)
        # matched = MacroTools.@capture(ex, @index names__ = range__)
        matched = MacroTools.@capture(ex, @index args__)
        !matched && continue

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

    uindices = unique(indices)
    if length(indices) != length(uindices)
        dups = [ u for u in uindices if count(i -> i === u, indices) > 1 ]
        error("@components: @index: an index can only be defined once, found multiple definitions for '$(join(dups,' '))'")
    end

    return indices, dims
end


"""
    gather_tensor_equations_definitions(exprs)

Gather lines from `exprs` which look like
```julia
    A[i,j] = B[i,j,k] * C[j] * D[k]
```

In particular, returns all lines for which `TensorOperations.isassignment(ex) = true`.

No checks on index contraction etc are done here.
"""
function gather_tensor_equations_definitions(exprs)
    return [ ex for ex in exprs if TO.isassignment(ex) ]
end


"""
    gather_tensor_heads_idxpairs(eqs, idx_names)

Return list of all tensor heads and a list of the indices they are used with.
"""
function gather_tensor_heads_idxpairs(eqs, idx_names)

    # record linenrs for debugging
    tensorheads, tensoridxs, linenrs = Symbol[], Vector{Any}[], Int[]
    for (nr, eq) in enumerate(eqs)

        lhs, rhs = TO.getlhs(eq), TO.getrhs(eq)

        ts_lhs = TO.gettensors(lhs)
        ts_rhs = TO.gettensors(rhs)
        lhs_heads_idxs = TO.decomposetensor.(ts_lhs)
        rhs_heads_idxs = TO.decomposetensor.(ts_rhs)

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

                # make sure all indices used were defined with @index
                for idx in idxs
                    if !(idx in idx_names) && !(idx isa Integer)
                        error("@components: unknown index '$idx' found in '$(eqs[nr])'; you need to declare '$idx' with @index first")
                    end
                end

                push!(tensorheads, head)
                push!(tensoridxs, idxs)
                push!(linenrs, nr)
            end
        end
    end

    return tensorheads, tensoridxs
end


"""
    gather_scalar_variables(eqs)

Return list of all variables appearing in scalar factors that multiply any tensors.
"""
function gather_scalar_variables(eqs)
    vars = Vector{Symbol}[]
    for eq in eqs
        lhs = TO.getlhs(eq)
        scalars = getscalars(lhs)
        @assert length(scalars) == 0
        rhs = TO.getrhs(eq)
        scalars = getscalars(rhs)
        vs = reduce(vcat, getvariables.(scalars), init=Symbol[])
        unique!(vs)
        push!(vars, vs)
    end
    return vars
end


"""
    resolve_name_clashes!(uidxs, uheads, uscalarvars)

Avoid name clashes between indices and tensor heads and scalar variables.
"""
function resolve_name_clashes!(uidxs, uheads, uscalarvars)
    # TODO Could use gensym to allow using the same name for tensor heads, index and scalar variable.
    # Duplicated names between tensors and variables are undesired, but it can be useful for
    # indices, e.g. a[a,b] = b[a,c] * b[c,a].
    if any(i -> i in uheads, uidxs) || any(i -> i in uscalarvars, uidxs) || any(h -> h in uscalarvars, uheads)
        msg = "@components: found duplicated symbols "
        dups1 = [ idx for idx in uidxs if idx in uheads ]
        length(dups1) > 0 && (msg *= "between indices and tensors ('$(join(dups1,','))'), ")
        dups2 = [ idx for idx in uidxs if idx in uscalarvars ]
        length(dups2) > 0 && (msg *= "between indices and scalar variables ('$(join(dups2,','))'), ")
        dups3 = [ var for var in uscalarvars if var in uheads ]
        length(dups3) > 0 && (msg *= "between tensors and scalar variables ('$(join(dups3,','))'), ")
        error(rstrip(msg[1:end-length(", ")]))
    end
end


"""
    determine_independents(eqs)

Walk through equations and determine tensors which we have not been computed before
they are used. E.g. returns all tensors that did not appear in a lhs before they first
appeared in a rhs.
"""
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


"""
    group_indices_by_slot(heads, idxpairs)

Given some `heads` and a list of indices `idxpairs` each head is used with,
find for each head the unique indices used for each of its slot.

Assumes that all `idxpairs` have consistent rank wrt their heads.

# Example

```julia
heads = [ :A, :B, :A ]
idxpairs = [ [:i,:j], [:k], [:i,:a] ]
group_indices_by_slot(heads, idxpairs)
[ :A, :B ], [ [[:i],[:j,:a]], [:k] ]
```
"""
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
    lhs_tensor = TO.decomposetensor.(TO.gettensors(lhs))[1]
    rhs_tensors = TO.decomposetensor.(TO.gettensors(rhs))

    lhs_head, lhs_idxs = lhs_tensor[1], lhs_tensor[2]
    vlhs_tensor = :($(Symbol(:v_,lhs_head)) = view($lhs_head, $(lhs_idxs...)))
    vrhs_tensors = [ :($(Symbol(:v_,head)) = view($head, $(idxs...))) for
                     (head,idxs,_) in rhs_tensors ]

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
    letargs = [ :($lhs_head = deepcopy($(Symbol(:v_,lhs_head)))) ]
    append!(letargs, :($head = $(Symbol(:v_,head))) for (head,_,_) in rhs_tensors)
    ret_lhs = Symbol(:ret_, lhs_head)
    let_tensor_expr = :($ret_lhs = let $(letargs...); @tensor $eq; $lhs_head end)

    # unpack the results
    # TODO only unpack the independent components of the lhs_tensor
    # components_expr = :(components = Basic[])
    components_expr = :(zip($lhs_head[:], $ret_lhs[:]))

    code = quote
        $vlhs_tensor
        $(vrhs_tensors...)
        $let_tensor_expr
        $components_expr
    end

    return code
end
