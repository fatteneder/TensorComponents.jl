# topological sort -- Kahn's algorithm
# see https://en.wikipedia.org/wiki/Topological_sorting
function topological_sort(nodes::Vector{T}, childs::Vector{Vector{T}}) where T

    # find set with no incoming edges/parents
    allchilds = unique!(reduce(vcat, childs))
    seq_noparents = Int[]
    for (i,n) in enumerate(nodes)
        if !(n in allchilds)
            push!(seq_noparents, i)
        end
    end

    tmpchilds = deepcopy(childs)

    seq = Int[] # sorted sequence
    while !isempty(seq_noparents)

        i = pop!(seq_noparents)
        push!(seq, i)

        # cut all ties to this node
        chs = copy(tmpchilds[i])
        empty!(tmpchilds[i])

        # add any of the dropped childrens which now have no parents to queue
        for c in chs
            if any(tmpchs -> c in tmpchs, tmpchilds)
                continue
            end
            j = findfirst(==(c), nodes)
            # ignore any nodes which are not in the graph
            isnothing(j) && continue
            push!(seq_noparents, j)
        end
    end

    n_rest_childs = mapreduce(sum, +, tmpchilds)
    if n_rest_childs != 0
        error("Failed to sort, because graph has at least on cycle")
    end

    return seq
end


# analog to TO.gettensors, but to extract all the scalar factors
# that multiply a (general)tensor in an expression
function getcoeffs(expr)
    scalars = Any[]
    _getcoeffs!(expr, scalars)
    return scalars
end


function _getcoeffs!(expr, scalars)
    if TO.istensor(expr)
        # nothing to do
    elseif TO.isscalarexpr(expr)
        push!(scalars, expr)
    elseif TO.isgeneraltensor(expr)
        _, _, _, s, _ = TO.decomposegeneraltensor(expr)
        push!(scalars, s)
    elseif expr.head === :call && expr.args[1] in (:+,:-,:*,:/)
        foreach(ex -> _getcoeffs!(ex, scalars), expr.args[2:end])
    else
        error("Found unknown expression: '$expr'")
    end
    return
end


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


# From https://stackoverflow.com/a/16464566
function isevenperm(perm)
    isperm(perm) || error()
    count = 0
    n = length(perm)
    for i = 1:n, j=i+1:n
        if perm[i] > perm[j]
            count += 1
        end
    end
    return count % 2 == 0
end
isoddperm(perm) = !isevenperm(perm)


# From https://discourse.julialang.org/t/all-permutations-example-in-erlang/71380
perms(l) = isempty(l) ? [l] : [[x; y] for x in l for y in perms(setdiff(l, x))]


"""
    epsilon_symbol(N)

A tensor representation of the total antisymmetric Levi-Civita
symbol in `N` dimensions.
"""
function epsilon_symbol(N)
    ϵ = zeros(Int, (N for _ = 1:N)...)
    allperms = perms(collect(1:N))
    for p in allperms
        ϵ[p...] = isevenperm(p) ? 1 : -1
    end
    return ϵ
end


"""
    delta_symbol(N)

A tensor representation of the Kronecker delta symbol in `N` dimensions.
"""
delta_symbol(N) = diagm(ones(Int,N))
