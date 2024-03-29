# topological sort -- Kahn's algorithm
# see https://en.wikipedia.org/wiki/Topological_sorting
function topological_sort(nodes::Vector{T}, childs::Vector{Vector{T}}) where T

    # find set with no incoming edges/parents
    allchilds = unique!(reduce(vcat, childs, init=T[]))

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

    if !isempty(tmpchilds)
        n_rest_childs = mapreduce(sum, +, tmpchilds)
        if n_rest_childs != 0
            error("Failed to sort, because graph has at least on cycle")
        end
    end

    return seq
end


lt_symint(x::Integer,y::Symbol) = true
lt_symint(x::Integer,y::Expr)   = true
lt_symint(x::Symbol,y::Integer) = false
lt_symint(x::Expr,y::Integer)   = false
lt_symint(x::Symbol,y::Expr)    = true
lt_symint(x::Expr,y::Symbol)    = false
lt_symint(x,y) = x < y
symsort!(xs) = sort!(xs, lt=lt_symint)


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


# permutation test for non-integer containers
ispermutation(seq, catalog) = isperm(catalog[s] for s in seq)


# provide a catalog that maps a container's elements to integers, for use with ispermutation
permutation_catalog(seq) = Dict{Any,Int}( s => i for (i,s) in enumerate(seq) )


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


# SymEngine.walk_expression converts 'a - b' into ':(a + -1 * b)', so this adds
# two artifical operations which would falsifies the count
# this does not hapeen when we first convert to string and then parse as an Expr
function count_operations(expr::SymEngine.Basic)
    count_operations(Meta.parse(string(expr)))
end
function count_operations(expr::Expr)
    stats = Dict{Symbol,Int}()
    MacroTools.postwalk(expr) do ex
        !(ex isa Expr) && return ex
        if ex.head === :call
            fn = ex.args[1]
            count = get!(stats, fn, 0)
            incr = if fn in (:+,:-,:*)
                # any *,+,- calls group arguments all on one level in ex.args,
                # e.g. dump(:(a+b+c)) gives
                # Expr
                #   head: Symbol call
                #   args: Array{Any}((4,))
                #     1: Symbol +
                #     2: Symbol a
                #     3: Symbol b
                #     4: Symbol c
                length(ex.args) - 2
            else
                1
            end
            stats[fn] = count + incr
        end
        return ex
    end
    return stats
end


function count_operations(lhsrhs::AbstractVector)
    allstats = Dict{Symbol,Int}()
    for (_,rhs) in lhsrhs
        stats = count_operations(rhs)
        mergecounts!(allstats,stats)
    end
    return allstats
end


function mergecounts!(d1::AbstractDict{K,T}, d2::AbstractDict{KK,TT}) where {K,T,KK<:K,TT<:T}
    for (k,v) in pairs(d2)
        count = get(d1, k, 0)
        d1[k] = count + v
    end
    return d1
end
mergecounts!(d1::AbstractDict, d2::AbstractDict, ds::AbstractDict...) =
    mergecounts!(mergecounts!(d1,d2), ds...)

function mergecounts(d1::AbstractDict, ds::AbstractDict...)
    counts = Dict{Symbol,Int}()
    mergecounts!(counts, d1, ds...)
    return counts
end
mergecounts(d1::AbstractDict, d2::AbstractDict) = mergecounts!(Dict{Symbol,Int},d1,d2)


# cofactor matrix of A, cf. https://en.wikipedia.org/wiki/Adjugate_matrix
function cofactor(A::AbstractMatrix)
    ax = axes(A)
    out = similar(A, ax)
    for col in ax[1], row in ax[2]
        notrow = [ i for i in ax[1] if i != row ]
        notcol = [ i for i in ax[2] if i != col ]
        out[col, row] = (-1)^(col + row) * det(A[notcol, notrow])
    end
    return out
end
adjugate(A::AbstractMatrix) = transpose(cofactor(A))
inverse(A::AbstractMatrix) = det(A) * adjugate(A)


function _parse_eq(str)
    ex = Meta.parse(str)
    ex = MacroTools.prewalk(rmlines, ex)
    lhs, rhs = getlhs(ex), getrhs(ex)
    lhs, rhs
end
function parse_eqs(filename::AbstractString)
    eqs = readlines(filename)
    filter!(eqs) do line
        !startswith(line,"#") && !isempty(line)
    end
    return _parse_eq.(eqs)
end
