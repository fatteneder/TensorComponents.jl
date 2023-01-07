using TensorComponents
using SymEngine
using LinearAlgebra
using RowEchelon


# function ispoly(b::Basic)
#     vars = SymEngine.CSetBasic()
#     # basic_is_symbol not exported by SymEngine C++ library
#     return ccall((:basic_is_polynomial, SymEngine.libsymengine), Bool, (Ref{Basic},Ref{SymEngine.CSetBasic}), b, vars)
# end


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


function test_topological_sort()
    nodes  = [ 10, 8, 5, 3, 2, 11, 9, 7 ]
    childs = Vector{Int}[ [], [9,10], [11], [8], [], [2,9], [], [11,8] ]
    seq = topological_sort(nodes, childs)
    sorted_nodes = nodes[seq]
end


function resolve_dependents(tensor, equation; debug=false)

    v_tensor = tensor[:]
    v_equation = equation[:]

    coeffs = zeros(Rational, length(equation), length(v_tensor))
    for (row, eq) in enumerate(v_equation)
        free_syms = SymEngine.free_symbols(eq)
        for (col, t) in enumerate(v_tensor)
            if t == 0 || !(t in free_syms)
                continue
            end
            coeff = SymEngine.coeff(eq, t)
            if length(SymEngine.free_symbols(coeff)) > 0
                # TODO How to check for polynomials?
                # e.g. x + x^2 will slip through here
                error("Rules must be linear!")
            end
            coeffs[row, col] = coeff
        end
    end

    # number of linear independent equations to consider for
    # determining dependent components
    rk = rank(coeffs)
    if rk == 0
        @info "Equation does not provide any constraints"
        return deepcopy(tensor)
    end

    # reduce coeff matrix to row echelon form
    rref!(coeffs)
    echelon_coeffs = convert.(Int64, coeffs)
    red_coeffs = echelon_coeffs[1:rk,:]

    # reassmble equations
    red_eqs = [ dot(v_tensor, row) for row in eachrow(red_coeffs) ]

    # determine dependent variables by solving equations one after another
    deps = SymEngine.Basic[]
    subs = SymEngine.Basic[]
    deps_zeros = SymEngine.Basic[]
    for eq in red_eqs

        # gather remaining variables
        vars = SymEngine.free_symbols(eq)
        nvars = length(vars)

        # solve equation
        if nvars == 1
            # coeff * var = 0 => var = 0
            push!(deps_zeros, first(vars))
            continue
        end

        # nvars > 1
        str_vars = string.(vars)
        p = sortperm(str_vars)
        vars = vars[p]
        next = findfirst(!in(deps), reverse!(vars))
        if isnothing(next)
            error("This should not have happened!")
        end
        dep = vars[something(next)]
        if dep in deps
            error("$dep duplicated")
        end
        coeff_dep = SymEngine.coeff(eq, dep)
        sub = SymEngine.expand(- eq + coeff_dep * dep)

        push!(deps, dep)
        push!(subs, sub)
    end

    @assert length(deps) + length(deps_zeros) == rk

    # topologically sort dependent variables such that when subsituting in
    # order only independent variables remain
    deps_vars = [ SymEngine.free_symbols(s) for s in subs ]
    perm = topological_sort(deps, deps_vars)

    deps = deps[perm]
    subs = subs[perm]

    # assemble reduced tensor
    redtensor = deepcopy(tensor)
    for idx = 1:length(redtensor)
        for (d, s) in zip(deps, subs)
            redtensor[idx] = SymEngine.subs(redtensor[idx], d, s)
        end
        for d in deps_zeros
            redtensor[idx] = SymEngine.subs(redtensor[idx], d, 0)
        end
        redtensor[idx] = SymEngine.expand(redtensor[idx])
    end

    return redtensor
end

N = 4
Riem = SymbolicTensor(:R,N,N,N,N)

function test_resolve_dependents(Riem)
    bianchi = permutedims(Riem,(1,2,3,4)) + permutedims(Riem,(1,3,4,2)) + permutedims(Riem,(1,4,2,3))
    Riem = resolve_dependents(Riem, bianchi)
    asym1 = permutedims(Riem,(1,2,3,4)) + permutedims(Riem,(2,1,3,4))
    Riem = resolve_dependents(Riem, asym1)
    asym2 = permutedims(Riem,(1,2,3,4)) + permutedims(Riem,(1,2,4,3))
    Riem = resolve_dependents(Riem, asym2)
    return Riem
end

red_Riem = test_resolve_dependents(Riem)
ideps = [ SymEngine.free_symbols(R) for R in red_Riem[:] ]
uideps = unique!(reduce(vcat, ideps))
n_ideps = Int(N^2*(N^2-1)/12)
@show length(uideps) == n_ideps
