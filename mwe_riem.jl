using TensorComponents
using SymEngine
using LinearAlgebra
using RowEchelon

# function ispoly(b::Basic)
#     vars = SymEngine.CSetBasic()
#     # basic_is_symbol not exported by SymEngine C++ library
#     return ccall((:basic_is_polynomial, SymEngine.libsymengine), Bool, (Ref{Basic},Ref{SymEngine.CSetBasic}), b, vars)
# end

function resolve_dependents(tensor, equation; debug=false)

    v_tensor = tensor[:]
    v_equation = equation[:]

    coeffs = zeros(Int64, length(v_tensor), length(v_tensor))
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
            coeffs[row, col] = SymEngine.coeff(eq, t)
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
    echelon_coeffs = convert.(Int64,rref(coeffs))
    red_coeffs = echelon_coeffs[1:rk,:]

    # reassmble equations
    red_eqs = [ dot(v_tensor, row) for row in eachrow(red_coeffs) ]
    # determine and order variables involved in those equations
    # allvars = [ SymEngine.free_symbols(eq) for eq in red_eqs ]
    allvars = mapreduce(SymEngine.free_symbols, vcat, red_eqs)
    # uvars = unique(reduce(vcat, allvars))
    uvars = unique(allvars)
    str_uvars = string.(uvars)
    perm = sortperm(str_uvars)
    uvars = uvars[perm]

    # determine dependent variables by solving equations one after another
    deps = SymEngine.Basic[]
    subs = SymEngine.Basic[]
    for eq in red_eqs

        # gather remaining variables
        vars = SymEngine.free_symbols(eq)
        nvars = length(vars)

        # solve equation
        dep, sub = if nvars == 1
            # coeff * var = 0 => var = 0
            first(vars), SymEngine.Basic(0)
        elseif nvars > 1
            next = findfirst(!in(deps), vars)
            if isnothing(next)
                error("This should not have happened!")
            end
            dep = vars[something(next)]
            coeff_dep = SymEngine.coeff(eq, dep)
            sub = SymEngine.expand(- eq + coeff_dep * dep)
            dep, sub
        end

        push!(deps, dep)
        push!(subs, sub)
    end

    @assert length(deps) == rk

    if debug
        println("debug")
        display(equation)
        display(rk)
        display(red_eqs)
        display(red_coeffs)
        display(sum(red_coeffs,dims=2))
        display(deps)
        display(subs)
        display(ideps)
    end

    # assemble reduced tensor
    redtensor = deepcopy(tensor)
    # for (dep, dep_sub) in zip(deps, deps_sub)
    #     idx = something(findfirst(==(dep), v_tensor))
    #     redtensor[idx] = dep_sub
    # end
    # display(deps)
    for idx = 1:length(redtensor)
        for (d, s) in zip(deps, subs)
            redtensor[idx] = SymEngine.subs(redtensor[idx], d, s)
        end
        redtensor[idx] = SymEngine.expand(redtensor[idx])
    end

    ideps = setdiff(uvars, deps)

    return redtensor
end

N = 3
Riem = SymbolicTensor(:R,N,N,N,N)

bianchi = permutedims(Riem,(1,2,3,4)) + permutedims(Riem,(1,3,4,2)) + permutedims(Riem,(1,4,2,3))
Riem = resolve_dependents(Riem, bianchi)
asym1 = permutedims(Riem,(1,2,3,4)) + permutedims(Riem,(2,1,3,4))
Riem = resolve_dependents(Riem, asym1)
asym2 = permutedims(Riem,(1,2,3,4)) + permutedims(Riem,(1,2,4,3))
display(asym2)
Riem = resolve_dependents(Riem, asym2)
# Riem = resolve_dependents(Riem, asym2, debug=true)
# bianchi = permutedims(Riem,(1,2,3,4)) + permutedims(Riem,(1,3,4,2)) + permutedims(Riem,(1,4,2,3))
# Riem = resolve_dependents(Riem, bianchi)

# display(Riem)
ideps = [ SymEngine.free_symbols(R) for R in Riem[:] ]
uideps = unique(reduce(vcat, ideps))
display(uideps)
# display(length(uideps))
n_ideps = Int(N^2*(N^2-1)/12)
display(n_ideps)
