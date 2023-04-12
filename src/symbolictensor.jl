mutable struct SymbolicTensor{N} <: DenseArray{Basic,N}
    head::Symbol
    comp::Array{Basic,N}
    deps::Vector{Basic}
    ideps::Vector{Basic}
    coeff::Basic
end


#######################################################################
#                      Abstract Array Interface                       #
#######################################################################
# https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array

Base.size(t::SymbolicTensor) = size(t.comp)
Base.getindex(t::SymbolicTensor, i::Int) = t.comp[i] * t.coeff
Base.getindex(t::SymbolicTensor, I::Vararg{Int,N}) where N = t.comp[I...] * t.coeff
Base.setindex!(t::SymbolicTensor, v, i::Int) = t.comp[i] = v
Base.setindex!(t::SymbolicTensor, v, I::Vararg{Int,N}) where N = t.comp[I...] = v


#######################################################################
#                            Constructors                             #
#######################################################################


function SymbolicTensor(head::Symbol, coeff::Basic, dims::Int64...)
    all(d -> d > 0, dims) || throw(ArgumentError("Dimensions must be positive, found: $(dims...)"))
    cidxs = CartesianIndices(dims)
    comps = [ symbols("$(head)$(join(Tuple(c)))") for c in cidxs ]
    ideps = deepcopy(comps[:])
    deps  = deepcopy(comps[:])
    return SymbolicTensor(head, comps, deps, ideps, coeff)
end


SymbolicTensor(head::Symbol, dims::Int64...) = SymbolicTensor(head, Basic(1), dims...)


#######################################################################
#                               Methods                               #
#######################################################################


# Should be done upstream in SymEgine: https://github.com/symengine/SymEngine.jl/pull/260
Base.promote_rule(::Type{Bool}, ::Type{Basic}) = Basic


# Overload scalar operations *,/,\,+,- to capture common coefficients, if possible.
for f in (:*, :/, :\)
    if f !== :\
        @eval function Base.$f(a::Union{Number,Basic}, t::SymbolicTensor)
            nt = deepcopy(t)
            nt.coeff = ($f)(nt.coeff, a)
            return nt
        end
    end
    if f !== :/
        @eval function Base.$f(t::SymbolicTensor, a::Union{Number,Basic})
            nt = deepcopy(t)
            nt.coeff = ($f)(a, nt.coeff)
            return nt
        end
    end
end


for f in (:+, :-)
    @eval function Base.$f(t::SymbolicTensor, s::AbstractArray)
        promote_rule(t,s)
        nt = deepcopy(t)
        nt.comp = $f(t.coeff * t.comp, s)
        return nt
    end
    # @eval Base.$f(t::SymbolicTensor, s::AbstractArray) = $f(s,t)
    @eval function Base.$f(t::SymbolicTensor, s::SymbolicTensor)
        # promote_rule(t,s)
        nt = deepcopy(t)
        if t.coeff == s.coeff
            nt.comp = $f(t.comp, s.comp)
        else
            nt.comp = $f(t.coeff * t.comp, s.coeff * s.comp)
        end
        return nt
    end
end


function Base.show(io::IO, ::MIME"text/plain", t::SymbolicTensor)
    println(io, t.coeff, " ×")
    show(io, MIME"text/plain"(), t.comp)
end


function extract_coefficient_matrix(equation, tensor)

    coeffs = zeros(Float64, length(equation), length(tensor))

    v_equation = view(equation, :)
    v_tensor = view(tensor, :)

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

    return coeffs
end


# function ispoly(b::Basic)
#     vars = SymEngine.CSetBasic()
#     # basic_is_symbol not exported by SymEngine C++ library
#     return ccall((:basic_is_polynomial, SymEngine.libsymengine), Bool, (Ref{Basic},Ref{SymEngine.CSetBasic}), b, vars)
# end


function resolve_dependents(tensor, equation)

    coeffs = extract_coefficient_matrix(equation, tensor)

    # number of linear independent equations to consider for determining dependent components
    rk = rank(coeffs)
    if rk == 0
        # TODO Should display also relevant line
        @warn "Equation does not provide any constraints"
        return deepcopy(tensor)
    end

    # reduce coeff matrix to row echelon form
    # this is slow because it uses Gaussian elimination
    RowEchelon.rref!(coeffs)
    echelon_coeffs = convert.(Rational{Int64}, coeffs)
    red_coeffs = view(echelon_coeffs, 1:rk, :)

    # re-assemble equations
    v_tensor = view(tensor, :)
    # assembling using a scalar product (dot) between v_tensor and each row of red_coeffs can
    # be slow, because it must call into SymEngine for every element.
    # red_coeffs is usually sparse, so we exploit that here to speed things up.
    red_eqs = Basic[]
    for row in eachrow(red_coeffs)
        eq = Basic(0)
        for (i,c) in enumerate(row)
            isapprox(c, 0) && continue
            eq += c * v_tensor[i]
        end
        push!(red_eqs, eq)
    end

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
            throw(ErrorException("This should not have happened!"))
        end
        dep = vars[something(next)]
        if dep in deps
            throw(ErrorException("$dep duplicated"))
        end
        coeff_dep = SymEngine.coeff(eq, dep)
        sub = SymEngine.expand(- eq + coeff_dep * dep)

        push!(deps, dep)
        push!(subs, sub)
    end

    @assert length(deps) + length(deps_zeros) == rk

    # topologically sort dependent variables such that when substituting
    # really only independent variables remain
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
