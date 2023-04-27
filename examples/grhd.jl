using TensorComponents
using SymEngine


# comps = @macroexpand1 @components begin
comps = @components begin

    @index a, b, c, d, k = 3
    @symmetry dg[k,a,b] = dg[k,b,a]
    @symmetry Γ[k,a,b] = Γ[k,b,a]
    @symmetry g[a,b] = g[b,a]
    @symmetry T[a,b] = T[b,a]
    # TODO Allow functions on RHS
    # logα[a] = log(α[i])
    s_S[k] = T[a,b] * (dg[k,a,b] - Γ[d,a,b] * g[d,k])
    s_τ    = T[a,1] * logα[a] - T[a,b] * Γ[1,a,b]

end

display(comps)
display(TC.count_operations(comps))
