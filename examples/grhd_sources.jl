using LinearAlgebra
using TensorComponents
using SymEngine


# comps = @macroexpand1 @components begin
# comps = @macroexpand @components begin
comps = @components begin
    @index a, b, c, d, k = 3
    @symmetry dg[k,a,b] = dg[k,b,a]
    @symmetry Γ[k,a,b]  = Γ[k,b,a]
    @symmetry g[a,b]    = g[b,a]
    @symmetry invg[a,b] = invg[b,a]
    @symmetry T[a,b]    = T[b,a]
    detg      = det(g[a,b])
    invg[a,b] = adjugate(g[a,b]) / detg
    logα[a]   = log(α[a])
    Γ[k,a,b]  = 1/2 * invg[k,c] * (dg[a,c,b] + dg[b,a,c] - dg[c,a,b])
    s_S[k]    = T[a,b] * (dg[k,a,b] - Γ[d,a,b] * g[d,k])
    s_τ       = T[a,1] * logα[a] - T[a,b] * Γ[1,a,b]
end


outs = [:s_τ, :s_S1, :s_S2, :s_S3]
@generate_code(comps, outs)
@test_code(comps, outs)
