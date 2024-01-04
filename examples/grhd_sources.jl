using TensorComponents

# comps = @macroexpand1 @components begin
# comps = @macroexpand @components begin
# comps = @components begin
#     @index a, b, c, d, k = 3
#     @symmetry dg[k,a,b] = dg[k,b,a]
#     @symmetry Γ[k,a,b]  = Γ[k,b,a]
#     @symmetry g[a,b]    = g[b,a]
#     @symmetry invg[a,b] = invg[b,a]
#     @symmetry T[a,b]    = T[b,a]
#     detg      = det(g[a,b])
#     invg[a,b] = adjugate(g[a,b]) / detg
#     logα[a]   = log(α[a])
#     Γ[k,a,b]  = 1/2 * invg[k,c] * (dg[a,c,b] + dg[b,a,c] - dg[c,a,b])
#     s_S[k]    = T[a,b] * (dg[k,a,b] - Γ[d,a,b] * g[d,k])
#     s_τ       = T[a,1] * logα[a] - T[a,b] * Γ[1,a,b]
# end
# outs = [:s_τ, :s_S1, :s_S2, :s_S3]
# @generate_code(comps, outs)
# @test_code(comps, outs)

# comps = @macroexpand1 @components begin
# comps = @macroexpand @components begin
comps = @components begin
    @index i, j, k, l   = 3
    @index μ, ν, λ = 4
    @symmetry hΓ[k,i,j] = hΓ[k,j,i]
    @symmetry K[i,j]    = K[j,i]
    @symmetry Tuu[i,j]  = Tuu[j,i]
    @symmetry ∂γ[i,j,k] = ∂γ[i,k,j]
    @symmetry γ[i,j]    = γ[j,i]
    @symmetry g[μ,ν]    = g[ν,μ]
    Tud[μ,ν] = Tuu[μ,λ] * g[λ,ν]
    s_S[i] = -1 * Tuu[1,1] * α * ∂α[i] + Tud[1,k] * (∂β[i,k] + hΓ[k,i,j] * β[j]) + 1/2 * ( Tuu[1,1] * β[j] * β[k] + 2 * Tuu[1,j] * β[k] + Tuu[j,k] ) * (∂γ[i,j,k] - 2*hΓ[l,i,j]*γ[l,k])
    s_tau  = Tuu[1,1] * (β[i]*β[j]*K[i,j] - β[i]*∂α[i]) + Tuu[1,i] * (2*β[j]*K[i,j] - ∂α[i] ) + Tuu[i,j]*K[i,j]
end

outs = [ :s_S1, :s_S2, :s_S3, :s_tau ]
@generate_code(comps, outs)
@test_code(comps, outs)
