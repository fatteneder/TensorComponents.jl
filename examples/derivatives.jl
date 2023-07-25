using LinearAlgebra
using TensorComponents
using SymEngine


# comps = @macroexpand1 @components begin
# comps = @macroexpand @components begin
comps = @components begin
    @index i, j, k, l, m = 3
    @symmetry hbar[i,j]           = hbar[j,i]
    @symmetry dhbar[j,i,k]        = dhbar[j,k,i]
    @symmetry Γbar_ddd[j,i,k]     = Γbar_ddd[j,k,i]
    @symmetry Γbar_udd[j,i,k]     = Γbar_udd[j,k,i]
    @symmetry dΓbar_dudd[l,j,i,k] = dΓbar_dudd[l,j,k,i]
    # @symmetry Rbar_uddd[l,j,i,k] + Rbar_uddd[l,i,k,j] + Rbar_uddd[l,k,j,i] = 0
    # @symmetry Rbar_uddd[l,j,i,k]                                           = - Rbar_uddd[j,l,i,k]
    # @symmetry Rbar_uddd[l,j,i,k]                                           = - Rbar_uddd[l,j,k,i]
    @symmetry ubar_uu[i,j]        = ubar_uu[i,j]

    Γbar_ddd[i,j,k]     = 1/2 * (dhbar[j,i,k] + dhbar[k,i,j] - dhbar[i,j,k])
    dΓbar_dddd[l,i,j,k] = 1/2 * (ddhbar[l,j,i,k] + ddhbar[l,k,i,j] - ddhbar[l,i,j,k])
    Γbar_udd[j,k,l]     = hbar_uu[j,m] * Γb_ddd[m,k,l]
    dΓbar_dudd[i,j,k,l] = dhbar_duu[i,j,m] * Γbar_ddd[m,k,l] + hbar_uu[j,m] * dΓbar_dddd[i,m,k,l]
    Rbar_uddd[i,j,k,l]  = dΓbar_dudd[k,i,j,l] - dΓbar_dudd[l,i,j,k] + Γbar_udd[i,k,m] * Γbar_udd[m,l,j] - Γbar_udd[i,l,m] * Γbar_udd[m,k,j]
    Rbar_ud[i,j]        = hbar_uu[i,l] * Rbar_uddd[k,l,k,j]
    Rbarsc              = Rbar_ud[i,i]
    DbarK_u[i]          = hbar_uu[i,j] * dK[j]
    Dbaralphabar[i]     = dalphabar[i]
    Dbarubar_u[i]       = dubarduu[j,j,i] + Γbar_udd[j,j,k] * ubar_uu[k,i] + Γbar_udd[i,j,k] * ubar_uu[j,k]

end


outs = [:s_τ, :s_S1, :s_S2, :s_S3]
@generate_code(comps, outs)
@test_code(comps, outs)
