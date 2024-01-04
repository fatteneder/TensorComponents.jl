using LinearAlgebra
using TensorComponents
using SymEngine

@components begin

    @index a, b, c, d, e = 4

    oox = 1/x
    # Aud[a,b] = - Delta[a,1] * Delta[b,2] - Delta[a,2] * Delta[b,1]

end
# (* derivative of rotation vector e_phi times -1 and times -1/x *)
# Aud[a_,b_] := - (delta[a,1] delta[b,2] - delta[a,2] delta[b,1]);
# Bud[a_,b_] := oox Aud[a,b];
#
# (* parity under (x,y) -> (-x,-y)
#    unfinished, doing this by hand below
# *)
# P[a_,b_] :=   delta[a,0] delta[b,0] + delta[a,3] delta[b,3]
#             - delta[a,1] delta[b,1] - delta[a,2] delta[b,2];
#
#
#
#
# (* compute in this order *)
# tocompute = {
#   x == xcart,
#
#   (* on axis *)
#   Cif == "dequal(x,0)",
#
#     ds[2] == 0,
#
#     dv[2,a] == Aud[a,b] dv[1,b],
#
#     dg[2,a,b] == Aud[a,c] dg[1,c,b] + Aud[b,c] dg[1,a,c],
#     dg[2,1,1] == 0,
#     dg[2,1,2] == 0,
#     dg[2,2,2] == 0,
#
#     (* for m_ab *)
#     dm[2,a,b] == Aud[a,c] dm[1,c,b] + Aud[b,c] dm[1,a,c],
#
