import sympy as sp
import sympy_utils as spu


def christoffels(coords, g, invg):
    assert(len(coords) == 4)
    assert(g.shape     == (4,4))
    assert(invg.shape  == (4,4))
    N = len(coords)
    gammas = [ sp.zeros(*g.shape) for i in range(N) ]
    for (k,gamma_k) in enumerate(gammas):
        for i in range(N):
            for j in range(N):
                for l in range(N):
                    gamma_k[i,j] += sp.Rational(1,2) * invg[k,l] * \
                        (sp.diff(g[j,l],coords[i]) + sp.diff(g[i,l],coords[j]) - sp.diff(g[i,j],coords[l]))         
    return gammas


def ricci_tensor(coords, g, gammas):
    N = len(coords)
    Ric = sp.zeros(*g.shape)
    for j in range(N):
        for k in range(N):
            if j > k:
                Ric[j,k] = Ric[k,j]
            else:
                for i in range(N):
                    gamma_i = gammas[i]
                    Ric[j,k] += sp.diff(gamma_i[j,k],coords[i]) - sp.diff(gamma_i[j,i],coords[k])
                    for p in range(N):
                        gamma_p = gammas[p]
                        Ric[j,k] += gamma_i[i,p] * gamma_p[j,k] - gamma_i[k,p] * gamma_p[i,j]
                Ric[j,k] = sp.expand(Ric[j,k])
    return Ric


def ricci_scalar(coords, invg, Ric):
    N = len(coords)
    R = 0
    for i in range(N):
        for j in range(N):
            R += invg[i,j] * Ric[i,j]
    return R

def einstein_tensor(g, Ric, R):
    N, _ = g.shape
    G = sp.zeros(*g.shape)
    for i in range(N):
        for j in range(N):
            if i > j:
                G[i,j] = G[j,i]
            else:
                G[i,j] = Ric[i,j] - sp.Rational(1,2) * g[i,j] * R
                G[i,j] = sp.expand(G[i,j])
    return G

# +
# def isvalid_valence(valence):
#     for v in valence:
#         if v != 'd' or v != 'u':
#             return false
#     return true

# def covdiff(f, valence, coord, coords, gammas):
#     if not val is list:
#         raise Exception("Argument `valence` must be a vector indicating index positions of `f`.")
#     if not isvalid_valence(valence):
#         raise Expeption("`valence` elements can either be 'd' (down) or 'u' (up).")
#     result = sp.diff(f, coord)
#     if len(val) == 0:
#         # scalar is ordinary diff
#         return result
#     for v in valence:
#         if v == 'd':
#             for (i,c) in enumerate(coords):
#                 gi = gammas[i]
#                 result += gi * f[i]
#         else: # v == 'u'
#             for (i, c) in enumerate(coords):
#                 gi = gammas[i]:
                
# -


