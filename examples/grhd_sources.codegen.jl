# Generated on 2023-04-30 - 20:57:07.949
# Operator counts
#   * = 236 ...  57.6 %
#   + = 118 ...  28.8 %
#   - =  41 ...  10.0 %
#   ^ =   6 ...   1.5 %
#   / =   6 ...   1.5 %
# log =   3 ...   0.7 %
#   Σ = 410 ... 100.0 %

# detected  input  variables: 'T11,T12,T13,T22,T23,T33,dg111,dg112,dg113,dg122,dg123,dg133,dg211,dg212,dg213,dg222,dg223,dg233,dg311,dg312,dg313,dg322,dg323,dg333,g11,g12,g13,g22,g23,g33,α1,α2,α3'
# requested output variables: 's_τ,s_S1,s_S2,s_S3'


function grhd_sources(CACHE, N)
    for IDX = 1:N
        T11 = CACHE.T11[IDX]
        T12 = CACHE.T12[IDX]
        T13 = CACHE.T13[IDX]
        T22 = CACHE.T22[IDX]
        T23 = CACHE.T23[IDX]
        T33 = CACHE.T33[IDX]
        dg111 = CACHE.dg111[IDX]
        dg112 = CACHE.dg112[IDX]
        dg113 = CACHE.dg113[IDX]
        dg122 = CACHE.dg122[IDX]
        dg123 = CACHE.dg123[IDX]
        dg133 = CACHE.dg133[IDX]
        dg211 = CACHE.dg211[IDX]
        dg212 = CACHE.dg212[IDX]
        dg213 = CACHE.dg213[IDX]
        dg222 = CACHE.dg222[IDX]
        dg223 = CACHE.dg223[IDX]
        dg233 = CACHE.dg233[IDX]
        dg311 = CACHE.dg311[IDX]
        dg312 = CACHE.dg312[IDX]
        dg313 = CACHE.dg313[IDX]
        dg322 = CACHE.dg322[IDX]
        dg323 = CACHE.dg323[IDX]
        dg333 = CACHE.dg333[IDX]
        g11 = CACHE.g11[IDX]
        g12 = CACHE.g12[IDX]
        g13 = CACHE.g13[IDX]
        g22 = CACHE.g22[IDX]
        g23 = CACHE.g23[IDX]
        g33 = CACHE.g33[IDX]
        α1 = CACHE.α1[IDX]
        α2 = CACHE.α2[IDX]
        α3 = CACHE.α3[IDX]
        detg = (g11 * g22 * g33 + 2 * g12 * g23 * g13) - (g11 * g23 ^ 2 + g12 ^ 2 * g33 + g13 ^ 2 * g22)
        invg11 = (g22 * g33 - g23 ^ 2) / detg
        invg12 = -((g12 * g33 - g23 * g13)) / detg
        invg13 = (g12 * g23 - g13 * g22) / detg
        invg22 = (g11 * g33 - g13 ^ 2) / detg
        invg23 = -((g11 * g23 - g12 * g13)) / detg
        invg33 = (g11 * g22 - g12 ^ 2) / detg
        logα1 = log(α1)
        logα2 = log(α2)
        logα3 = log(α3)
        Γ111 = 0.5 * dg111 * invg11 + 0.5 * (2dg112 - dg211) * invg12 + 0.5 * (2dg113 - dg311) * invg13
        Γ211 = 0.5 * dg111 * invg12 + 0.5 * (2dg112 - dg211) * invg22 + 0.5 * (2dg113 - dg311) * invg23
        Γ311 = 0.5 * dg111 * invg13 + 0.5 * (2dg112 - dg211) * invg23 + 0.5 * (2dg113 - dg311) * invg33
        Γ112 = 0.5 * dg122 * invg12 + 0.5 * dg211 * invg11 + 0.5 * ((dg123 + dg213) - dg312) * invg13
        Γ212 = 0.5 * dg122 * invg22 + 0.5 * dg211 * invg12 + 0.5 * ((dg123 + dg213) - dg312) * invg23
        Γ312 = 0.5 * dg122 * invg23 + 0.5 * dg211 * invg13 + 0.5 * ((dg123 + dg213) - dg312) * invg33
        Γ113 = 0.5 * dg133 * invg13 + 0.5 * dg311 * invg11 + 0.5 * ((dg123 - dg213) + dg312) * invg12
        Γ213 = 0.5 * dg133 * invg23 + 0.5 * dg311 * invg12 + 0.5 * ((dg123 - dg213) + dg312) * invg22
        Γ313 = 0.5 * dg133 * invg33 + 0.5 * dg311 * invg13 + 0.5 * ((dg123 - dg213) + dg312) * invg23
        Γ122 = 0.5 * dg222 * invg12 + 0.5 * (-dg122 + 2dg212) * invg11 + 0.5 * (2dg223 - dg322) * invg13
        Γ222 = 0.5 * dg222 * invg22 + 0.5 * (-dg122 + 2dg212) * invg12 + 0.5 * (2dg223 - dg322) * invg23
        Γ322 = 0.5 * dg222 * invg23 + 0.5 * (-dg122 + 2dg212) * invg13 + 0.5 * (2dg223 - dg322) * invg33
        Γ123 = 0.5 * dg233 * invg13 + 0.5 * dg322 * invg12 + 0.5 * (-dg123 + dg213 + dg312) * invg11
        Γ223 = 0.5 * dg233 * invg23 + 0.5 * dg322 * invg22 + 0.5 * (-dg123 + dg213 + dg312) * invg12
        Γ323 = 0.5 * dg233 * invg33 + 0.5 * dg322 * invg23 + 0.5 * (-dg123 + dg213 + dg312) * invg13
        Γ133 = 0.5 * dg333 * invg13 + 0.5 * (-dg133 + 2dg313) * invg11 + 0.5 * (-dg233 + 2dg323) * invg12
        Γ233 = 0.5 * dg333 * invg23 + 0.5 * (-dg133 + 2dg313) * invg12 + 0.5 * (-dg233 + 2dg323) * invg22
        Γ333 = 0.5 * dg333 * invg33 + 0.5 * (-dg133 + 2dg313) * invg13 + 0.5 * (-dg233 + 2dg323) * invg23
        s_S1 = T11 * (dg111 - (g11 * Γ111 + g12 * Γ211 + g13 * Γ311)) + 2 * T12 * (dg112 - (g11 * Γ112 + g12 * Γ212 + g13 * Γ312)) + 2 * T13 * (dg113 - (g11 * Γ113 + g12 * Γ213 + g13 * Γ313)) + T22 * (dg122 - (g11 * Γ122 + g12 * Γ222 + g13 * Γ322)) + 2 * T23 * (dg123 - (g11 * Γ123 + g12 * Γ223 + g13 * Γ323)) + T33 * (dg133 - (g11 * Γ133 + g12 * Γ233 + g13 * Γ333))
        s_S2 = T11 * (dg211 - (g12 * Γ111 + g22 * Γ211 + g23 * Γ311)) + 2 * T12 * (dg212 - (g12 * Γ112 + g22 * Γ212 + g23 * Γ312)) + 2 * T13 * (dg213 - (g12 * Γ113 + g22 * Γ213 + g23 * Γ313)) + T22 * (dg222 - (g12 * Γ122 + g22 * Γ222 + g23 * Γ322)) + 2 * T23 * (dg223 - (g12 * Γ123 + g22 * Γ223 + g23 * Γ323)) + T33 * (dg233 - (g12 * Γ133 + g22 * Γ233 + g23 * Γ333))
        s_S3 = T11 * (dg311 - (g13 * Γ111 + g23 * Γ211 + g33 * Γ311)) + 2 * T12 * (dg312 - (g13 * Γ112 + g23 * Γ212 + g33 * Γ312)) + 2 * T13 * (dg313 - (g13 * Γ113 + g23 * Γ213 + g33 * Γ313)) + T22 * (dg322 - (g13 * Γ122 + g23 * Γ222 + g33 * Γ322)) + 2 * T23 * (dg323 - (g13 * Γ123 + g23 * Γ223 + g33 * Γ323)) + T33 * (dg333 - (g13 * Γ133 + g23 * Γ233 + g33 * Γ333))
        s_τ = (T11 * logα1 + T12 * logα2 + T13 * logα3) - (T11 * Γ111 + 2 * T12 * Γ112 + 2 * T13 * Γ113 + T22 * Γ122 + 2 * T23 * Γ123 + T33 * Γ133)
        CACHE.s_τ[IDX] = s_τ
        CACHE.s_S1[IDX] = s_S1
        CACHE.s_S2[IDX] = s_S2
        CACHE.s_S3[IDX] = s_S3
    end
end
