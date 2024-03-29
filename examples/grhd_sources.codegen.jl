# Generated on 2024-01-04 - 19:12:06.223
# Operator counts
# * = 383 ...  57.4 %
# + = 241 ...  36.1 %
# - =  31 ...   4.6 %
# ^ =  12 ...   1.8 %
# ===================
# Σ = 667 ... 100.0 %


# detected input variables:
# K11,K12,K13,K22,K23,K33,Tuu11,Tuu12,Tuu13,Tuu14,Tuu22,Tuu23,Tuu24,Tuu33,Tuu34,Tuu41,Tuu42,Tuu43,Tuu44,g11,g12,g13,g14,g22,g23,g24,g33,g34,g44,hΓ111,hΓ112,hΓ113,hΓ122,hΓ123,hΓ133,hΓ211,hΓ212,hΓ213,hΓ222,hΓ223,hΓ233,hΓ311,hΓ312,hΓ313,hΓ322,hΓ323,hΓ333,α,β1,β2,β3,γ11,γ12,γ13,γ22,γ23,γ33,∂α1,∂α2,∂α3,∂β11,∂β12,∂β13,∂β21,∂β22,∂β23,∂β31,∂β32,∂β33,∂γ111,∂γ112,∂γ113,∂γ122,∂γ123,∂γ133,∂γ211,∂γ212,∂γ213,∂γ222,∂γ223,∂γ233,∂γ311,∂γ312,∂γ313,∂γ322,∂γ323,∂γ333

# requested output variables:
# s_S1,s_S2,s_S3,s_tau


function grhd_sources(CACHE, N)
    for IDX = 1:N
        K11 = CACHE.K11[IDX]
        K12 = CACHE.K12[IDX]
        K13 = CACHE.K13[IDX]
        K22 = CACHE.K22[IDX]
        K23 = CACHE.K23[IDX]
        K33 = CACHE.K33[IDX]
        Tuu11 = CACHE.Tuu11[IDX]
        Tuu12 = CACHE.Tuu12[IDX]
        Tuu13 = CACHE.Tuu13[IDX]
        Tuu14 = CACHE.Tuu14[IDX]
        Tuu22 = CACHE.Tuu22[IDX]
        Tuu23 = CACHE.Tuu23[IDX]
        Tuu24 = CACHE.Tuu24[IDX]
        Tuu33 = CACHE.Tuu33[IDX]
        Tuu34 = CACHE.Tuu34[IDX]
        Tuu41 = CACHE.Tuu41[IDX]
        Tuu42 = CACHE.Tuu42[IDX]
        Tuu43 = CACHE.Tuu43[IDX]
        Tuu44 = CACHE.Tuu44[IDX]
        g11 = CACHE.g11[IDX]
        g12 = CACHE.g12[IDX]
        g13 = CACHE.g13[IDX]
        g14 = CACHE.g14[IDX]
        g22 = CACHE.g22[IDX]
        g23 = CACHE.g23[IDX]
        g24 = CACHE.g24[IDX]
        g33 = CACHE.g33[IDX]
        g34 = CACHE.g34[IDX]
        g44 = CACHE.g44[IDX]
        hΓ111 = CACHE.hΓ111[IDX]
        hΓ112 = CACHE.hΓ112[IDX]
        hΓ113 = CACHE.hΓ113[IDX]
        hΓ122 = CACHE.hΓ122[IDX]
        hΓ123 = CACHE.hΓ123[IDX]
        hΓ133 = CACHE.hΓ133[IDX]
        hΓ211 = CACHE.hΓ211[IDX]
        hΓ212 = CACHE.hΓ212[IDX]
        hΓ213 = CACHE.hΓ213[IDX]
        hΓ222 = CACHE.hΓ222[IDX]
        hΓ223 = CACHE.hΓ223[IDX]
        hΓ233 = CACHE.hΓ233[IDX]
        hΓ311 = CACHE.hΓ311[IDX]
        hΓ312 = CACHE.hΓ312[IDX]
        hΓ313 = CACHE.hΓ313[IDX]
        hΓ322 = CACHE.hΓ322[IDX]
        hΓ323 = CACHE.hΓ323[IDX]
        hΓ333 = CACHE.hΓ333[IDX]
        α = CACHE.α[IDX]
        β1 = CACHE.β1[IDX]
        β2 = CACHE.β2[IDX]
        β3 = CACHE.β3[IDX]
        γ11 = CACHE.γ11[IDX]
        γ12 = CACHE.γ12[IDX]
        γ13 = CACHE.γ13[IDX]
        γ22 = CACHE.γ22[IDX]
        γ23 = CACHE.γ23[IDX]
        γ33 = CACHE.γ33[IDX]
        ∂α1 = CACHE.∂α1[IDX]
        ∂α2 = CACHE.∂α2[IDX]
        ∂α3 = CACHE.∂α3[IDX]
        ∂β11 = CACHE.∂β11[IDX]
        ∂β12 = CACHE.∂β12[IDX]
        ∂β13 = CACHE.∂β13[IDX]
        ∂β21 = CACHE.∂β21[IDX]
        ∂β22 = CACHE.∂β22[IDX]
        ∂β23 = CACHE.∂β23[IDX]
        ∂β31 = CACHE.∂β31[IDX]
        ∂β32 = CACHE.∂β32[IDX]
        ∂β33 = CACHE.∂β33[IDX]
        ∂γ111 = CACHE.∂γ111[IDX]
        ∂γ112 = CACHE.∂γ112[IDX]
        ∂γ113 = CACHE.∂γ113[IDX]
        ∂γ122 = CACHE.∂γ122[IDX]
        ∂γ123 = CACHE.∂γ123[IDX]
        ∂γ133 = CACHE.∂γ133[IDX]
        ∂γ211 = CACHE.∂γ211[IDX]
        ∂γ212 = CACHE.∂γ212[IDX]
        ∂γ213 = CACHE.∂γ213[IDX]
        ∂γ222 = CACHE.∂γ222[IDX]
        ∂γ223 = CACHE.∂γ223[IDX]
        ∂γ233 = CACHE.∂γ233[IDX]
        ∂γ311 = CACHE.∂γ311[IDX]
        ∂γ312 = CACHE.∂γ312[IDX]
        ∂γ313 = CACHE.∂γ313[IDX]
        ∂γ322 = CACHE.∂γ322[IDX]
        ∂γ323 = CACHE.∂γ323[IDX]
        ∂γ333 = CACHE.∂γ333[IDX]
        Tud11 = g11 * Tuu11 + g12 * Tuu12 + g13 * Tuu13 + g14 * Tuu14
        Tud21 = g11 * Tuu12 + g12 * Tuu22 + g13 * Tuu23 + g14 * Tuu24
        Tud31 = g11 * Tuu13 + g12 * Tuu23 + g13 * Tuu33 + g14 * Tuu34
        Tud41 = g11 * Tuu41 + g12 * Tuu42 + g13 * Tuu43 + g14 * Tuu44
        Tud12 = g12 * Tuu11 + g22 * Tuu12 + g23 * Tuu13 + g24 * Tuu14
        Tud22 = g12 * Tuu12 + g22 * Tuu22 + g23 * Tuu23 + g24 * Tuu24
        Tud32 = g12 * Tuu13 + g22 * Tuu23 + g23 * Tuu33 + g24 * Tuu34
        Tud42 = g12 * Tuu41 + g22 * Tuu42 + g23 * Tuu43 + g24 * Tuu44
        Tud13 = g13 * Tuu11 + g23 * Tuu12 + g33 * Tuu13 + g34 * Tuu14
        Tud23 = g13 * Tuu12 + g23 * Tuu22 + g33 * Tuu23 + g34 * Tuu24
        Tud33 = g13 * Tuu13 + g23 * Tuu23 + g33 * Tuu33 + g34 * Tuu34
        Tud43 = g13 * Tuu41 + g23 * Tuu42 + g33 * Tuu43 + g34 * Tuu44
        Tud14 = g14 * Tuu11 + g24 * Tuu12 + g34 * Tuu13 + g44 * Tuu14
        Tud24 = g14 * Tuu12 + g24 * Tuu22 + g34 * Tuu23 + g44 * Tuu24
        Tud34 = g14 * Tuu13 + g24 * Tuu23 + g34 * Tuu33 + g44 * Tuu34
        Tud44 = g14 * Tuu41 + g24 * Tuu42 + g34 * Tuu43 + g44 * Tuu44
        s_S1 = ((Tud11 * (∂β11 + β1 * hΓ111 + β2 * hΓ112 + β3 * hΓ113) + Tud12 * (∂β12 + β1 * hΓ211 + β2 * hΓ212 + β3 * hΓ213) + Tud13 * (∂β13 + β1 * hΓ311 + β2 * hΓ312 + β3 * hΓ313)) - α * Tuu11 * ∂α1) + 0.5 * ((Tuu11 + 2 * β1 * Tuu11 + β1 ^ 2 * Tuu11) * (∂γ111 - 2 * (γ11 * hΓ111 + γ12 * hΓ211 + γ13 * hΓ311)) + (Tuu12 + 2 * β1 * Tuu12 + β2 * β1 * Tuu11) * (∂γ112 - 2 * (γ11 * hΓ112 + γ12 * hΓ212 + γ13 * hΓ312)) + (Tuu12 + 2 * β2 * Tuu11 + β2 * β1 * Tuu11) * (∂γ112 - 2 * (γ12 * hΓ111 + γ22 * hΓ211 + γ23 * hΓ311)) + (Tuu13 + 2 * β1 * Tuu13 + β3 * β1 * Tuu11) * (∂γ113 - 2 * (γ11 * hΓ113 + γ12 * hΓ213 + γ13 * hΓ313)) + (Tuu13 + 2 * β3 * Tuu11 + β3 * β1 * Tuu11) * (∂γ113 - 2 * (γ13 * hΓ111 + γ23 * hΓ211 + γ33 * hΓ311)) + (Tuu22 + 2 * β2 * Tuu12 + β2 ^ 2 * Tuu11) * (∂γ122 - 2 * (γ12 * hΓ112 + γ22 * hΓ212 + γ23 * hΓ312)) + (Tuu23 + 2 * β2 * Tuu13 + β3 * β2 * Tuu11) * (∂γ123 - 2 * (γ12 * hΓ113 + γ22 * hΓ213 + γ23 * hΓ313)) + (Tuu23 + 2 * β3 * Tuu12 + β3 * β2 * Tuu11) * (∂γ123 - 2 * (γ13 * hΓ112 + γ23 * hΓ212 + γ33 * hΓ312)) + (Tuu33 + 2 * β3 * Tuu13 + β3 ^ 2 * Tuu11) * (∂γ133 - 2 * (γ13 * hΓ113 + γ23 * hΓ213 + γ33 * hΓ313)))
        s_S2 = ((Tud11 * (∂β21 + β1 * hΓ112 + β2 * hΓ122 + β3 * hΓ123) + Tud12 * (∂β22 + β1 * hΓ212 + β2 * hΓ222 + β3 * hΓ223) + Tud13 * (∂β23 + β1 * hΓ312 + β2 * hΓ322 + β3 * hΓ323)) - α * Tuu11 * ∂α2) + 0.5 * ((Tuu11 + 2 * β1 * Tuu11 + β1 ^ 2 * Tuu11) * (∂γ211 - 2 * (γ11 * hΓ112 + γ12 * hΓ212 + γ13 * hΓ312)) + (Tuu12 + 2 * β1 * Tuu12 + β2 * β1 * Tuu11) * (∂γ212 - 2 * (γ11 * hΓ122 + γ12 * hΓ222 + γ13 * hΓ322)) + (Tuu12 + 2 * β2 * Tuu11 + β2 * β1 * Tuu11) * (∂γ212 - 2 * (γ12 * hΓ112 + γ22 * hΓ212 + γ23 * hΓ312)) + (Tuu13 + 2 * β1 * Tuu13 + β3 * β1 * Tuu11) * (∂γ213 - 2 * (γ11 * hΓ123 + γ12 * hΓ223 + γ13 * hΓ323)) + (Tuu13 + 2 * β3 * Tuu11 + β3 * β1 * Tuu11) * (∂γ213 - 2 * (γ13 * hΓ112 + γ23 * hΓ212 + γ33 * hΓ312)) + (Tuu22 + 2 * β2 * Tuu12 + β2 ^ 2 * Tuu11) * (∂γ222 - 2 * (γ12 * hΓ122 + γ22 * hΓ222 + γ23 * hΓ322)) + (Tuu23 + 2 * β2 * Tuu13 + β3 * β2 * Tuu11) * (∂γ223 - 2 * (γ12 * hΓ123 + γ22 * hΓ223 + γ23 * hΓ323)) + (Tuu23 + 2 * β3 * Tuu12 + β3 * β2 * Tuu11) * (∂γ223 - 2 * (γ13 * hΓ122 + γ23 * hΓ222 + γ33 * hΓ322)) + (Tuu33 + 2 * β3 * Tuu13 + β3 ^ 2 * Tuu11) * (∂γ233 - 2 * (γ13 * hΓ123 + γ23 * hΓ223 + γ33 * hΓ323)))
        s_S3 = ((Tud11 * (∂β31 + β1 * hΓ113 + β2 * hΓ123 + β3 * hΓ133) + Tud12 * (∂β32 + β1 * hΓ213 + β2 * hΓ223 + β3 * hΓ233) + Tud13 * (∂β33 + β1 * hΓ313 + β2 * hΓ323 + β3 * hΓ333)) - α * Tuu11 * ∂α3) + 0.5 * ((Tuu11 + 2 * β1 * Tuu11 + β1 ^ 2 * Tuu11) * (∂γ311 - 2 * (γ11 * hΓ113 + γ12 * hΓ213 + γ13 * hΓ313)) + (Tuu12 + 2 * β1 * Tuu12 + β2 * β1 * Tuu11) * (∂γ312 - 2 * (γ11 * hΓ123 + γ12 * hΓ223 + γ13 * hΓ323)) + (Tuu12 + 2 * β2 * Tuu11 + β2 * β1 * Tuu11) * (∂γ312 - 2 * (γ12 * hΓ113 + γ22 * hΓ213 + γ23 * hΓ313)) + (Tuu13 + 2 * β1 * Tuu13 + β3 * β1 * Tuu11) * (∂γ313 - 2 * (γ11 * hΓ133 + γ12 * hΓ233 + γ13 * hΓ333)) + (Tuu13 + 2 * β3 * Tuu11 + β3 * β1 * Tuu11) * (∂γ313 - 2 * (γ13 * hΓ113 + γ23 * hΓ213 + γ33 * hΓ313)) + (Tuu22 + 2 * β2 * Tuu12 + β2 ^ 2 * Tuu11) * (∂γ322 - 2 * (γ12 * hΓ123 + γ22 * hΓ223 + γ23 * hΓ323)) + (Tuu23 + 2 * β2 * Tuu13 + β3 * β2 * Tuu11) * (∂γ323 - 2 * (γ12 * hΓ133 + γ22 * hΓ233 + γ23 * hΓ333)) + (Tuu23 + 2 * β3 * Tuu12 + β3 * β2 * Tuu11) * (∂γ323 - 2 * (γ13 * hΓ123 + γ23 * hΓ223 + γ33 * hΓ323)) + (Tuu33 + 2 * β3 * Tuu13 + β3 ^ 2 * Tuu11) * (∂γ333 - 2 * (γ13 * hΓ133 + γ23 * hΓ233 + γ33 * hΓ333)))
        s_tau = K11 * Tuu11 + 2 * K12 * Tuu12 + 2 * K13 * Tuu13 + K22 * Tuu22 + 2 * K23 * Tuu23 + K33 * Tuu33 + Tuu11 * (-∂α1 + 2 * (K11 * β1 + K12 * β2 + K13 * β3)) + Tuu11 * ((K11 * β1 ^ 2 + K22 * β2 ^ 2 + K33 * β3 ^ 2 + 2 * K12 * β2 * β1 + 2 * K13 * β3 * β1 + 2 * K23 * β3 * β2) - (β1 * ∂α1 + β2 * ∂α2 + β3 * ∂α3)) + Tuu12 * (-∂α2 + 2 * (K12 * β1 + K22 * β2 + K23 * β3)) + Tuu13 * (-∂α3 + 2 * (K13 * β1 + K23 * β2 + K33 * β3))
        CACHE.s_S1[IDX] = s_S1
        CACHE.s_S2[IDX] = s_S2
        CACHE.s_S3[IDX] = s_S3
        CACHE.s_tau[IDX] = s_tau
    end
end
