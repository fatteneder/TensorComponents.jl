@testset "@components examples" begin

    @testset "GRHD source terms" begin
        expected = :( (s_tau,
                       K11*Tuu11 + 2*K12*Tuu12 + 2*K13*Tuu13 + K22*Tuu22 + 2*K23*Tuu23 + K33*Tuu33 +
                       Tuu11*(-∂α1 + 2*(K11*β1 + K12*β2 + K13*β3)) + Tuu11*(K11*β1^2 + K22*β2^2 +
                       K33*β3^2 + 2*K12*β2*β1 + 2*K13*β3*β1 + 2*K23*β3*β2 - (β1*∂α1 + β2*∂α2 +
                       β3*∂α3)) + Tuu12*(-∂α2 + 2*(K12*β1 + K22*β2 + K23*β3)) +
                       Tuu13*(-∂α3 + 2*(K13*β1 + K23*β2 + K33*β3))) )
        fulloutput = @components begin
            @index i, j, k, l   = 3
            @symmetry hΓ[k,i,j] = hΓ[k,j,i]
            @symmetry K[i,j]    = K[j,i]
            @symmetry Tuu[i,j]  = Tuu[j,i]
            @symmetry g[i,j]    = g[j,i]
            Tud[i,k] = Tuu[i,j] * g[j,k]
            # grhd source term for tau equation, cf. arXiv:1104.4751v3, eq 26
            s_tau  = Tuu[1,1] * (β[i]*β[j]*K[i,j] - β[i]*∂α[i]) + Tuu[1,i] * (2*β[j]*K[i,j] - ∂α[i] ) + Tuu[i,j]*K[i,j]
        end
        # only compare final result
        output = fulloutput[end] |> string |> Meta.parse
        @test output == expected
    end

end


@testset "@components ref test dummy" begin

    @testset "GRHD source terms" begin
        eqs_ref = TC.parse_eqs(joinpath(@__DIR__,"dummyeq.ref.txt"))
        deps_ref, ideps_ref = TensorComponents.determine_dependents_independents(eqs_ref)
        code_ref = TC.generate_code("dummyeq_ref", eqs_ref, deps_ref)
        fname = tempname()
        open(fname, "w") do io
            println(io, code_ref)
        end
        fn_ref = include(fname)

        eqs = @components begin
            @index i, j = 3
            A2 = A[i,j] * A[i,j]
        end
        # only compute the dependents that are provided from the reference
        code = TC.generate_code("dummyeq", eqs, deps_ref)
        fname = tempname()
        open(fname, "w" ) do io
            println(io, code)
        end
        fn = include(fname)

        # setup test code and caches
        Random.seed!(123)
        Nsmpl = 100
        cache_ref = NamedTuple( (var,abs.(randn(Nsmpl))) for var in vcat(deps_ref,ideps_ref) )
        cache = deepcopy(cache_ref)
        # run tests
        fn_ref(cache_ref, Nsmpl)
        fn(cache, Nsmpl)
        # compare dependents
        for d in deps_ref
            v, v_ref = cache[d], cache_ref[d]
            @test v ≈ v_ref
        end
    end

end


@testset "@components ref test" begin

    @testset "GRHD source terms" begin
        @info "ref test: GRHD source terms"
        eqs_ref = TC.parse_eqs(joinpath(@__DIR__,"grhd_sources.ref.txt"))
        display(eqs_ref)
        deps_ref, ideps_ref = TensorComponents.determine_dependents_independents(eqs_ref)
        code_ref = TC.generate_code("grhd_sources_ref", eqs_ref, deps_ref)
        fname = tempname()
        open(fname, "w") do io
            println(io, code_ref)
        end
        @info "ref func: $fname"
        fn_ref = include(fname)

        eqs = @components begin
            @index i, j, k, l   = 3
            # grhd source term for tau equation, cf. arXiv:1104.4751v3, eq 26
            s_S[k] = Tuu[1,1] * (1/2 * β[i] * β[j] * ∂γ[k,i,j] - α * ∂α[k]) + Tuu[1,i] * β[j] * ∂γ[k,i,j] + Tud[1,i] * ∂β[k,i] + 1/2 * Tuu[i,j] * ∂γ[k,i,j]
            s_tau  = Tuu[1,1] * (β[i]*β[j]*K[i,j] - β[i]*∂α[i]) + Tuu[1,i] * (2*β[j]*K[i,j] - ∂α[i] ) + Tuu[i,j]*K[i,j]
        end
        # only compute the dependents that are provided from the reference
        code = TC.generate_code("grhd_sources", eqs, deps_ref)
        fname = tempname()
        open(fname, "w" ) do io
            println(io, code)
        end
        @info "func: $fname"
        fn = include(fname)

        # setup test code and caches
        Random.seed!(123)
        Nsmpl = 100
        cache_ref = NamedTuple( (var,abs.(randn(Nsmpl))) for var in vcat(deps_ref,ideps_ref) )
        cache = deepcopy(cache_ref)
        sum_cache_ref = mapreduce(sum, +, cache_ref)
        sum_cache     = mapreduce(sum, +, cache)
        @show sum_cache_ref
        @show sum_cache
        # run tests
        fn_ref(cache_ref, Nsmpl)
        sum_cache_ref = mapreduce(sum, +, cache_ref)
        sum_cache     = mapreduce(sum, +, cache)
        @show sum_cache_ref
        @show sum_cache
        fn(cache, Nsmpl)
        sum_cache_ref = mapreduce(sum, +, cache_ref)
        sum_cache     = mapreduce(sum, +, cache)
        @show sum_cache_ref
        @show sum_cache
        # compare dependents
        for d in deps_ref
            v, v_ref = cache[d], cache_ref[d]
            @test v ≈ v_ref
        end
    end

end
