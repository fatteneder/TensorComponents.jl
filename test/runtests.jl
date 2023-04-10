using Test

using TensorComponents
using TensorOperations
using SymEngine

const TC = TensorComponents
const TO = TensorOperations


@testset "@index tests" begin

    ### valid use

    @test (TC.components(quote @index a = 2 end); true)
    @test (TC.components(quote @index a, b = 2 end); true)
    @test (TC.components(quote @index a, b, i123 = 2 end); true)
    @test (TC.components(quote @index asdf = 1:123 end); true)
    @test (TC.components(quote @index wasd, xyz = 3:4 end); true)
    # multiple index definitions
    @test (TC.components(quote @index a = 2; @index b = 2 end); true)
    @test (TC.components(quote @index a = 2; @index b = 1:3 end); true)
    @test (TC.components(quote @index a, c = 2; @index b = 1:2 end); true)
    @test (TC.components(quote @index a, c = 2; @index b = 3 end); true)
    @test (TC.components(quote @index a, c = 3:8; @index b, d = 123 end); true)
    @test (TC.components(quote @index a, c = 2; @index b, d = 7:11; @index e, f, g = 1:5 end); true)


    ### invalid use

    # no indices
    @test_throws ErrorException TC.components(quote @index end)
    # invalid index name
    @test_throws ErrorException TC.components(quote @index 1 = 2 end)
    @test_throws ErrorException TC.components(quote @index 1k = 2 end)
    @test_throws ErrorException TC.components(quote @index a, 123abc = 2 end)
    @test_throws ErrorException TC.components(quote @index 1:4 end)
    # no tuple syntax
    @test_throws ErrorException TC.components(quote @index a b c = 123 end)
    # duplicated index names
    @test_throws ErrorException TC.components(quote @index a, a = 123 end)
    @test_throws ErrorException TC.components(quote @index a, b = 123; @index b = 1 end)
    @test_throws ErrorException TC.components(quote @index b = 123; @index b = 1:4 end)
    @test_throws ErrorException TC.components(quote @index b = 123; @index b = 123 end)
    # invalid index range
    @test_throws ErrorException TC.components(quote @index a = 0 end)
    @test_throws ErrorException TC.components(quote @index a = -1 end)
    @test_throws ErrorException TC.components(quote @index a, b, c = 12.5 end)
    @test_throws ErrorException TC.components(quote @index a, d, e, f = -123 end)
    @test_throws ErrorException TC.components(quote @index a = 1:-5 end)
    @test_throws ErrorException TC.components(quote @index a = 0:4 end)
    @test_throws ErrorException TC.components(quote @index a, b, c = -1:-12 end)
    @test_throws ErrorException TC.components(quote @index a, d, e, f = -1:2 end)

end


@testset "@components equations test" begin

    ### valid use

    # one equation
    @test (eval(TC.components(quote
        @index i, j = 4
        A[i,j] = B[i,j]
    end)); true)
    # multiple equations
    @test (eval(TC.components(quote
        @index i, j, k = 1:3
        A[i,j] = B[i,j,k] * C[k]
        D[i]   = A[i,j] * C[j]
        E[i]   = B[i,k,k]
    end)); true)
    # multiple equations with mixed indices
    @test (eval(TC.components(quote
        @index i, j, k = 1:3
        @index a, b, c = 3:5
        A[i,j] = B[i,j,a] * C[a]
        D[i]   = A[i,b] * C[b]
        E[i,a] = B[i,a,b] * C[b]
    end)); true)

    # TODO Why is here a stack overflow?
    # we can handle name clashes between tensors and indices
    # @test (eval(TC.components(quote
    #     @index I, J = 1:3
    #     # I[I,J] = J[I,J]
    # end)); true)
    # TODO How to deal with the coefficients? Those need to be Basics ...
    # @test (eval(TC.components(quote
    #     @index i, j = 4
    #     A[i,j] = α * A[i,j]
    # end)); true)


    ### invalid use

    # undefined indices
    @test_throws ErrorException TC.components(quote
        @index i, j = 4
        A[a,b] = B[b,a]
    end)
    @test_throws ErrorException TC.components(quote
        @index i, j = 4
        A[j,i] = B[a,j,i]
    end)
    @test_throws ErrorException TC.components(quote
        @index i, j = 4
        A[a,i] = B[a,i]
    end)
    # undefined indices with multiple equations
    @test_throws ErrorException TC.components(quote
        @index i, j, k = 4
        A[i,j] = B[i,j] * C[k,k]
        D[a]   = B[i,j]
    end)
    # invalid contractions (errors coming from TensorOperations.@tensor
    # so we need to use @eval to force evaluation of @tensor)
    @test_throws TO.IndexError eval(TC.components(quote
        @index i, j, k = 4
        A[i,j] = B[i,j,j]
    end))
    @test_throws TO.IndexError eval(TC.components(quote
        @index i, j = 1:4
        A[i,j] = B[i,j,j]
    end))
    @test_throws TO.IndexError eval(TC.components(quote
        @index i, j, k = 3
        A[i,j] = B[i,j] + C[k]
    end))

end


@testset "utils" begin

    # topological sort
    nodes  = [ 10, 8, 5, 3, 2, 11, 9, 7 ]
    childs = Vector{Int}[ [], [9,10], [11], [8], [], [2,9], [], [11,8] ]
    seq = TC.topological_sort(nodes, childs)
    sorted_nodes  = nodes[seq]
    sorted_childs = childs[seq]
    N = length(nodes)
    # test that when walking the sorted nodes list, all childs of a node
    # occur after the node
    @test all(1:N) do idx
        all(c -> c in view(sorted_nodes, idx:N), sorted_childs[idx])
    end

    # getscalars
    @test TC.getscalars(:(a)) == Any[:a]
    @test TC.getscalars(:(a * A[i,j])) == Any[:(a * true)]
    @test TC.getscalars(:(2 * a * A[i,j])) == Any[:(2 * a * true)]
    @test TC.getscalars(:(a * 2 * A[i,j])) == Any[:(a * 2 * true)]
    @test TC.getscalars(:(x * A[i,j] + B[j,i] * y)) == Any[:(x * true), :(true * y)]
    @test TC.getscalars(:((x^2 + y)^2 * A[i,j] + B[j,i] * y)) == Any[:((x^2 + y)^2 * true), :(true * y)]
    @test TC.getscalars(:(cos(α) * A[i,j] + B[j,i] * sin(α))) == Any[:(cos(α) * true), :(true * sin(α))]

    # getvariables
    @test TC.getvariables.(TC.getscalars(:(a))) == [ [:a] ]
    @test TC.getvariables.(TC.getscalars(:(a * A[i,j]))) == [ [:a] ]
    @test TC.getvariables.(TC.getscalars(:(2 * a * A[i,j]))) == [ [:a] ]
    @test TC.getvariables.(TC.getscalars(:(a * 2 * A[i,j]))) == [ [:a] ]
    @test TC.getvariables.(TC.getscalars(:(x * A[i,j] + B[j,i] * y))) == [ [:x], [:y] ]
    @test TC.getvariables.(TC.getscalars(:((x^2 + y)^2 * A[i,j] + B[j,i] * y))) == [ [:x, :y], [:y] ]
    @test TC.getvariables.(TC.getscalars(:(cos(α) * A[i,j] + B[j,i] * sin(α)))) == [ [:α], [:α] ]

end
