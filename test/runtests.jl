using Test

using TensorComponents
using TensorOperations
using SymEngine

const TC = TensorComponents
const TO = TensorOperations


@testset "@components equations" begin

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
    # Shuffle in some scalar variables, also with functions
    @test (eval(TC.components(quote
        @index i, j = 4
        A[i,j] = a * B[i,j]
        C[i,j] = cos(α) * D[i,j] + sin(β) * E[i,j]
    end)); true)
    # can also use purely scalar equations
    @test (eval(@components begin
        a = b + 1
    end); true)
    # can also use purely scalar equations
    @test (eval(@components begin
        @index i = 4
        a = A[i,i]
    end); true)
    # Impose symmetries on tensors to reduce number of operations
    let
        comps = eval(@components begin
            @index i, j = 4
            @symmetry A[i,j] = A[j,i]
            A[i,j] = a[i] * a[j]
        end)
        A = TC.SymbolicTensor(:A,4,4)
        a = TC.SymbolicTensor(:a,4)
        aa = a .* a'
        test_comps = [ (A[1,1],aa[1,1]), (A[1,2],aa[1,2]), (A[1,3],aa[1,3]), (A[1,4],aa[1,4]),
                       (A[2,2],aa[2,2]), (A[2,3],aa[2,3]), (A[2,4],aa[2,4]),
                       (A[3,3],aa[3,3]), (A[3,4],aa[3,4]),
                       (A[4,4],aa[4,4]) ]
        @test all(comps .== test_comps)
    end

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

    # name clashes
    # you can't use indices as scalars or tensors
    @test_throws ErrorException TC.components(quote
        @index I, J = 1:3
        A[I,J] = I * B[I,J]
    end)
    @test_throws ErrorException TC.components(quote
        @index I, J = 1:3
        A[I,J] = I[I,J]
    end)
    # also fails if index is unused
    @test_throws ErrorException TC.components(quote
        @index I, J, A = 1:3
        A[I,J] = B[I,J]
    end)
    # you can't use a tensor as a scalar
    @test_throws ErrorException TC.components(quote
        @index I, J = 1:3
        A[I,J] = A * B[I,J]
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
    # TODO Fix this: can't assign to lhs with contractions
    # @test_throws ErrorException eval(TC.components(quote
    @test_broken eval(TC.components(quote
        @index i = 3
        A[i,i] = a
    end))

end


@testset "@index" begin

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


@testset "@symmetry" begin

    ### invalid use

    # no indices
    @test_throws ErrorException TC.components(quote @symmetry end)
    # no assignment
    @test_throws ErrorException TC.components(quote @symmetry A[i,j] end)
    @test_throws ErrorException TC.components(quote @symmetry A[i,j] * B[i,j] end)
    # more than one tensor
    @test_throws ErrorException TC.components(quote @symmetry A[i,j] = B[i,j] end)
    # no indices
    @test_throws ErrorException TC.components(quote @symmetry A = B end)
    # no indices on one side
    # TODO Fix this
    @test_throws ErrorException TC.components(quote @symmetry A[i,i] = B end)
    @test_throws ErrorException TC.components(quote @symmetry A = B[i,i] end)
    # inconsistent index pattern
    @test_throws ErrorException TC.components(quote @symmetry A[i,j,k] = B[k] end)
    # TODO Fix this
    @test_throws ErrorException TC.components(quote @symmetry A[i,j] = B[i,j,k,k] end)

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

    # getcoeffs
    @test TC.getcoeffs(:(a)) == Any[:a]
    @test TC.getcoeffs(:(a * A[i,j])) == Any[:(a * true)]
    @test TC.getcoeffs(:(2 * a * A[i,j])) == Any[:(2 * a * true)]
    @test TC.getcoeffs(:(a * 2 * A[i,j])) == Any[:(a * 2 * true)]
    @test TC.getcoeffs(:(x * A[i,j] + B[j,i] * y)) == Any[:(x * true), :(true * y)]
    @test TC.getcoeffs(:((x^2 + y)^2 * A[i,j] + B[j,i] * y)) == Any[:((x^2 + y)^2 * true), :(true * y)]
    @test TC.getcoeffs(:(cos(α) * A[i,j] + B[j,i] * sin(α))) == Any[:(cos(α) * true), :(true * sin(α))]

    # getscalars
    @test TC.getscalars.(TC.getcoeffs(:(a))) == [ [:a] ]
    @test TC.getscalars.(TC.getcoeffs(:(a * A[i,j]))) == [ [:a] ]
    @test TC.getscalars.(TC.getcoeffs(:(2 * a * A[i,j]))) == [ [:a] ]
    @test TC.getscalars.(TC.getcoeffs(:(a * 2 * A[i,j]))) == [ [:a] ]
    @test TC.getscalars.(TC.getcoeffs(:(x * A[i,j] + B[j,i] * y))) == [ [:x], [:y] ]
    @test TC.getscalars.(TC.getcoeffs(:((x^2 + y)^2 * A[i,j] + B[j,i] * y))) == [ [:x, :y], [:y] ]
    @test TC.getscalars.(TC.getcoeffs(:(cos(α) * A[i,j] + B[j,i] * sin(α)))) == [ [:α], [:α] ]

end


function reduce_components_riemann_tensor(Riem)
    bianchi = permutedims(Riem,(1,2,3,4)) + permutedims(Riem,(1,3,4,2)) + permutedims(Riem,(1,4,2,3))
    Riem    = TC.resolve_dependents(Riem, bianchi)
    asym1   = permutedims(Riem,(1,2,3,4)) + permutedims(Riem,(2,1,3,4))
    Riem    = TC.resolve_dependents(Riem, asym1)
    asym2   = permutedims(Riem,(1,2,3,4)) + permutedims(Riem,(1,2,4,3))
    Riem    = TC.resolve_dependents(Riem, asym2)
    return Riem
end


@testset "SymbolicTensor" begin

    # compute independent components of Riemann tensor in 2-5 dimensions
    # only testing against number of independents found
    for N = 2:5
        Riem = TC.SymbolicTensor(:R,N,N,N,N)
        red_Riem = reduce_components_riemann_tensor(Riem)
        ideps = [ SymEngine.free_symbols(R) for R in red_Riem[:] ]
        uideps = unique!(reduce(vcat, ideps))
        # https://physics.stackexchange.com/a/506095
        n_ideps = Int(N^2*(N^2-1)/12)
        @test length(uideps) == n_ideps
    end

    # verify explicitly that reducing components succeeded with a symmetry tensor
    A = TC.SymbolicTensor(:A,4,4)
    AT = transpose(A)
    sym_eq = A .- AT
    A = TC.resolve_dependents(A, sym_eq)
    AT = transpose(A)
    @test all( (A .- AT .== zeros(Basic,4,4))[:] )

end
