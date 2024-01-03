using Test

using LinearAlgebra
using TensorComponents
using SymEngine
using Random
Random.seed!(1234)

const TC = TensorComponents


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

    @test TC.isevenperm([1,2,3]) == true
    @test TC.isevenperm([2,1,3]) == false
    @test TC.isevenperm([2,3,1]) == true
    @test TC.isoddperm([1,2,3]) == false
    @test TC.isoddperm([2,1,3]) == true
    @test TC.isoddperm([2,3,1]) == false

    @test Set(TC.perms([1,2,3])) == Set([ [1,2,3], [2,1,3], [1,3,2], [3,2,1], [2,3,1], [3,1,2] ])

    ϵ = TC.epsilon_symbol(3)
    @test ϵ[1,2,3] == ϵ[2,3,1] == ϵ[3,1,2] == 1
    @test ϵ[2,1,3] == ϵ[1,3,2] == ϵ[3,2,1] == -1
    @test ϵ[1,1,1] == ϵ[1,1,2] == ϵ[1,1,3] == ϵ[2,2,1] == ϵ[2,2,2] == ϵ[2,2,3] ==
          ϵ[3,3,1] == ϵ[3,3,2] == ϵ[3,3,3] == 0

    δ = TC.delta_symbol(3)
    @test δ == diagm(ones(3))

    @test TC.count_operations(:(a + b)) == Dict(:+ => 1)
    @test TC.count_operations(:(a + 2b + c)) == Dict(:+ => 2, :* => 1)
    @test TC.count_operations(:(a + b * c)) == Dict(:+ => 1, :* => 1)
    @test TC.count_operations(:(a + 2b)) == Dict(:+ => 1, :* => 1)
    @test TC.count_operations(:((a + b)^2 * c)) == Dict(:+ => 1, :* => 1, :^ => 1)
    @test TC.count_operations(:(tan(a + b)^2 / (4^3 - 1))) == Dict(:+ => 1, :- => 1,
                                                                   :^ => 2, :/ => 1,
                                                                   :tan => 1)

    exs = [ :(a + b), :(b + c), :(a * b - c), :(d^-3 * tan(a/b) / f) ]
    ops = TC.count_operations.(exs)
    expected = Dict(:+ => 1+1, :* => 1+1, :- => 1, :^ => 1, :tan => 1, :/ => 2)
    got = TC.mergecounts(ops...)
    @test expected == got

end


@testset "contraction utilities" begin

    @test TC.hasindices(:(a)) == false
    @test TC.hasindices(:(a / b)) == false
    @test TC.hasindices(:(a \ b)) == false
    @test TC.hasindices(:(a \ b + 1)) == false
    @test TC.hasindices(:(a \ 2 + c)) == false
    @test TC.hasindices(:((a \ 2)^c)) == false
    @test TC.hasindices(:(A[i,j])) == true
    @test TC.hasindices(:(a * A[i,j])) == true
    @test TC.hasindices(:(a \ A[i,j])) == true
    @test TC.hasindices(:(a / A[i,j])) == true
    @test TC.hasindices(:(a / A[i,j] + B[i,j])) == true

    @test TC.isscalarexpr(:(a)) == true
    @test TC.isscalarexpr(:(a / b)) == true
    @test TC.isscalarexpr(:(a \ b)) == false
    @test TC.isscalarexpr(:(a \ b + 1)) == false
    @test TC.isscalarexpr(:(a \ 2 + c)) == false
    @test TC.isscalarexpr(:((a \ 2)^c)) == false
    @test TC.isscalarexpr(:(A[i,j])) == false
    @test TC.isscalarexpr(:(a * A[i,j])) == false
    @test TC.isscalarexpr(:(a * A[i,j] / b)) == false
    @test TC.isscalarexpr(:(a \ A[i,j])) == false
    @test TC.isscalarexpr(:(a / A[i,j])) == false
    @test TC.isscalarexpr(:(a / A[i,j] + B[i,j])) == false
    @test TC.isscalarexpr(:(A[i] * A[i])) == true
    @test TC.isscalarexpr(:(a + A[i] * A[i])) == true
    @test TC.isscalarexpr(:(A[0])) == true
    @test TC.isscalarexpr(:(A[i,0,i])) == true
    @test TC.isscalarexpr(:(A[i,j,0])) == false
    @test TC.isscalarexpr(:(A[0,j,0])) == false

    @test TC.istensor(:(a)) == false
    @test TC.istensor(:(A[i,j])) == true
    @test TC.istensor(:(a * A[i,j])) == false
    @test TC.istensor(:(a / A[i,j])) == false
    @test TC.istensor(:(A[i,j] / b)) == false
    @test TC.istensor(:(a / b * A[i,j])) == false
    @test TC.istensor(:(A[i,j] * B[k])) == false
    @test TC.istensor(:(A[i,j] + B[k])) == false
    @test TC.istensor(:(A[i,j] + B[i,j])) == false
    @test TC.istensor(:(b \ C[i,j])) == false
    @test TC.istensor(:(b \ C[i,j] + B[k])) == false
    @test TC.istensor(:(C[i,j] \ b + B[k])) == false
    @test TC.istensor(:(A[i,j] * (b * B[k] - c * C[k]) / d)) == false

    @test TC.isgeneraltensor(:(a)) == false
    @test TC.isgeneraltensor(:(A[i,j])) == true
    @test TC.isgeneraltensor(:(a * A[i,j])) == true
    @test TC.isgeneraltensor(:(a / A[i,j])) == false
    @test TC.isgeneraltensor(:(A[i,j] / b)) == true
    @test TC.isgeneraltensor(:(a * A[i,j] / b)) == true
    @test TC.isgeneraltensor(:((a * A[i,j] / b) * c)) == true
    @test TC.isgeneraltensor(:(a \ A[i,j])) == false
    @test TC.isgeneraltensor(:(A[i,j] \ b)) == false
    @test TC.isgeneraltensor(:(a / b * A[i,j])) == true
    @test TC.isgeneraltensor(:(A[i,j] * B[k])) == false
    @test TC.isgeneraltensor(:(A[i,j] + B[k])) == false
    @test TC.isgeneraltensor(:(A[i,j] + B[i,j])) == false
    @test TC.isgeneraltensor(:(b \ C[i,j])) == false
    @test TC.isgeneraltensor(:(b \ C[i,j] + B[k])) == false
    @test TC.isgeneraltensor(:(C[i,j] \ b + B[k])) == false
    @test TC.isgeneraltensor(:(b \ C[i,j] * B[k])) == false
    @test TC.isgeneraltensor(:(C[i,j] \ b * B[k])) == false
    @test TC.isgeneraltensor(:(A[i,j] * (b * B[k] - c * C[k]) / d)) == false

    @test TC.istensorexpr(:(a)) == false
    @test TC.istensorexpr(:(A[i,j])) == true
    @test TC.istensorexpr(:(A[i,j] + a)) == false
    @test TC.istensorexpr(:(a * A[i,j])) == true
    @test TC.istensorexpr(:(a / A[i,j])) == false
    @test TC.istensorexpr(:(A[i,j] / b)) == true
    @test TC.istensorexpr(:(C[i,j] * D[j] / x)) == true
    @test TC.istensorexpr(:(a / b * A[i,j])) == true
    @test TC.istensorexpr(:(A[i,j] * B[k])) == true
    @test TC.istensorexpr(:(A[i,j] + B[k])) == true
    @test TC.istensorexpr(:(A[i,j] + B[i,j])) == true
    @test TC.istensorexpr(:(β / D[i, j] * C[k,l])) == false
    @test TC.istensorexpr(:(α * A[i, j] * B[k, l] + (β / C[k, l]) * D[i, j])) == false
    @test TC.istensorexpr(:((a + b)^c * A[i,j] * B[j] - C[i,j] * D[j] / x * y^2)) == true
    @test TC.istensorexpr(:(A[i,j] * (b * B[k] - c * C[k]) / d)) == true
    @test TC.istensorexpr(:((D[k] * D[k] + b) * C[i,j])) == true
    @test TC.istensorexpr(:(T[a,b] * (dg[k,a,b] - Γ[d,a,b] * g[d,k]))) == true

    @test TC.istensorprod(:(a)) == false
    @test TC.istensorprod(:(A[i])) == true
    @test TC.istensorprod(:(a * A[i])) == true
    @test TC.istensorprod(:(A[i] * B[i])) == true
    @test TC.istensorprod(:(A[i] * B[j])) == true
    @test TC.istensorprod(:(A[i] * B[j] / d)) == true
    @test TC.istensorprod(:(A[i] / d * B[j])) == true
    @test TC.istensorprod(:(A[i] * B[j] + C[k])) == false
    @test TC.istensorprod(:(A[i] * (B[j] + C[k]))) == false
    @test TC.istensorprod(:(A[i] * (B[j] + C[j]))) == true
    @test TC.istensorprod(:(A[i] * (B[j] + C[k]) / b)) == false
    @test TC.istensorprod(:(A[i] * (B[j] + C[j]) / b)) == true
    @test TC.istensorprod(:(A[i] * (2 * B[j] + C[k]) / b)) == false
    @test TC.istensorprod(:(A[i] * (2 * B[j] + C[j]) / b)) == true

    @test TC.iscontraction(:(a)) == false
    @test TC.iscontraction(:(A[i,j])) == false
    @test TC.iscontraction(:(A[i,j] + a)) == false
    @test TC.iscontraction(:(a * A[i,j])) == false
    @test TC.iscontraction(:(a / A[i,j])) == false
    @test TC.iscontraction(:(A[i,j] / b)) == false
    @test TC.iscontraction(:(C[i,j] * D[j] / x)) == true
    @test TC.iscontraction(:(a / b * A[i,j])) == false
    @test TC.iscontraction(:(A[i,j] * B[k])) == false
    @test TC.iscontraction(:(A[i,j] + B[k])) == false
    @test TC.iscontraction(:(A[i,j] + B[i,j])) == false
    @test TC.iscontraction(:(β / D[i, j] * C[k,l])) == false
    @test TC.iscontraction(:(α * A[i, j] * B[k, l] + (β / C[k, l]) * D[i, j])) == false
    @test TC.iscontraction(:((a + b)^c * A[i,j] * B[j] - C[i,j] * D[j] / x * y^2)) == false
    @test TC.iscontraction(:(A[i,j] * B[j,i])) == true
    @test TC.iscontraction(:(A[i,i])) == true
    @test TC.iscontraction(:((a+b)^c * A[i,i])) == true
    @test TC.iscontraction(:(A[i] * (2 * B[i] - C[i]))) == true
    @test TC.iscontraction(:(A[i,j] * B[j])) == true
    @test TC.iscontraction(:(A[i,j] * B[j] + C[i])) == false

    @test TC.getallindices(:(A[i,j])) == [:i,:j]
    @test TC.getallindices(:(A[i,j] * B[k,l])) == [:i,:j,:k,:l]
    @test TC.getallindices(:(A[i,j] * B[k,l] * C[k,l])) == [:i,:j,:k,:l]
    @test TC.getallindices(:(A[i,j] * B[k,l] + C[k,l] * D[i,j])) == [:i,:j,:k,:l]
    @test TC.getallindices(:(A[i,j] * B[k,l] + C[m,n] * D[o,p])) == [:i,:j,:k,:l,:m,:n,:o,:p]
    @test TC.getallindices(:(a)) == []
    @test TC.getallindices(:(a + b)) == []
    @test TC.getallindices(:((a + b)^c)) == []
    @test TC.getallindices(:((a + b)^c * A[i,j])) == [:i,:j]
    @test TC.getallindices(:(A[i,j] * B[j] - C[i,j] * D[j])) == [:i,:j]
    @test TC.getallindices(:((a + b)^c * A[i,j] * B[x] - C[y,z] * D[j] / x * y^2)) == [:i,:j,:x,:y,:z]
    @test TC.getallindices(:(A[i,j] * (b * B[k] - c * C[k]) / d)) == [:i,:j,:k]
    @test TC.getallindices(:((D[k] * D[k] + b) * C[i,j])) |> sort == [:i,:j,:k]

    # TODO This should fail with comprehensive msg
    @test_skip TC.getallindices(:(C[k]^3)) == [:k]
    # should be written as (really, looks unnecessariyl complicated...)
    @test_skip TC.getallindices(:(C[i]*C[j]*C[k]*δ[i,j]*δ[j,k]*δ[k,i])) == [:i,:j,:k]

    @test TC.getindices(:(A[i,j])) == [:i,:j]
    @test TC.getindices(:(A[i,j] * B[k,l])) == [:i,:j,:k,:l]
    @test TC.getindices(:(A[i,j] * B[k,l] * C[k,l])) == [:i,:j]
    @test TC.getindices(:(A[i,j] * B[k,l] + C[k,l] * D[i,j])) == [:i,:j,:k,:l]
    @test TC.getindices(:(A[i,j] * B[k,l] + C[m,n] * D[o,p])) == [:i,:j,:k,:l,:m,:n,:o,:p]
    @test TC.getindices(:(a)) == []
    @test TC.getindices(:(a + b)) == []
    @test TC.getindices(:((a + b)^c)) == []
    @test TC.getindices(:((a + b)^c * A[i,j])) == [:i,:j]
    @test TC.getindices(:(A[i,j] * B[j] - C[i,j] * D[j])) == [:i]
    @test TC.getindices(:((a + b)^c * A[i,j] * B[x] - C[y,z] * D[j] / x * y^2)) == [:i,:j,:x,:y,:z]
    @test TC.getindices(:(A[i,j] * (b * B[k] - c * C[k]) / d)) == [:i,:j,:k]
    @test TC.getindices(:(A[i,j] * A[i,j])) == []
    @test TC.getindices(:(A[i,k] * A[i,l])) == [:k,:l]
    @test TC.getindices(:(A[i,k] * A[i,l] + B[l,k])) == [:k,:l]
    @test TC.getindices(:(A[i,i])) == []

    @test TC.getcontractedindices(:(A[i,j])) == []
    @test TC.getcontractedindices(:(A[i,j] * B[k,l])) == []
    @test TC.getcontractedindices(:(A[i,j] * B[k,l] * C[k,l])) == [:k,:l]
    @test TC.getcontractedindices(:(A[i,j] * B[k,l] + C[k,l] * D[i,j])) == []
    @test TC.getcontractedindices(:(A[i,j] * B[k,l] + C[m,n] * D[o,p])) == []
    @test TC.getcontractedindices(:(a)) == []
    @test TC.getcontractedindices(:(a + b)) == []
    @test TC.getcontractedindices(:((a + b)^c)) == []
    @test TC.getcontractedindices(:((a + b)^c * A[i,j])) == []
    @test TC.getcontractedindices(:(A[i,j] * B[j] - C[i,j] * D[j])) == [:j]
    @test TC.getcontractedindices(:((a + b)^c * A[i,j] * B[x] - C[y,z] * D[j] / x * y^2)) == []
    @test TC.getcontractedindices(:(A[i,j] * (b * B[k] - c * C[k]) / d)) == []
    @test TC.getcontractedindices(:(A[i,j] * A[i,j])) == [:i,:j]
    @test TC.getcontractedindices(:(A[i,k] * A[i,l])) == [:i]
    @test TC.getcontractedindices(:(A[i,k] * A[i,l] + B[l,k])) == [:i]
    @test TC.getcontractedindices(:(A[i,i])) == [:i]


    @test TC.gettensors(:(A[i,j])) == [ :(A[i,j]) ]
    @test TC.gettensors(:(A[i,j] * B[k,l])) == [ :(A[i,j]), :(B[k,l]) ]
    @test TC.gettensors(:(A[i,j] * B[k,l] * C[k,l])) == [ :(A[i,j]), :(B[k,l]), :(C[k,l]) ]
    @test TC.gettensors(:(A[i,j] * B[k,l] + C[k,l] * D[i,j])) == [ :(A[i,j]), :(B[k,l]), :(C[k,l]), :(D[i,j]) ]
    @test TC.gettensors(:(A[i,j] * B[k,l] + C[m,n] * D[o,p])) == [ :(A[i,j]), :(B[k,l]), :(C[m,n]), :(D[o,p]) ]
    @test TC.gettensors(:(a)) == []
    @test TC.gettensors(:(a + b)) == []
    @test TC.gettensors(:((a + b)^c)) == []
    @test TC.gettensors(:((a + b)^c * A[i,j])) == [ :(A[i,j]) ]
    @test TC.gettensors(:(A[i,j] * B[j] - C[i,j] * D[j])) == [ :(A[i,j]), :(B[j]), :(C[i,j]), :(D[j]) ]
    @test TC.gettensors(:((a + b)^c * A[i,j] * B[x] - C[y,z] * D[j] / x * y^2)) == [ :(A[i,j]), :(B[x]), :(C[y,z]), :(D[j]) ]
    @test TC.gettensors(:(A[i,j] * (b * B[k] - c * C[k]) / d)) == [ :(A[i,j]), :(B[k]), :(C[k]) ]

    @test TC.gettensorheads(:(A[i,j])) == [ :A ]
    @test TC.gettensorheads(:(A[i,j] * B[k,l])) == [ :A, :B ]
    @test TC.gettensorheads(:(A[i,j] * B[k,l] * C[k,l])) == [ :A, :B, :C ]
    @test TC.gettensorheads(:(A[i,j] * B[k,l] + C[k,l] * D[i,j])) == [ :A, :B, :C, :D ]
    @test TC.gettensorheads(:(A[i,j] * B[k,l] + C[m,n] * D[o,p])) == [ :A, :B, :C, :D ]
    @test TC.gettensorheads(:(a)) == []
    @test TC.gettensorheads(:(a + b)) == []
    @test TC.gettensorheads(:((a + b)^c)) == []
    @test TC.gettensorheads(:((a + b)^c * A[i,j])) == [ :A ]
    @test TC.gettensorheads(:(A[i,j] * B[j] - C[i,j] * D[j])) == [ :A, :B, :C, :D ]
    @test TC.gettensorheads(:((a + b)^c * A[i,j] * B[x] - C[y,z] * D[j] / x * y^2)) == [ :A, :B, :C, :D ]
    @test TC.gettensorheads(:(A[i,j] * (b * B[k] - c * C[k]) / d)) == [ :A, :B, :C ]

    @test TC.getscalars(:(a)) == [ :a ]
    @test TC.getscalars(:(1/a)) == [ :(1/a) ]
    @test TC.getscalars(:(b/a)) == [ :(b/a) ]
    @test TC.getscalars(:(b/a + c)) == [ :(b/a), :c ]
    @test TC.getscalars(:((b+c)/c)) == [ :((b+c)/c) ]
    @test TC.getscalars(:(a * A[i,j])) == [ :a ]
    @test TC.getscalars(:(2 * a * A[i,j])) == [ :a ]
    @test TC.getscalars(:(a * 2 * A[i,j])) == [ :a ]
    @test TC.getscalars(:(x * A[i,j] + B[j,i] * y)) == [ :x, :y ]
    @test TC.getscalars(:((x^2 + y)^2 * A[i,j] + B[j,i] * y)) == [ :x, :y ]
    @test TC.getscalars(:(cos(α) * A[i,j] + B[j,i] * sin(α))) == [ :α ]
    @test TC.getscalars(:((a * A[i,j] + B[i,j]) * C[j,k] * d + e * F[i])) == [ :a, :d, :e ]

    @test TC.splitprod(:(a)) == (nothing, [:a])
    @test TC.splitprod(:(a/b)) == (nothing, [:(a/b)])
    @test TC.splitprod(:((a+c)/b)) == (nothing, [:((a+c)/b)])
    @test TC.splitprod(:((a/b+c))) == (nothing, [:(a/b+c)])
    @test TC.splitprod(:(a * A[i])) == (:(A[i]), [:a])
    @test TC.splitprod(:(a * A[i] * b)) == (:(A[i]), [:a, :(b)])
    @test TC.splitprod(:(a * A[i] / b)) == (:(A[i]), [:a, :(1/b)])
    @test TC.splitprod(:(a * A[i] / b * c)) == (:(A[i]), [:a, :c, :(1/b)])
    @test TC.splitprod(:(a * A[i] * B[j] / b * c)) == (:(A[i] * B[j]), [:a, :c, :(1/b)])
    @test TC.splitprod(:(A[i] * B[j])) == (:(A[i] * B[j]), [])
    @test TC.splitprod(:(A[i] * A[j])) == (:(A[i] * A[j]), [])
    @test TC.splitprod(:(A[j] * A[j])) == (:(A[j] * A[j]), [])

    # ### invalid use
    # # invalid index pattern
    # @test_throws ArgumentError TC.getopenindices(:(A[i,j] * B[k,l] + C[k,l]))
    # # disallow / operator
    # @test_throws ArgumentError TC.getopenindices(:(α * A[i,j] * B[k,l] + β / C[k,l] * D[i,j]))
    # @test_throws ArgumentError TC.getopenindices(:(β \ C[k,l] * D[i,j]))
end


@testset "meinsum" begin

    A,B,C = [ TC.SymbolicTensor(s,4,4) for s in (:A,:B,:C) ]
    i,j,k = [ TC.Index(4) for _ = 1:3 ]

    expected = B
    got = TC.@meinsum A[i,j] = B[i,j]
    @test got == expected

    expected = B * B .+ C
    got = TC.@meinsum A[i,j] = B[i,k] * B[k,j] + C[i,j]
    @test got == expected

    expected = sum(bik -> bik^2, B[:])
    got = TC.@meinsum A = B[i,k] * B[i,k]
    @test got == expected

    D = TC.SymbolicTensor(:D,4)
    E = TC.SymbolicTensor(:E,4,4,4)
    expected = [ tr(E[i,:,:]) for i = 1:4 ]
    got = TC.@meinsum D[i] = E[i,k,k]
    @test got == expected

    M = TC.SymbolicTensor(:M,3,3)
    i,j = [ TC.Index(3) for _ = 1:2 ]
    expected = det(M)
    got = TC.@meinsum detM = det(M[i,j])
    @test got == expected

    A = TC.SymbolicTensor(:A,3,3)
    B = TC.SymbolicTensor(:B,3,3,3,3)
    C = TC.SymbolicTensor(:C,3,3)
    i,j,k,l = [ TC.Index(3) for _ = 1:4 ]
    got = TC.@meinsum begin
        C[i,j] = A[i,l] * B[k,l,k,j]
    end
    expected = zero(similar(A))
    for ii = i.rng, jj = j.rng, ll = l.rng, kk = k.rng
        expected[ii,jj] += A[ii,ll] * B[kk,ll,kk,jj]
    end
    subvals = Dict( var => rand(1:10) for var in vcat(A[:],B[:]) )
    eval_expected = [ SymEngine.subs(v, subvals) for v in expected ]
    eval_got = [ SymEngine.subs(v, subvals) for v in got ]
    @test eval_got == eval_expected

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


@testset "SymbolicTensor" begin

    # compute independent components of Riemann tensor in 2-5 dimensions
    # only testing against number of independents found
    for N = 2:4
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


function reduce_components_riemann_tensor(Riem)
    bianchi = permutedims(Riem,(1,2,3,4)) + permutedims(Riem,(1,3,4,2)) + permutedims(Riem,(1,4,2,3))
    Riem    = TC.resolve_dependents(Riem, bianchi)
    asym1   = permutedims(Riem,(1,2,3,4)) + permutedims(Riem,(2,1,3,4))
    Riem    = TC.resolve_dependents(Riem, asym1)
    asym2   = permutedims(Riem,(1,2,3,4)) + permutedims(Riem,(1,2,4,3))
    Riem    = TC.resolve_dependents(Riem, asym2)
    return Riem
end


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
    @test (eval(@components begin
        @index i,j,k = 4
        @symmetry B[i,j,k] = B[k,j,i]
    end); true)
    @test (eval(@components begin
        @index i = 4
        A[i] = B[1,i]
    end); true)
    @test (eval(@components begin
        @index i,j,k = 4
        @symmetry A[i,j,k] = A[j,i,k]
        @symmetry A[i,j,k] = A[i,k,j]
        @symmetry A[i,j,k] = A[k,j,i]
    end); true)

    # there are three kinds of functions we can use
    # - TensorComponents.scalar_to_scalar_funcs()
    # - TensorComponents.tensor_to_scalar_funcs()
    # - TensorComponents.tensor_to_tensor_funcs()
    #
    # scalar_to_scalar_funcs
    @test (eval(@components begin
        @index i = 4
        A[i] = cos(α[i])
    end); true)
    # also with nested functions
    @test (eval(@components begin
        @index i = 4
        A[i] = log(abs(α[i]))
    end); true)
    # tensor_to_scalar_funcs
    @test (eval(@components begin
        @index i,j = 4
        detA = det(A[i,j])
    end); true)
    # tensor_to_tensor_funcs
    @test (eval(@components begin
        @index i,j = 4
        adjugate_A[i,j] = adjugate(A[i,j])
    end); true)
    # we can mix them too
    @test (eval(@components begin
        @index i,j = 4
        logdetA = log(abs(det(A[i,j])))
    end); true)


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
    # reports an inconsinstency in rank by us, not @tensor
    @test_throws ErrorException TC.components(quote
        @index i, j, k = 4
        A[i,j] = A[i,j,k]
    end)
    # undefined indices with multiple equations
    @test_throws ErrorException TC.components(quote
        @index i, j, k = 4
        A[i,j] = B[i,j] * C[k,k]
        D[a]   = B[i,j]
    end)
    # inconsistent rank between symmetries and equations
    @test_throws ErrorException TC.components(quote
        @index i, j, k = 4
        @symmetry A[i,j,k] = A[j,i,k]
        A[i,j] = B[i,j,k] * C[k]
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
    @test_throws LoadError eval(TC.components(quote
        @index i, j, k = 4
        A[i,j] = B[i,j,j]
    end))
    @test_throws LoadError eval(TC.components(quote
        @index i, j = 1:4
        A[i,j] = B[i,j,j]
    end))
    @test_throws LoadError eval(TC.components(quote
        @index i, j, k = 3
        A[i,j] = B[i,j] + C[k]
    end))
    @test_throws LoadError eval(TC.components(quote
        @index i = 3
        A[i,i] = a
    end))

    # invalid function calls
    @test_throws ErrorException eval(TC.components(quote
        @index i,j = 3
        A[i,j] = a[i] * a[j]
        detA = det(A)
    end))
    @test_throws ErrorException eval(TC.components(quote
        @index i,j = 3
        A[i,j] = a[i] * a[j]
        adjugate_A[i,j] = adjugate(A)
    end))
    @test_throws ErrorException eval(TC.components(quote
        @index i,j = 3
        A[i,j] = a[i] * a[j]
        adjugate_A[i,j] = adjugate(A)
    end))


    # This should not throw
    # comps = @components begin
    #     @index a, b, c, d, k = 3
    #     ss[i] = 1/2 * ( Tuu[1,1] * β[j] * β[k] ) * (∂γ[i,j,k] - 2*hΓ[l,i,j]*γ[l,k])
    # end

end


@testset "codegen" begin

    input = "a1,a2,a3,a4,a5"

    output = TC.breakstring(input, breakchars=[','], breakafter=4)
    expected = "a1,a2,\na3,a4,\na5"
    @test output == expected

    output = TC.breakstring(input, breakchars=[','], breakafter=5)
    expected = "a1,a2,\na3,a4,a5"
    @test output == expected

    output = TC.breakstring(input, breakchars=[',',';',':'], breakafter=4)
    expected = "a1,a2,\na3,a4,\na5"
    @test output == expected

    output = TC.breakstring(input, breakchars=[';',':'], breakafter=4)
    expected = "a1,a2,a3,a4,a5"
    @test output == expected

    # input = "a1, a2, a3, a4, a5"

    # output = TC.breakstring(input, breakchars=[','], breakafter=6)
    # expected = "a1, a2,\na3, a4,\na5"
    # @test output == expected
    #
    # output = TC.breakstring(input, breakchars=[','], breakafter=7)
    # expected = "a1, a2, a3,\na4, a5"
    # @test output == expected

end
