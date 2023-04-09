using Test

using TensorComponents
using TensorOperations
using SymEngine

const TC = TensorComponents


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

# # multiple index definitions
# @index (4) i j k l m i j k
#
# # invalid tensor definition
# @input A[i,j]<1 B[i,j,k] C[i]
#
# # multiple tensor definitions
# @input A[i,j] A[i,j] B[i,j,k] C[i]
#
# # invalid symmetry relation
# @input A[i,j] - A[j,i]
# @input A[i,j] == A[j,i]
# @input A[i,j] = A[j,i,j]
