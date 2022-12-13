using TensorComponents
using TensorComponents.TensorOperations

i, j, k, u = [ TensorComponents.Index(1:4) for _ = 1:4 ]
A, B, C = [ TensorComponents.SymbolicTensor(h, i, j) for h in [:A, :B, :C] ]

# expr = @macroexpand1 @expand begin
@expand begin
    # begin
    #     A[1,k] = 0
    #     A[k,1] = 0
    #     A[a,b] = diag([1,1,1])
    #     B[i,j] = A[i,k] * A[k,j]
    # end
    D[i,j] := A[u,i] * A[u,k] * B[k,j] - A[k,j] * B[i,k]
    F[i,j] := A[i,k] * B[k,j] - A[k,j] * B[i,k]
end

# display(expr)
