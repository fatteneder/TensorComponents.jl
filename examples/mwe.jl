using TensorComponents
using TensorOperations
using SymEngine


# @macroexpand @components begin
@components begin

    @index (4) i j k l

    # Ignoring this for the first draft.
    # # optional for now
    # @symmetry A[i,j] = A[j,i]

    # TODO How do detect that a here is conflicting with index a above?
    # We don't.
    A[i,j] = B[i,j,k] * C[k]
    # A[i,j,k] = B[i,j,k] * C[k]
    A[i,j] = B[i,j,k] * C[k] # * F[i,j]
    F[i,j] = G[i,j,k] * H[k]
    # F[i,j] * F[a,b] = G[i,j,k] * H[k] * A[a,b]
    # ...
    # ...

    # Ignoring this for the first draft.
    # # figure out if this can be converted to A11 or wave have to error and instead need
    # # to define a11 = a[1,1] first and then use if a11 > 0
    # if A[1,1] > 0
    #     display("orsch")
    # end
    # begin
    #     sers = oida
    # end

end
