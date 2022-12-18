struct Index
    rng::UnitRange{Int}
    function Index(rng::UnitRange{Int})
        rng.start == 1 || error("Index must start at 0")
        rng.start < rng.stop || error("Index range must be increasing and not empty")
        return new(rng)
    end
end

Base.view(t::AbstractArray, idxs::Index...) = view(t.t, (i.rng for i in idxs)...)

# macro indices(expr)
#     return esc(indices(expr))
# end
#
# function indices(expr)
#     display(expr)
# end
