struct Index
    rng::UnitRange{Int}
    function Index(rng::UnitRange{Int})
        rng.start == 1 || error("Index must start at 0")
        rng.start < rng.stop || error("Index range must be increasing and not empty")
        return new(rng)
    end
end

# macro indices(expr)
#     return esc(indices(expr))
# end
#
# function indices(expr)
#     display(expr)
# end
