struct Index
    rng::UnitRange{Int}
    function Index(rng::UnitRange{Int})
        rng.start == 1 || error("Index must start at 1")
        rng.start < rng.stop || error("Index range must be increasing and not empty")
        return new(rng)
    end
end

Index(i::Int) = Index(1:i)

Base.view(t::AbstractArray, idxs::Index...) = view(t, (i.rng for i in idxs)...)

Base.length(i::Index) = length(i.rng)
