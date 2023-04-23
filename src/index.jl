struct Index
    rng::UnitRange{Int}
    function Index(rng::UnitRange{Int})
        rng.start == 1 || error("Index must start at 1")
        rng.start < rng.stop || error("Index range must be increasing and not empty")
        return new(rng)
    end
end

Index(i::Int) = Index(1:i)

Base.view(t::AbstractArray, idxs::Union{Index,Int}...) = view(t, (i isa Index ? i.rng : i for i in idxs)...)

Base.length(i::Index) = length(i.rng)

start(idx::Index) = idx.rng.start
stop(idx::Index) = idx.rng.stop
start(idx::Int) = idx
stop(idx::Int) = idx


Base.iterate(i::Index) = iterate(i.rng)
Base.iterate(i::Index,state) = iterate(i.rng,state)
