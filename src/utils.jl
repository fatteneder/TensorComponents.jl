# topological sort -- Kahn's algorithm
# see https://en.wikipedia.org/wiki/Topological_sorting
function topological_sort(nodes::Vector{T}, childs::Vector{Vector{T}}) where T

    # find set with no incoming edges/parents
    allchilds = unique!(reduce(vcat, childs))
    seq_noparents = Int[]
    for (i,n) in enumerate(nodes)
        if !(n in allchilds)
            push!(seq_noparents, i)
        end
    end

    tmpchilds = deepcopy(childs)

    seq = Int[] # sorted sequence
    while !isempty(seq_noparents)

        i = pop!(seq_noparents)
        push!(seq, i)

        # cut all ties to this node
        chs = copy(tmpchilds[i])
        empty!(tmpchilds[i])

        # add any of the dropped childrens which now have no parents to queue
        for c in chs
            if any(tmpchs -> c in tmpchs, tmpchilds)
                continue
            end
            j = findfirst(==(c), nodes)
            # ignore any nodes which are not in the graph
            isnothing(j) && continue
            push!(seq_noparents, j)
        end
    end

    n_rest_childs = mapreduce(sum, +, tmpchilds)
    if n_rest_childs != 0
        error("Failed to sort, because graph has at least on cycle")
    end

    return seq
end


function test_topological_sort()
    nodes  = [ 10, 8, 5, 3, 2, 11, 9, 7 ]
    childs = Vector{Int}[ [], [9,10], [11], [8], [], [2,9], [], [11,8] ]
    seq = topological_sort(nodes, childs)
    sorted_nodes = nodes[seq]
end
