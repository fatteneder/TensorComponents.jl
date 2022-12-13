macro components(expr)
    return esc(components(expr))
end


function components(expr)
    expr = TO.removelinenumbernode(expr)::Expr
    if expr.head === :block
        for ex in expr.args
            components(ex)
        end
        return
    end

    display(expr.head)
    if expr.head === :macrocall
        if expr.args[1] === Symbol("@indices")
            range, idxs = parseindices(expr.args[2:end])
        else
            error("Unknown macro '$(expr.args[1])' encountered in scope of @components: $expr")
        end
        return
    end

    TO.isassignment(expr) ||
        error("Can only process assignments like A[i,j] = a[i] * b[j], found '$expr'")

    lhs, rhs = TO.getlhs(expr), TO.getrhs(expr)
    !TO.isassignment(rhs) ||
        error("Do not support nested assignments like A[i,j] = B[i,j] = C[i,j], found '$expr'")

    TO.istensor(lhs) ||
        error("LHS must a simple tensor like A[i,j], found $lhs")

    rhstensors = TO.gettensors(rhs)
    rhsdecmps = TO.decomposetensor.(rhstensors)
    display(rhsdecmps)

    # TODO
    # 1. Declare indices
    # 2. (Optional) Declare symmetries
    # 3. processes each line of @components with 
end


function parseindices(expr)
    for e in expr
        if hasproperty(e, :head)
            println(e.head)
            println(e.args)
        end
    end
    return (), ()
end


macro expand(expr)
    return esc(expand(expr))
end

function expand(expr)
    expr = TO.removelinenumbernode(expr)::Expr
    if expr.head === :block
        # results = [ :($(Symbol(:result,i)) = $(expandrow(ex))) for (i,ex) in enumerate(expr.args) ]
        blocks = [ expandrow(ex) for ex in expr.args ]
        vars = [ Symbol(:var, i) for i = 1:length(blocks) ]
        vars_blocks = [ :($var = $block) for (var, block) in zip(vars, blocks) ]
        # vec_vars = :($(vars...))
        # return Expr(:block, blocks...)
        # return Expr(:block, vars_blocks..., return_vars)
        return quote
            $(vars_blocks...)
            return [$(vars...)]
        end
    end
end

function expandrow(expr)

    expr.head === :block && error("Nested blocks not implemented")
    # display(expr.head)
    # if expr.head === :macrocall
    #     if expr.args[1] === Symbol("@indices")
    #         range, idxs = parseindices(expr.args[2:end])
    #     else
    #         error("Unknown macro '$(expr.args[1])' encountered in scope of @components: $expr")
    #     end
    #     return
    # end

    TO.isdefinition(expr) ||
        error("Can only process definitions like A[i,j] := a[i] * b[j], found '$expr'")

    lhs, rhs = TO.getlhs(expr), TO.getrhs(expr)
    (TO.isassignment(rhs) || TO.isdefinition(rhs)) &&
        error("Do not support nested assignments like A[i,j] = B[i,j] = C[i,j], found '$expr'")

    TO.istensor(lhs) ||
        error("LHS must a simple tensor like A[i,j], found '$lhs'")

    rhstensors = TO.gettensors(rhs)
    rhsdecmps = TO.decomposetensor.(rhstensors)
    # display(rhsdecmps)
    lhstensors = TO.gettensors(lhs)
    lhsdecmps = TO.decomposetensor.(lhstensors)
    # display(lhsdecmps)

    # TODO unqiue the views

    rhsheads = [ r[1] for r in rhsdecmps ]
    rhsindices = [ r[2] for r in rhsdecmps ]
    lhsheads = [ l[1] for l in lhsdecmps ]
    lhsindices = [ l[2] for l in lhsdecmps ]

    # we should have all ingredients now
    rhsviews = [ :($(Symbol(h, is...)) = view($h, $(is...))) for (h,is) in zip(rhsheads, rhsindices) ]
    # display(rhsviews)

    # tensor = TO.defaultparser(expr)
    tensor = :(@tensor $expr)
    # tensor = Expr(:macrocall, Symbol("@tensor"),
    #               LineNumberNode(@__LINE__, Symbol(@__FILE__)), expr)
    # display(tensor)

    letins = [ :($r=$r) for r in rhsheads ]
    unique!(letins)
    letouts = [ :($l=$l) for l in lhsheads ]
    unique!(letouts)
    # display(letins)

    code = quote
        # let $(letins...), $(letouts...)
        let $(letins...)
            $(rhsviews...)
            $(tensor)
            $(lhsheads[1])
        end
    end
    # display(code)

    return code
end
