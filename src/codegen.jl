macro generate_code(comps, outputs, kwargs...)
    return esc(:(TensorComponents.generate_code($(string(__source__.file)),
                                                $comps, $outputs; $kwargs...)))
end


function nextind_bychars(str, start, nchars)
    c = 1
    i = nextind(str, min(start,ncodeunits(str)))
    while c <= nchars && i < ncodeunits(str)
        i = nextind(str, i)
        c += 1
    end
    return i
end


breakstring(str::AbstractString, breakchar::Char=','; breakafter=80) =
    breakstring(str, [breakchar], breakafter=breakafter)
function breakstring(str::AbstractString, breakchars::Vector{Char}=[',']; breakafter=80)

    display(str)
    length(str) <= breakafter && return str
    buf = IOBuffer()
    start = nextind(str,0)
    stop = nextind_bychars(str,0,breakafter)

    while start <= ncodeunits(str)
        stop = findnext(str, stop) do c
            c in breakchars
        end
        if isnothing(stop)
            stop = ncodeunits(str)
        end
        println(buf, view(str,start:stop))
        # move on
        start = nextind(str,stop)
        skipspace = match(r"^(\s*)", view(str, start:length(str)))
        if !isnothing(skipspace)
            start += length(skipspace[1])
        end
        stop = min(nextind_bychars(str, start, breakafter),ncodeunits(str))
    end

    return String(take!(buf))
end


function generate_code(codegen_filename, comps, outputs; kwargs...)
    rest, ext = splitext(codegen_filename)
    func_name = basename(rest)
    src_filename = length(ext) == 0 ? "$rest.codegen" : "$rest.codegen$ext"

    @info "Exporting to '$src_filename' ..."
    code, ins, outs, libdeps = _generate_code(func_name, comps, outputs; kwargs...)

    str_ins  = join(ins,',')
    str_outs = join(outs,',')
    str_ins = breakstring(str_ins,',')
    str_outs = breakstring(str_outs,',')

    str_detected  = "detected input variables:\n$str_ins"
    str_requested = "requested output variables:\n$str_outs"
    @info str_detected
    @info str_requested

    # insert comment chars
    str_detected = "# " * replace(str_detected, '\n' => "\n# ", count=max(count('\n',str_detected)-1,1))
    str_requested = "# " * replace(str_requested, '\n' => "\n# ", count=max(count('\n',str_requested)-1,1))

    # gather operation counts and prepare printer
    stats = count_operations(comps)
    ks, vs = collect(keys(stats)), collect(values(stats))
    P = sortperm(vs, rev=true)
    ks, vs = ks[P], vs[P]
    total = sum(v for v in vs)
    prcnts = [ v/total * 100 for v in vs ]
    longest_key = maximum(length(string(k)) for k in ks)
    longest_val = ndigits(total)
    fmt = Printf.Format("# %$(longest_key)s = %$(longest_val)d ... %5.1f %%\n")

    open(src_filename,"w") do file
        n = now()
        println(file,
                """
                # Generated on $(Date(n)) - $(Time(n))
                # Operator counts""")
        for (k,v,p) in zip(ks,vs,prcnts)
            Printf.format(file, fmt, k, v, p)
        end
        Printf.format(file, fmt, "Σ", total, 100)
        println(file)
        println(file, str_detected)
        println(file, str_requested)
        println(file)
        for dep in libdeps
            println(file, "using $dep")
        end
        println(file)
        write(file, string(code))
    end

    return src_filename
end


function _generate_code(func_name, comps, outputs; with_LV::Bool=false)

    # convert to expression again
    eqs = Expr[]
    for comp in comps
        lhs, rhs = Meta.parse(string(comp[1])), Meta.parse(string(comp[2]))
        push!(eqs, :($lhs = $rhs))
    end

    # extract independents
    deps, ideps = determine_dependents_independents(eqs)

    if isempty(outputs)
        throw(ArgumentError("no dependent variables provided"))
    end

    computed_outputs = []
    inputs = ideps
    foreach(outputs) do out
        if !(out in deps)
            @warn "skipping unused output variable '$out'"
            return
        end
        push!(computed_outputs, out)
    end

    cachevar = :CACHE
    idxvar   = :IDX

    if cachevar in deps || cachevar in ideps || idxvar in deps || idxvar in ideps
        throw(ArgumentError("detected name clash between a variable and the function internal variables '$idxvar,$cachevar'"))
    end

    loopbody = Expr(:block)
    loop = Expr(:for, :($idxvar = 1:N), loopbody)
    for idep in ideps
        push!(loopbody.args, :($idep = $cachevar.$idep[$idxvar]))
    end
    for eq in eqs
        push!(loopbody.args, eq)
    end
    for dep in computed_outputs
        push!(loopbody.args, :($cachevar.$dep[$idxvar] = $dep))
    end

    libdependencies = Symbol[]
    code = if with_LV
        push!(libdependencies, :LoopVectorization)
        quote
            function $(Symbol(func_name))($cachevar, N)
                @tturbo $loop
            end
        end
    else
        quote
            function $(Symbol(func_name))($cachevar, N)
                $loop
            end
        end
    end |> MacroTools.unblock
    code = MacroTools.postwalk(MacroTools.rmlines, code)

    return code, inputs, computed_outputs, libdependencies
end


# Walk through equations and determine tensors which we have not been computed before
# they are used. E.g. returns all tensors that did not appear in a lhs before they first
# appeared in a rhs.
function determine_dependents_independents(eqs)

    deps = Symbol[]
    ideps = Symbol[]

    N = length(eqs)

    N == 0 && return deps, ideps

    for eq in eqs
        lhs, rhs = getlhs(eq), getrhs(eq)
        !(lhs in deps) && push!(deps, lhs)
        foreach(getscalars(rhs)) do s
            !(s in deps) && !(s in ideps) && push!(ideps, s)
        end
    end

    return deps, sort!(ideps)
end


macro test_code(comps, outs)
    filename = string(__source__.file)
    rest, ext = splitext(filename)
    func_name = Symbol(basename(rest))
    src_filename = length(ext) == 0 ? "$rest.codegen" : "$rest.codegen$ext"

    code = quote
        let
            eqs = Expr[]
            for comp in $comps
                lhs, rhs = Meta.parse(string(comp[1])), Meta.parse(string(comp[2]))
                push!(eqs, Expr(:(=), lhs, rhs))
            end

            _, ideps = TensorComponents.determine_dependents_independents(eqs)

            using Random
            Random.seed!(123)
            N = 100
            cache = NamedTuple( (var,abs.(randn(N))) for var in vcat($outs,ideps) )

            @info "Testing '$($(src_filename))'"
            @time "include(\"$($(src_filename))\")" include($src_filename)
            @time "$($(func_name))(cache, $N)" $func_name(cache,N)
        end
    end

    return esc(code)
end
