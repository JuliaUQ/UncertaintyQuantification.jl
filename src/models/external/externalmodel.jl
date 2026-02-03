struct ExternalModel <: UQModel
    sourcedir::String
    sources::Vector{String}
    extractors::Vector{Extractor}
    solver::Solver
    workdir::String
    extras::Vector{String}
    formats::Dict{Symbol,String}
    cleanup::Bool
    scheduler::Union{<:AbstractHPCScheduler,Nothing}

    function ExternalModel(
        sourcedir::String,
        sources::Union{String,Vector{String}},
        extractors::Union{Extractor,Vector{Extractor}},
        solver::Solver,
        workdir::String,
        extras::Union{String,Vector{String}},
        formats::Dict{Symbol,String},
        cleanup::Bool,
        scheduler::Union{<:AbstractHPCScheduler,Nothing},
    )
        sources, extractors, extras = wrap.([sources, extractors, extras])
        return new(
            sourcedir,
            sources,
            extractors,
            solver,
            workdir,
            extras,
            formats,
            cleanup,
            scheduler,
        )
    end
end

function ExternalModel(
    sourcedir::String,
    sources::Union{String,Vector{String}},
    extractors::Union{Extractor,Vector{Extractor}},
    solver::Solver;
    workdir::String=tempname(),
    extras::Union{String,Vector{String}}=String[],
    formats::Dict{Symbol,String}=Dict{Symbol,String}(),
    cleanup::Bool=false,
    scheduler::Union{<:AbstractHPCScheduler,Nothing}=nothing,
)
    return ExternalModel(
        sourcedir, sources, extractors, solver, workdir, extras, formats, cleanup, scheduler
    )
end

function makedirectory(m::ExternalModel, sample, path::String)
    mkpath(path)

    row = formatinputs(sample, m.formats)

    for file in m.sources
        if isempty(file)
            continue
        end
        tokens = Mustache.load(joinpath(m.sourcedir, file))

        open(joinpath(path, file), "w") do io
            render(io, tokens, row)
        end
    end

    for file in m.extras
        if isempty(file)
            continue
        end
        cp(joinpath(m.sourcedir, file), joinpath(path, file))
    end
    return nothing
end

function getresult(m::ExternalModel, path::String)
    result = map(e -> e.f(path), m.extractors)
    if m.cleanup
        rm(path; recursive=true)
    end
    return result
end

function evaluate!(
    m::ExternalModel,
    df::DataFrame;
    datetime::String=Dates.format(now(), "YYYY-mm-dd-HH-MM-SS"),
)
    if !isnothing(m.scheduler)
        return evaluate!(m, df, m.scheduler; datetime=datetime)
    end

    n = size(df, 1)
    digits = ndigits(n)

    results = pmap(1:n) do i
        path = joinpath(m.workdir, datetime, "sample-$(lpad(i, digits, "0"))")

        makedirectory(m, df[i, :], path)
        run(m.solver, path)
        result = getresult(m, path)

        return result
    end

    results = hcat(results...)

    for (i, name) in enumerate(names(m.extractors))
        df[!, name] = results[i, :]
    end
end

function evaluate!(
    m::ExternalModel,
    df::DataFrame,
    scheduler::AbstractHPCScheduler;
    datetime::String=Dates.format(now(), "YYYY-mm-dd-HH-MM-SS"),
)
    n = size(df, 1)
    digits = ndigits(n)

    for i in 1:n
        path = joinpath(m.workdir, datetime, "sample-$(lpad(i, digits, "0"))")
        makedirectory(m, df[i, :], path)
    end

    setup_hpc_jobs(scheduler, m, n, datetime)

    run_hpc_jobs(scheduler, m, n, datetime)

    results = map(1:n) do i
        path = joinpath(m.workdir, datetime, "sample-$(lpad(i, digits, "0"))")
        return getresult(m, path)
    end

    results = hcat(results...)

    for (i, name) in enumerate(names(m.extractors))
        df[!, name] = results[i, :]
    end
end

function formatinputs(row::DataFrameRow, formats::Dict{Symbol,String})
    names = propertynames(row)
    values = []
    for symbol in names
        if haskey(formats, symbol)
            push!(values, formatinput(row[symbol], formats[symbol]))
        elseif haskey(formats, :*)
            push!(values, formatinput(row[symbol], formats[:*]))
        else
            push!(values, row[symbol])
        end
    end
    return (; zip(names, values)...)
end

function formatinput(data::Real, formatstring::String)
    return pyfmt(formatstring, data)
end

function formatinput(data::AbstractVector{<:Real}, formatstring::String)
    return pyfmt.(formatstring, data)
end
