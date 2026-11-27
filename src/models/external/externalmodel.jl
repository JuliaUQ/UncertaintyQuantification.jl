struct ExternalModel <: UQModel
    sourcedir::String
    sources::Vector{String}
    extractors::Vector{Extractor}
    solver::Solver
    workdir::String
    extras::Vector{String}
    formats::Dict{Symbol, String}
    cleanup::Bool
    scheduler::Union{<:AbstractHPCScheduler, Nothing}

    function ExternalModel(
            sourcedir::String,
            sources::Union{String, Vector{String}},
            extractors::Union{Extractor, Vector{Extractor}},
            solver::Solver,
            workdir::String,
            extras::Union{String, Vector{String}},
            formats::Dict{Symbol, String},
            cleanup::Bool,
            scheduler::Union{<:AbstractHPCScheduler, Nothing},
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
        sources::Union{String, Vector{String}},
        extractors::Union{Extractor, Vector{Extractor}},
        solver::Solver;
        workdir::String = tempname(),
        extras::Union{String, Vector{String}} = String[],
        formats::Dict{Symbol, String} = Dict{Symbol, String}(),
        cleanup::Bool = false,
        scheduler::Union{<:AbstractHPCScheduler, Nothing} = nothing,
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
        try
            rm(path; recursive = true)
        catch e
            @warn "Error during cleanup: $e"
        end
    end
    return result
end

function createpath(workdir::String, sim_folder, subset_lvl, sample_folder, i, digits, add_s::Bool, add_num::Bool)
    folder = isnothing(sample_folder) ? "sample-$(lpad(i, digits, "0"))" : (string(sample_folder,add_s ? "-sample" : "" , add_num ? "-$(lpad(i, digits, "0"))" : ""))
    return joinpath(workdir, sim_folder, isnothing(subset_lvl) ? "" : "level-$(subset_lvl)", folder)
end

function evaluate!(
        m::ExternalModel,
        df::DataFrame;
        sim_folder::String = Dates.format(now(), "YYYY-mm-dd-HH-MM-SS"),
        subset_lvl = nothing,
        sample_folder = nothing,
        is_subsetinfinity = false,
        add_num = true,
    )
    print("Evaluate EXT folder $sim_folder lvl $subset_lvl\n")
    if !isnothing(m.scheduler)
        print("no scheduler")
        return evaluate!(
            m, df, m.scheduler; sim_folder = sim_folder, subset_lvl = subset_lvl, sample_folder = sample_folder, is_subsetinfinity = is_subsetinfinity
        )
    end


    n = size(df, 1)
    digits = ndigits(n)

    results = pmap(1:n) do i
        path = createpath(m.workdir, sim_folder, subset_lvl, sample_folder, i, digits, is_subsetinfinity, add_num)
        makedirectory(m, df[i, :], path)
        run(m.solver, path)
        result = getresult(m, path)

        return result
    end

    results = hcat(results...)

    for (i, name) in enumerate(names(m.extractors))
        df[!, name] = results[i, :]
    end
    return
end

function evaluate!(
        m::ExternalModel,
        df::DataFrame,
        scheduler::AbstractHPCScheduler;
        sim_folder::String = Dates.format(now(), "YYYY-mm-dd-HH-MM-SS"),
        subset_lvl = nothing,
        sample_folder = nothing,
        is_subsetinfinity = false,
        add_num=true,
    )
    n = size(df, 1)
    digits = ndigits(n)

    for i in 1:n
        path = createpath(m.workdir, sim_folder, subset_lvl, sample_folder, i, digits, is_subsetinfinity, add_num)
        makedirectory(m, df[i, :], path)
    end

    setup_hpc_jobs(scheduler, m, n, sim_folder)

    run_hpc_jobs(scheduler, m, n, sim_folder)

    results = map(1:n) do i
        path = createpath(m.workdir, sim_folder, subset_lvl, sample_folder, i, digits, is_subsetinfinity, add_num)
        return getresult(m, path)
    end

    results = hcat(results...)

    for (i, name) in enumerate(names(m.extractors))
        df[!, name] = results[i, :]
    end
    return
end

function formatinputs(row::DataFrameRow, formats::Dict{Symbol, String})
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
