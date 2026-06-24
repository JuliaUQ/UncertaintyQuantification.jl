struct StochasticProcessModel <: UQModel
    proc::AbstractStochasticProcess
    name::Symbol
end

function StochasticProcessModel(proc::AbstractStochasticProcess)
    return StochasticProcessModel(proc, proc.name)
end

function evaluate!(m::StochasticProcessModel, df::DataFrame)
    if isimprecise(m.proc)
        df[!, m.name] = Vector{AbstractVector{IntervalArithmetic.Interval}}(
            undef, size(df, 1)
        )
    else
        df[!, m.name] = missings(Vector{eltype(m.proc.time)}, size(df, 1))
    end

    ϕ = Matrix(df[:, m.proc.ϕnames])

    for i in axes(ϕ, 1)
        df[i, m.name] = evaluate(m.proc, ϕ[i, :])
    end

    return nothing
end

function isimprecise(m::StochasticProcessModel)
    return m.proc.psd isa ImprecisePSD
end

function bounds(m::StochasticProcessModel)
    @assert isimprecise(m)
    return [], []
end