struct StochasticProcessModel <: UQModel
    proc::AbstractStochasticProcess
    name::Symbol
end

function StochasticProcessModel(proc::AbstractStochasticProcess)
    return StochasticProcessModel(proc, proc.name)
end

function evaluate!(m::StochasticProcessModel, df::DataFrame)
    ϕ = Matrix(df[:, m.proc.ϕnames])

    df[!, m.name] = missings(Vector{eltype(m.proc.time)}, size(df, 1))
    for i in axes(ϕ, 1)
        df[i, m.name] =
            sqrt(2) * vec(df[i, "$(m.proc.name)_A"]' * cos.(m.proc.ωt .+ ϕ[i, :]))
    end

    return nothing
end