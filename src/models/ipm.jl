struct IntervalPredictorModel{T<:AbstractBasis} <: UQModel
    b::T
    p_lb::Vector{<:Real}
    p_ub::Vector{<:Real}
    inputs::Vector{Symbol}
    out::Symbol
    function IntervalPredictorModel(
        df::DataFrame,
        out::Symbol,
        b::T,
        inputs::Vector{Symbol}=propertynames(df[:, Not(out)]),
    ) where {T<:AbstractBasis}
        X = permutedims(Matrix(df[:, inputs]))
        y = df[:, out]

        φ = b(X)

        N = size(X, 2)

        n = length(b)

        m = JuMP.Model(Ipopt.Optimizer)
        @variable(m, p_lb[1:n])
        @variable(m, p_ub[1:n])

        @constraint(
            m, vec(p_ub' * ((φ - abs.(φ)) ./ 2) + p_lb' * ((φ + abs.(φ)) ./ 2)) <= y
        )

        @constraint(
            m, vec(p_ub' * ((φ + abs.(φ)) ./ 2) + p_lb' * ((φ - abs.(φ)) ./ 2)) >= y
        )

        @objective(m, Min, sum((p_ub - p_lb)' * abs.(φ)))

        JuMP.optimize!(m)

        return new{T}(b, value.(p_lb), value.(p_ub), inputs, out)
    end
end

function evaluate!(ipm::IntervalPredictorModel, df::DataFrame)
    X = permutedims(Matrix{Float64}(df[:, ipm.inputs])) # convert to matrix, sort by bfm.inputs
    φ = ipm.b(X)

    lo = vec(ipm.p_ub' * ((φ - abs.(φ)) ./ 2) + ipm.p_lb' * ((φ + abs.(φ)) ./ 2))
    hi = vec(ipm.p_ub' * ((φ + abs.(φ)) ./ 2) + ipm.p_lb' * ((φ - abs.(φ)) ./ 2))

    idx = findall(.!(lo .<= hi))

    @show lo[idx], hi[idx]

    df[!, ipm.out] = Interval.(lo, hi)
    return nothing
end