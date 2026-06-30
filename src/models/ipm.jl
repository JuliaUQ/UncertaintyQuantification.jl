struct IntervalPredictorModel{T<:AbstractBasis} <: UQModel
    b::T
    p_lb::Vector{<:Real}
    p_ub::Vector{<:Real}
    inputs::Vector{Symbol}
    out::Symbol
    N::Integer # number of data points used to fit the IPM
    function IntervalPredictorModel(
        df::DataFrame,
        out::Symbol,
        b::T,
        inputs::Vector{Symbol}=propertynames(df[:, Not(out)]);
        tol::Real=1e-12,
    ) where {T<:AbstractBasis}
        X = permutedims(Matrix(df[:, inputs]))
        y = df[:, out]

        φ = b(X)

        N = size(X, 2)

        n = length(b)

        m = JuMP.Model(Clarabel.Optimizer)
        set_attribute(m, "tol_gap_abs", tol)
        set_attribute(m, "tol_gap_rel", tol)
        set_attribute(m, "tol_feas", tol)
        set_attribute(m, "tol_infeas_abs", tol)
        set_attribute(m, "tol_infeas_rel", tol)
        set_silent(m)

        @variable(m, p_lb[1:n])
        @variable(m, p_ub[1:n])

        @constraint(
            m, vec(p_ub' * ((φ - abs.(φ)) ./ 2) + p_lb' * ((φ + abs.(φ)) ./ 2)) .<= y
        )

        @constraint(
            m, vec(p_ub' * ((φ + abs.(φ)) ./ 2) + p_lb' * ((φ - abs.(φ)) ./ 2)) .>= y
        )

        @constraint(m, p_lb <= p_ub)

        @objective(m, Min, mean((p_ub - p_lb)' * abs.(φ)))

        JuMP.optimize!(m)

        return new{T}(b, value.(p_lb), value.(p_ub), inputs, out, size(df, 1))
    end
end

function evaluate!(ipm::IntervalPredictorModel, df::DataFrame; bound::Symbol=:both)
    X = permutedims(Matrix{Float64}(df[:, ipm.inputs])) # convert to matrix, sort by bfm.inputs
    φ = ipm.b(X)

    lo = vec(ipm.p_ub' * ((φ - abs.(φ)) ./ 2) + ipm.p_lb' * ((φ + abs.(φ)) ./ 2))
    hi = vec(ipm.p_ub' * ((φ + abs.(φ)) ./ 2) + ipm.p_lb' * ((φ - abs.(φ)) ./ 2))

    # check for possible crossover of the bounds
    idx = lo .> hi
    if !isempty(idx)
        lo[idx], hi[idx] = hi[idx], lo[idx]
    end

    df[!, ipm.out] = Interval.(lo, hi)

    return nothing
end

"""
    reliability(ipm::IntervalPredictorModel, ϵ::Real)
Returns the confidence parameter ``\\beta \\in (0,1)``, such that the reliability of the IPM,
that is the probability unobserved data points will fall in its bounds,
is no less than ``1 -  \\epsilon`` with confidence ``1 - \\beta``, with ``\epsilon \\in (0,1)``. 
"""
function reliability(ipm::IntervalPredictorModel, ϵ::Real)
    @assert 0 < ϵ < 1
    return cdf(Binomial(ipm.N, ϵ), 2 * length(ipm.b) - 1)
end

isimprecise(ipm::IntervalPredictorModel) = true

function bounds(ipm::IntervalPredictorModel)
    return min.(ipm.p_lb, ipm.p_ub), max.(ipm.p_lb, ipm.p_ub)
end

name(ipm::IntervalPredictorModel) = ipm.out