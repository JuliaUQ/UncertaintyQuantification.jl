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

        # spread of the IPM
        function δ(p::Vector{<:Real})
            p_lb = p[1:n]
            p_ub = p[(n + 1):end]
            return sum([abs(sum(p_ub .* x) - sum(p_lb .* x)) for x in eachcol(φ)])
        end

        # constraints for the upper and lower bounds
        function con!(c, p::Vector{<:Real})
            c_lb = [dot(p[1:n], x) for x in eachcol(φ)]
            c_ub = [dot(p[(n + 1):end], x) for x in eachcol(φ)]
            c[1:N] .= c_lb
            c[(N + 1):end] .= c_ub
            return nothing
        end

        lc = [fill(-Inf, N)..., y...]
        uc = [y..., fill(Inf, N)...]

        lx = fill(-Inf, 2 * n)
        ux = fill(Inf, 2 * n)

        dfc = TwiceDifferentiableConstraints(con!, lx, ux, lc, uc)

        x0 = [-10.0 * ones(n)..., 10.0 * ones(n)...]

        res = optimize(δ, dfc, x0, IPNewton(); autodiff=ADTypes.AutoForwardDiff())

        p_lb = res.minimizer[1:n]
        p_ub = res.minimizer[(n + 1):end]

        return new{T}(b, p_lb, p_ub, inputs, out)
    end
end

function evaluate!(ipm::IntervalPredictorModel, df::DataFrame)
    X = permutedims(Matrix{Float64}(df[:, ipm.inputs])) # convert to matrix, sort by bfm.inputs
    φ = ipm.b(X)
    lo = map(x -> dot(x, ipm.p_lb), eachcol(φ))
    hi = map(x -> dot(x, ipm.p_ub), eachcol(φ))

    df[!, ipm.out] = Interval.(lo, hi)
    return nothing
end