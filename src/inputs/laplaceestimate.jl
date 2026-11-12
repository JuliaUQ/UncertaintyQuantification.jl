"""
    LaplaceEstimate(
        x0::DataFrame; islog = true, lowerbounds, upperbounds,
        filtertolerance = 1.0e-6, optimizer = Optim.LBFGS(),
        optimoptions = Optim.Options(), adbackend = AutoFiniteDiff()
    )

Configure a Laplace approximation of a target density. Each row of `x0` is an
optimization start, and its columns name the target's variables. The estimator
does not contain a prior. Use [`laplaceestimate`](@ref) for a general density or
[`bayesianupdating`](@ref) for a Bayesian posterior.

`islog` specifies whether the target (or Bayesian likelihood) returns log
densities. `lowerbounds` and `upperbounds` default to unbounded intervals, with
one entry per variable. Modes closer than `filtertolerance` are merged; set it
to zero to retain all optimization results.

`optimizer` and `optimoptions` control Optim.jl. `adbackend` selects the
DifferentiationInterface.jl backend used to calculate the Hessian of the negative
log density at each mode. Its inverse gives the Gaussian covariance. The result
is a `JointDistribution` containing a Gaussian mixture, with component weights
proportional to the target density at the modes.
"""
struct LaplaceEstimate
    x0::DataFrame
    islog::Bool
    lowerbounds::Vector{Float64}
    upperbounds::Vector{Float64}
    filtertolerance::Float64
    optimizer::Optim.AbstractOptimizer
    optimoptions::Optim.Options
    adbackend::ADTypes.AbstractADType

    function LaplaceEstimate(
            x0::DataFrame;
            islog::Bool = true,
            lowerbounds::AbstractVector{<:Real} = fill(-Inf, ncol(x0)),
            upperbounds::AbstractVector{<:Real} = fill(Inf, ncol(x0)),
            filtertolerance::Real = 1.0e-6,
            optimizer::Optim.AbstractOptimizer = Optim.LBFGS(),
            optimoptions::Optim.Options = Optim.Options(),
            adbackend::ADTypes.AbstractADType = AutoFiniteDiff(),
        )
        nrow(x0) > 0 && ncol(x0) > 0 || throw(ArgumentError("x0 must contain at least one starting point and variable."))
        length(lowerbounds) == length(upperbounds) == ncol(x0) || throw(DimensionMismatch("Bounds must have one entry per variable."))
        all(lowerbounds .< upperbounds) || throw(ArgumentError("Lower bounds must be smaller than upper bounds."))
        isfinite(filtertolerance) && filtertolerance >= 0 || throw(ArgumentError("filtertolerance must be finite and nonnegative."))
        starts = Matrix{Float64}(x0)
        all(isfinite, starts) || throw(ArgumentError("Starting points must be finite."))
        all(lowerbounds' .<= starts .<= upperbounds') || throw(ArgumentError("Starting points must lie within the bounds."))
        return new(
            DataFrame(starts, names(x0)), islog, Float64.(lowerbounds), Float64.(upperbounds),
            Float64(filtertolerance), optimizer, optimoptions, adbackend,
        )
    end
end

"""
    laplaceestimate(target::Function, estimate::LaplaceEstimate)

Approximate `target` with a Gaussian mixture using the settings in `estimate`.
The target accepts a DataFrame and returns one density per row (log densities
when `estimate.islog` is true). Returns a [`JointDistribution`](@ref) with the
variable names from `estimate.x0`.

```julia
estimate = LaplaceEstimate(DataFrame(x = [-1.0, 1.0]))
approximation = laplaceestimate(df -> -0.5 .* df.x .^ 2, estimate)
```
"""
function laplaceestimate(target::Function, estimate::LaplaceEstimate)
    logtarget = estimate.islog ? target : df -> log.(target(df))
    return _laplaceestimate(logtarget, estimate)
end

# Shared core: Bayesian wrappers already produce a log posterior regardless of
# whether the user's likelihood was supplied as a density or a log density.
function _laplaceestimate(logtarget::Function, estimate::LaplaceEstimate)
    variables = Symbol.(names(estimate.x0))
    objective = x -> -only(logtarget(DataFrame(permutedims(x), variables)))
    modes = Vector{Float64}[]
    logvalues = Float64[]
    unbounded = all(==(-Inf), estimate.lowerbounds) && all(==(Inf), estimate.upperbounds)

    for row in eachrow(estimate.x0)
        start = collect(row)
        result = if unbounded
            Optim.optimize(objective, start, estimate.optimizer, estimate.optimoptions)
        else
            Optim.optimize(
                objective, estimate.lowerbounds, estimate.upperbounds,
                start, Optim.Fminbox(estimate.optimizer), estimate.optimoptions
            )
        end
        mode = Optim.minimizer(result)
        logvalue = -Optim.minimum(result)
        isfinite(logvalue) || throw(ArgumentError("The estimated mode must have finite log density."))
        if estimate.filtertolerance > 0 && any(x -> norm(x - mode) < estimate.filtertolerance, modes)
            continue
        end
        push!(modes, copy(mode))
        push!(logvalues, logvalue)
    end

    components = map(modes) do mode
        curvature = DifferentiationInterface.hessian(objective, estimate.adbackend, mode)
        MvNormal(mode, Symmetric(inv(Symmetric(curvature))))
    end
    # Subtracting the largest log value avoids underflow for unnormalized targets.
    weights = exp.(logvalues .- maximum(logvalues))
    weights ./= sum(weights)
    return JointDistribution(MixtureModel(components, weights), variables)
end
