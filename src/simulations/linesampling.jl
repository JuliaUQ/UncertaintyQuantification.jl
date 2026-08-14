mutable struct LineSampling <: AbstractSimulation
    lines::Integer
    points::Vector{<:Real}
    direction::NamedTuple

    function LineSampling(
            lines::Integer,
            points::Vector{<:Real} = collect(0.5:0.5:5),
            direction::NamedTuple = NamedTuple(),
        )
        if maximum(points) > 8.12
            error("LineSampling does not support lines longer than 8.12.")
        end
        return new(lines, points, direction)
    end
end

function sample(inputs::Vector{<:UQInput}, sim::LineSampling)
    random_inputs = filter(i -> isa(i, RandomUQInput), inputs)
    deterministic_inputs = filter(i -> isa(i, DeterministicUQInput), inputs)

    n_rv = count_rvs(random_inputs)
    n_samples = length(sim.points) * sim.lines
    rv_names = names(random_inputs)

    α = map(n -> sim.direction[n], rv_names)
    normalize!(α)

    θ = rand(Normal(), n_rv, sim.lines)

    θ = θ - α * (α' * θ)
    θ = repeat(θ; outer = [length(sim.points), 1])

    θ = θ[:] + repeat(α * sim.points'; outer = [1, sim.lines])[:]

    samples = transpose(reshape(θ, n_rv, n_samples))
    samples = DataFrame(rv_names .=> eachcol(samples))

    if !isempty(deterministic_inputs)
        DataFrames.hcat!(samples, sample(deterministic_inputs, n_samples))
    end

    to_physical_space!(inputs, samples)

    return samples
end

mutable struct AdvancedLineSampling <: AbstractSimulation
    lines::Integer
    points::Vector{<:Real}
    direction::NamedTuple
    tolerance::Float64
    stepsize::Float64
    maxiterations::Int64

    function AdvancedLineSampling(
            lines::Integer,
            points::Vector{<:Real} = collect(0.5:0.5:5),
            direction::NamedTuple = NamedTuple(),
            tolerance::Float64 = 1.0e-3,
            stepsize::Float64 = 1.0e-5,
            maxiterations::Int64 = 20,
        )
        return new(lines, points, direction, tolerance, stepsize, maxiterations)
    end
end

function AdvancedLineSampling(
        lines::Integer, tolerance::Float64, stepsize::Float64, maxiterations::Int64
    )
    return AdvancedLineSampling(
        lines, collect(0.5:0.5:5), NamedTuple(), tolerance, stepsize, maxiterations
    )
end

function rootinterpolation(x::Vector{<:Real}, y::Vector{<:Real}, i::Integer = 0)
    if all(y[:] .<= 0)
        @warn "All samples for line $i are inside the failure domain"
        return Inf
    elseif all(y[:] .> 0)
        @warn "All samples for line $i are outside the failure domain"
        return -Inf
    end
    spl = Spline1D(x, y[:])
    try
        return Dierckx.roots(spl)[1]
    catch e
        @warn "Intersection with failure domain not found for line $i ($e)"
        return Inf
    end
end

function _grad1D(x::Float64, f::Function, stepsize::Float64)
    fₓ = only(f(x))
    fₓ₊ = only(f(x + stepsize))
    ∇fₓ = (fₓ₊ - fₓ) / stepsize
    return ∇fₓ, fₓ, fₓ₊
end

function newtonraphson(
        x₀::Float64, f::Function, stepsize::Float64, tolerance::Float64, maxiterations::Integer
    )
    # Initialize
    err = Inf
    steps = 0

    while abs(err) > tolerance
        # One dimensional Netwon's method
        ∇fₓ, fₓ, _ = _grad1D(x₀, f, stepsize)
        x₁ = x₀ - fₓ / ∇fₓ

        # Calculate error
        err = norm(x₁ - x₀) / x₀
        x₀ = x₁

        # Counter
        steps += 1

        if steps >= maxiterations
            @warn "No root found after $steps iterations."
            return Inf
        end
    end

    return x₀
end
