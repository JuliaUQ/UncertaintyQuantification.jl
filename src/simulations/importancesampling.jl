mutable struct ImportanceSampling <: AbstractSimulation
    n::Integer
    β::Real
    dp::NamedTuple
    α::NamedTuple
    c::Real
    function ImportanceSampling(n, β, dp, α, c = 2.0)
        @assert n > 0 "n must be greater than zero"
        return new(n, β, dp, α, c)
    end
end

function ImportanceSampling(n::Integer, c = 2.0)
    return ImportanceSampling(n, 0.0, NamedTuple(), NamedTuple(), c)
end

function sample(inputs::Vector{<:UQInput}, sim::ImportanceSampling)
    β = sim.β
    dp = DataFrame([sim.dp])
    to_standard_normal_space!(inputs, dp)
    α = -1 * collect(sim.α)

    random_names = names(filter(i -> isa(i, RandomUQInput), inputs))
    deterministic_names = names(filter(i -> isa(i, DeterministicUQInput), inputs))

    # generate [n x m] samples in SNS
    Z = randn(sim.n, length(random_names))

    # samples perpendicular to important direction
    Z -= (Z * α) * α'

    # force in direction parallel to important direction
    b = exp(-(β^2 / 2)) / (cdf(Normal(), -β) * sqrt(2 * π)) # mean
    v = sim.c * (b - β) # std
    Zforced = randn(sim.n) .* v .+ b

    Z += Zforced * α'

    weights = pdf.(Normal(), Zforced) ./ pdf.(Normal(b, v), Zforced)
    samples = DataFrame(Z, random_names)
    to_physical_space!(filter(i -> isa(i, RandomUQInput), inputs), samples)

    if !isempty(deterministic_names)
        deterministic_samples = sample(
            filter(i -> isa(i, DeterministicUQInput), inputs), sim.n
        )
        DataFrames.hcat!(samples, deterministic_samples)
    end

    return samples, weights
end
