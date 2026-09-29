struct MonteCarlo <: AbstractMonteCarlo
    n::Integer
    MonteCarlo(n) = n > 0 ? new(n) : error("n must be greater than zero")
end

"""
    QuasiMonteCarloSampling(n::Integer, m::QuasiMonteCarlo.SamplingAlgorithm)

A structure to facilitate Quasi Monte Carlo (QMC) sampling.

# Fields
- `n::Integer`: The number of samples to generate. Must be greater than 0.
- `m::QuasiMonteCarlo.SamplingAlgorithm`: The desired QMC sampling method from QuasiMonteCarlo.jl.

# Errors
Throws an error if `n ≤ 0`.

# Example

```julia
sobol = QuasiMonteCarlo.SobolSample()
qmc = QuasiMonteCarloSampling(1024, sobol)
```
"""
struct QuasiMonteCarloSampling <: AbstractMonteCarlo
    n::Integer
    m::QuasiMonteCarlo.SamplingAlgorithm
    QuasiMonteCarloSampling(n, m) = n > 0 ? new(n, m) : error("n must be greater than zero")
end

function sample(inputs::Vector{<:UQInput}, sim::MonteCarlo)
    return sample(inputs, sim.n)
end

function sample(inputs::Vector{<:UQInput}, sim::QuasiMonteCarloSampling, T::Type = Float64)
    random_inputs = filter(i -> isa(i, RandomUQInput) || isa(i, ProbabilityBox), inputs)
    deterministic_inputs = filter(i -> isa(i, Parameter) || isa(i, Interval), inputs)

    if isempty(random_inputs)
        return sample(inputs, sim.n)
    end
    n_rv = count_rvs(random_inputs)

    u = QuasiMonteCarlo.sample(sim.n, n_rv, sim.m, T)

    samples = quantile.(Normal(), u)
    samples = DataFrame(names(random_inputs) .=> eachrow(samples))

    if !isempty(deterministic_inputs)
        DataFrames.hcat!(samples, sample(deterministic_inputs, size(samples, 1)))
    end
    to_physical_space!(inputs, samples)

    DataFrames.select!(samples, names(inputs))

    return samples
end


sample(input::UQInput, sim::AbstractMonteCarlo) =
    sample([input], sim)

sample(input::UQInput, sim::QuasiMonteCarloSampling, T::Type) =
    sample([input], sim, T)

double_samples(sim::MonteCarlo) = MonteCarlo(2 * sim.n)
double_samples(sim::QuasiMonteCarloSampling) = QuasiMonteCarloSampling(2 * sim.n, sim.m)
