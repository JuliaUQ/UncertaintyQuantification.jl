struct MonteCarlo <: AbstractMonteCarlo
    n::Integer
    MonteCarlo(n) = n > 0 ? new(n) : error("n must be greater than zero")
end

struct QMC <: AbstractQuasiMonteCarlo
    n::Integer
    m::QuasiMonteCarlo.SamplingAlgorithm
    QMC(n, m) = n > 0 ? new(n, m) : error("n must be greater than zero")
end

function sample(inputs::Vector{<:UQInput}, sim::MonteCarlo)
    return sample(inputs, sim.n)
end

function sample(inputs::Vector{<:UQInput}, sim::QMC, T::Type = Float64)
    random_inputs = filter(i -> isa(i, RandomUQInput) || isa(i, ProbabilityBox), inputs)
    deterministic_inputs = filter(i -> isa(i, Parameter) || isa(i, Interval), inputs)

    n_rv = count_rvs(random_inputs)

    u = QuasiMonteCarlo.sample(sim.n, n_rv, sim.m, T)

    samples = quantile.(Normal(), u)
    samples = DataFrame(names(random_inputs) .=> eachrow(samples))

    if !isempty(deterministic_inputs)
        DataFrames.hcat!(samples, sample(deterministic_inputs, size(samples, 1)))
    end

    to_physical_space!(inputs, samples)

    return samples
end

double_samples(sim::MonteCarlo) = MonteCarlo(2 * sim.n)
double_samples(sim::QMC) = QMC(2 * sim.n, sim.m)
