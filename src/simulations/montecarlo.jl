struct MonteCarlo <: AbstractMonteCarlo
    n::Integer
    MonteCarlo(n) = n > 0 ? new(n) : error("n must be greater than zero")
end

struct SobolSampling <: AbstractQuasiMonteCarlo
    n::Integer
    randomization::Symbol

    function SobolSampling(n::Integer, randomization::Symbol=:matousek)
        randomization ∉ [:matousek, :owen, :none] &&
            error("type must be :matousek :owen or :none")
        if n > 0
            if !isinteger(log2(n))
                n = Int(2^ceil(log2(n)))
                @warn("n must be a power of 2, automatically increased to $n")
            end
            return new(n, randomization)
        else
            error("n must be greater than zero")
        end
    end
end

struct FaureSampling <: AbstractQuasiMonteCarlo
    n::Integer
    randomization::Symbol

    function FaureSampling(n::Integer, randomization::Symbol=:matousek)
        randomization ∉ [:matousek, :owen, :none] &&
            error("type must be :matousek, :owen or :none")
        if n > 0
            return new(n, randomization)
        else
            error("n must be greater than zero")
        end
    end
end

struct HaltonSampling <: AbstractQuasiMonteCarlo
    n::Integer
    randomization::Symbol

    function HaltonSampling(n::Integer, randomization::Symbol=:none)
        randomization ∉ [:none] && error("type must be :none")
        if n > 0
            return new(n, randomization)
        else
            error("n must be greater than zero")
        end
    end
end

struct LatinHypercubeSampling <: AbstractQuasiMonteCarlo
    n::Integer
    LatinHypercubeSampling(n) = n > 0 ? new(n) : error("n must be greater than zero")
end

struct LatticeRuleSampling <: AbstractQuasiMonteCarlo
    n::Integer
    randomization::Symbol

    function LatticeRuleSampling(n::Integer, randomization::Symbol=:shift)
        randomization ∉ [:shift, :none] && error("type must be :shift or :none")
        if n > 0
            return new(n, randomization)
        else
            error("n must be greater than zero")
        end
    end
end

function sample(inputs::Vector{<:UQInput}, sim::MonteCarlo)
    return sample(inputs, sim.n)
end

"""
    sample(inputs::Vector{UQInput}, sim::AbstractQuasiMonteCarlo; intervals::Bool=true, n_internal::Integer=sim.n * 10)

    Generate Quasi-Monte Carlo samples of the `inputs` using the QMC sampling method `sim`. By default any [`IntervalVariable`](@ref) or [`'JointInterval`]('ref) will be included as the original intervals. To apply the QMC sampling also to the intervals pass the keyword `;intervals=true`. For [`JointInterval`]('ref)s this will internally sample 10 times the desired samples and discard the samples outside the permissible set and any excess samples. If the permissible set is small more samples might be required to generated sufficient samples. In this case, the keyword `n_internal` can be used to increase the number of samples used.
"""
function sample(
    inputs::Vector{<:UQInput},
    sim::AbstractQuasiMonteCarlo;
    intervals::Bool=true,
    n_internal::Integer=sim.n * 10,
)
    rvs = filter(i -> isa(i, RandomUQInput), inputs)
    ivs = filter(i -> isa(i, IntervalVariable) || isa(i, JointInterval), inputs)
    parameters = filter(i -> isa(i, Parameter), inputs)

    dependent = any(isa.(ivs, JointInterval))
    # verify randomized QMC for dependent intervals
    if dependent &
        !intervals &
        !(isa(sim, LatinHypercubeSampling) || isa(sim, RandomizedHaltonSample))
        if sim.randomization == :none
            error("QMC sampling must be randomized to be applied to joint intervals")
        end
    end

    n_rv = count_rvs(rvs)
    n_int = !isempty(ivs) ? mapreduce(dimensions, +, ivs) : 0

    u = @suppress_err begin
        # if intervals is true only rvs need to be sampled. If not we also obtain qmc samples for intervals
        # if dependent intervals are involved sample much more than requested
        qmc_samples(
            dependent & !intervals ? typeof(sim)(n_internal, sim.randomization) : sim,
            intervals ? n_rv : n_rv + n_int,
        )
    end

    samples = if intervals
        DataFrame(names(rvs) .=> eachrow(u))
    else
        DataFrame(vcat(names(rvs), names(ivs)) .=> eachrow(u))
    end

    # map rvs into standard normal space
    samples[:, names(rvs)] = quantile.(Normal(), samples[:, names(rvs)])

    if !isempty(ivs)
        if intervals
            # append intervals if not sampled
            DataFrames.hcat!(samples, sample(ivs, size(samples, 1)))
        else
            # translate qmc samples to interval ranges
            for i in mapreduce(i -> isa(i, IntervalVariable) ? i : i.intervals, vcat, ivs)
                samples[:, i.name] = samples[:, i.name] .* (i.ub - i.lb) .+ i.lb
            end
        end
    end

    # discard samples outside the permissible set
    if dependent & !intervals
        for ji in filter(i -> isa(i, JointInterval), ivs)
            idx = findall(.!in.(eachrow(Matrix(samples[:, names(ji)])), ji))
            deleteat!(samples, idx)
        end
        # discard excess samples
        if size(samples)[1] > sim.n
            deleteat!(samples, (sim.n + 1):size(samples)[1])
        else
            @warn "Only $(size(samples)[1]) of $(sim.n) samples generated. Try increasing 'n_internal'"
        end
    end

    # finally append any parameters
    if !isempty(parameters)
        DataFrames.hcat!(samples, sample(parameters, size(samples, 1)))
    end

    # map rvs to physical space before returning the samples
    to_physical_space!(inputs, samples)

    return samples
end

sample(input::UQInput, sim::AbstractMonteCarlo) = sample([input], sim)

function qmc_samples(sim::SobolSampling, rvs::Integer)
    return randomize(sim, QuasiMonteCarlo.sample(sim.n, rvs, SobolSample()))
end

function qmc_samples(sim::FaureSampling, rvs::Integer)
    b = nextprime(rvs)
    n = sim.n
    if !isinteger(log(b, sim.n))
        n = Int(b^ceil(log(b, sim.n)))
        @warn(
            "n must be a power of the base (here $b), automatically increased to $n for these samples."
        )
    end
    return randomize(sim, QuasiMonteCarlo.sample(n, rvs, FaureSample()), b)
end

function qmc_samples(sim::HaltonSampling, rvs::Integer)
    samples = QuasiMonteCarlo.sample(sim.n, rvs, HaltonSample())
    return randomize(sim, rvs > 1 ? samples : reshape(samples, 1, sim.n))
end

function qmc_samples(sim::LatinHypercubeSampling, rvs::Integer)
    return QuasiMonteCarlo.sample(sim.n, rvs, LatinHypercubeSample())
end

function qmc_samples(sim::LatticeRuleSampling, rvs::Integer)
    return randomize(sim, QuasiMonteCarlo.sample(sim.n, rvs, LatticeRuleSample()))
end

function randomize(sim::AbstractQuasiMonteCarlo, u::Matrix, b=2)
    if sim.randomization == :matousek
        u = QuasiMonteCarlo.randomize(u, MatousekScramble(; base=b))
    elseif sim.randomization == :owen
        u = QuasiMonteCarlo.randomize(u, OwenScramble(; base=b))
    elseif sim.randomization == :shift
        u = QuasiMonteCarlo.randomize(u, Shift())
    end

    return u
end

double_samples(sim::MonteCarlo) = MonteCarlo(2 * sim.n)
double_samples(sim::SobolSampling) = SobolSampling(2 * sim.n, sim.randomization)
double_samples(sim::FaureSampling) = FaureSampling(2 * sim.n, sim.randomization)
double_samples(sim::HaltonSampling) = HaltonSampling(2 * sim.n, sim.randomization)
double_samples(sim::LatinHypercubeSampling) = LatinHypercubeSampling(2 * sim.n)
double_samples(sim::LatticeRuleSampling) = LatticeRuleSampling(2 * sim.n, sim.randomization)
