"""
    AdaptiveMetropolisHastings(
        proposal, x0::DataFrame, n, burnin, γ = 2.4^2 / length(proposal), islog = true;
        adapt_start = max(2, 2length(proposal)), ε = 1.0e-6
    )

Adaptive Gaussian random-walk Metropolis sampling. `proposal` must have zero mean;
its covariance is used for the first `adapt_start` transitions. Thereafter the
proposal covariance is `γ * (empirical_covariance + ε*I)`, updated after every
transition using all states, including rejected moves and burn-in states.
Adaptation continues after burn-in. Returns `(samples, rejection_fraction)` with
`n` retained states per chain, after discarding `burnin` states per chain; the initial state counts as a
sample when `burnin == 0`. The target accepts a one-row DataFrame and returns a
one-element vector of log densities (or densities when `islog=false`).

`x0` is a DataFrame with one starting point per row and one parameter per column.
Each adaptive chain maintains its own covariance history.
Each row starts an independent chain with `n` retained samples. Results are
stacked in starting-row order, with `n` consecutive rows per chain. The rejection
fraction is aggregated across chains. Targets receive one-row DataFrames.

# References

[haarioAdaptiveMetropolisAlgorithm2001](@cite)
"""
struct AdaptiveMetropolisHastings <: AbstractMarkovChainMonteCarlo
    proposal::MvNormal
    x0::DataFrame
    n::Int
    burnin::Int
    γ::Float64
    islog::Bool
    adapt_start::Int
    ε::Float64

    function AdaptiveMetropolisHastings(
            proposal::MvNormal, x0::DataFrame, n::Int, burnin::Int,
            γ::Real = 2.4^2 / length(proposal), islog::Bool = true;
            adapt_start::Int = max(2, 2length(proposal)), ε::Real = 1.0e-6,
        )
        nrow(x0) > 0 || throw(ArgumentError("x0 must contain at least one starting point."))
        n > 0 || throw(ArgumentError("Number of samples `n` must be positive."))
        burnin >= 0 || throw(ArgumentError("`burnin` must be nonnegative."))
        ncol(x0) == length(proposal) || throw(ArgumentError("Proposal dimension must match x0."))
        all(iszero, mean(proposal)) || throw(ArgumentError("Proposal must have zero mean."))
        isfinite(γ) && γ > 0 || throw(ArgumentError("γ must be finite and positive."))
        isfinite(ε) && ε > 0 || throw(ArgumentError("ε must be finite and positive."))
        adapt_start >= 2 || throw(ArgumentError("adapt_start must be at least two."))
        return new(proposal, x0, n, burnin, γ, islog, adapt_start, ε)
    end
end

function _sample_chain(target::Function, sim::AdaptiveMetropolisHastings, start::DataFrame; islog::Bool = sim.islog)
    inputvariables = names(start)
    current = Float64.(collect(start[1, :]))
    d = length(current)
    # Use fresh frames because targets may add model-output columns in place.
    logtarget = x -> begin
        value = only(target(DataFrame(reshape(copy(x), 1, d), inputvariables)))
        islog ? value : log(value)
    end
    current_logdensity = logtarget(current)
    isfinite(current_logdensity) || throw(ArgumentError("Initial state must have finite log density."))

    total = sim.n + sim.burnin
    states = Matrix{Float64}(undef, total, d)
    states[1, :] = current
    running_mean = copy(current)
    scatter = zeros(d, d)
    proposal = sim.proposal
    rejected = 0

    for i in 2:total
        candidate = current + rand(proposal)
        candidate_logdensity = logtarget(candidate)
        if log(rand()) < min(0.0, candidate_logdensity - current_logdensity)
            current = candidate
            current_logdensity = candidate_logdensity
        else
            rejected += 1
        end
        states[i, :] = current

        # Welford's update includes repeated states following rejection.
        δ = current - running_mean
        running_mean += δ / i
        scatter += δ * (current - running_mean)'
        # This covariance is used only for the NEXT proposal.
        if i - 1 >= sim.adapt_start && i < total
            Σ = sim.γ * (Symmetric(scatter / (i - 1)) + sim.ε * I)
            proposal = MvNormal(zeros(d), Symmetric(Σ))
        end
    end

    samples = DataFrame(states[(sim.burnin + 1):end, :], inputvariables)
    return samples, rejected / max(total - 1, 1)
end
