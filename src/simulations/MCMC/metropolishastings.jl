"""
    MetropolisHastings(proposal::MvNormal, x0::DataFrame, n, burnin, islog = true)

Multivariate Gaussian random-walk Metropolis-Hastings with a fixed, zero-mean
increment distribution `proposal`. All components are proposed together and
accepted or rejected with a single decision.

`sample(target, sim)` returns `(samples, rejection_fraction)`. The target accepts
a one-row DataFrame and returns a one-element vector of log densities, or
densities when `islog=false`. It may add model-output columns to that frame.
The returned DataFrame contains only the parameters named in `x0` and exactly
`n` states per chain after discarding `burnin` states per chain. When `burnin == 0`, the initial
state is included. The rejection fraction covers all attempted transitions,
including burn-in, and is zero when no transitions are attempted.

`x0` is a DataFrame with one starting point per row and one parameter per column.
Each row starts an independent chain with `n` retained samples. Results are
stacked in starting-row order, with `n` consecutive rows per chain. The rejection
fraction is aggregated across chains. Targets receive one-row DataFrames.
"""
struct MetropolisHastings <: AbstractMarkovChainMonteCarlo
    proposal::MvNormal
    x0::DataFrame
    n::Int
    burnin::Int
    islog::Bool

    function MetropolisHastings(
            proposal::MvNormal, x0::DataFrame, n::Int, burnin::Int,
            islog::Bool = true,
        )
        nrow(x0) > 0 || throw(ArgumentError("x0 must contain at least one starting point."))
        n > 0 || throw(ArgumentError("Number of samples `n` must be positive."))
        burnin >= 0 || throw(ArgumentError("`burnin` must be nonnegative."))
        ncol(x0) == length(proposal) || throw(ArgumentError("Proposal dimension must match x0."))
        all(iszero, mean(proposal)) || throw(ArgumentError("Proposal must have zero mean."))
        return new(proposal, x0, n, burnin, islog)
    end
end

function _sample_chain(target::Function, sim::MetropolisHastings, start::DataFrame; islog::Bool = sim.islog)
    inputvariables = names(start)
    current = Float64.(collect(start[1, :]))
    d = length(current)
    # Evaluate on fresh frames so model-output columns do not enter the chain.
    logtarget = x -> begin
        value = only(target(DataFrame(reshape(copy(x), 1, d), inputvariables)))
        islog ? value : log(value)
    end
    current_logdensity = logtarget(current)
    isfinite(current_logdensity) || throw(ArgumentError("Initial state must have finite log density."))

    total = sim.n + sim.burnin
    states = Matrix{Float64}(undef, total, d)
    states[1, :] = current
    rejected = 0
    for i in 2:total
        candidate = current + rand(sim.proposal)
        candidate_logdensity = logtarget(candidate)
        # The symmetric proposal cancels in the Hastings ratio. Out-of-support
        # proposals are rejected rather than redrawn, preserving this symmetry.
        if log(rand()) < min(0.0, candidate_logdensity - current_logdensity)
            current = candidate
            current_logdensity = candidate_logdensity
        else
            rejected += 1
        end
        states[i, :] = current
    end

    return DataFrame(states[(sim.burnin + 1):end, :], inputvariables),
        rejected / max(total - 1, 1)
end

# Each chain has the same number of attempted transitions, so the mean of its
# rejection fractions is also the aggregate rejection fraction.
"""
    sample(
        target, method::Union{
            MetropolisHastings, AdaptiveMetropolisHastings,
            SingleComponentMetropolisHastings,
        }; islog = method.islog
    )

Sample a density with one chain per starting row in `method.x0`. The target
accepts a one-row DataFrame and returns a one-element vector. `islog` specifies
whether the target returns log densities, defaulting to the method's setting.
Returns `(samples, rejection_fraction)` with `method.n` samples per chain.
"""
function sample(
        target::Function,
        sim::Union{MetropolisHastings, AdaptiveMetropolisHastings, SingleComponentMetropolisHastings};
        islog::Bool = sim.islog,
    )
    chains = [_sample_chain(target, sim, sim.x0[i:i, :]; islog = islog) for i in 1:nrow(sim.x0)]
    return reduce(vcat, first.(chains)), mean(last.(chains))
end
