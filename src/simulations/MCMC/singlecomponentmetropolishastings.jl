"""
    SingleComponentMetropolisHastings(proposal, x0::DataFrame, n, burnin, islog = true)

Passed to [`bayesianupdating`](@ref) to run the single-component Metropolis-Hastings algorithm starting from `x0` with univariate proposal distibution `proposal` (or vector of proposal distributions per dimension). Will generate `n` samples *after* performing `burnin` steps of the Markov chain and discarding the samples. The flag `islog` specifies whether the likelihood passed to [`bayesianupdating`](@ref) returns logarithms. The prior is passed separately as a distribution.

Alternative constructor

```julia
    SingleComponentMetropolisHastings(proposal, x0, n, burnin)  # `islog` = true
```
`x0` is a DataFrame with one starting point per row and one parameter per column.
Each row starts an independent chain with `n` retained samples. Results are
stacked in starting-row order, with `n` consecutive rows per chain. The rejection
fraction is aggregated across chains. Targets receive one-row DataFrames.
"""
struct SingleComponentMetropolisHastings <: AbstractMarkovChainMonteCarlo
    proposal::Vector{<:UnivariateDistribution}
    x0::DataFrame
    n::Int
    burnin::Int
    islog::Bool

    function SingleComponentMetropolisHastings(
            proposal::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}},
            x0::DataFrame,
            n::Int,
            burnin::Int,
            islog::Bool = true,
        )
        nrow(x0) > 0 || throw(ArgumentError("x0 must contain at least one starting point."))
        burnin >= 0 || throw(ArgumentError("burnin must be nonnegative."))
        if n <= 0
            error("Number of samples `n` must be positive")
        end

        if !isa(proposal, Vector)
            proposal = fill(proposal, ncol(x0))
        else
            @assert ncol(x0) == length(proposal) "Number of proposal distribution must match the dimension."
        end
        return new(proposal, x0, n, burnin, islog)
    end
end

function _sample_chain(target::Function, sim::SingleComponentMetropolisHastings, start::DataFrame; islog::Bool = sim.islog)
    number_of_dimensions = ncol(start)

    # Starting columns may contain integers, but random-walk states are continuous.
    samples = DataFrame(Matrix{Float64}(start), names(start))
    inputvariables = names(start)

    logtarget = x -> begin
        value = only(target(copy(x)))
        islog ? value : log(value)
    end
    current = copy(samples)
    current_logdensity = logtarget(current)
    isfinite(current_logdensity) || throw(ArgumentError("Initial state must have finite log density."))
    rejection = 0
    total = sim.n + sim.burnin
    for i in 2:total
        for d in 1:number_of_dimensions
            candidate = copy(current)
            step = rand(sim.proposal[d])
            candidate[1, d] += step
            proposed_logdensity = logtarget(candidate)
            # Include the Hastings correction for asymmetric increments.
            α = min(
                0.0, proposed_logdensity - current_logdensity +
                    logpdf(sim.proposal[d], -step) - logpdf(sim.proposal[d], step)
            )
            if log(rand()) < α
                current = candidate
                current_logdensity = proposed_logdensity
            else
                rejection += 1
            end
        end
        push!(samples, current[1, inputvariables])
    end
    return samples[(sim.burnin + 1):end, :],
        rejection / max((total - 1) * number_of_dimensions, 1)
end
