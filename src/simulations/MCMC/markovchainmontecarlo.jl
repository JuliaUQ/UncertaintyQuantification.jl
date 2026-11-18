# Methods for general MCMC

"""
    sample(
        target, method::AbstractMarkovChainMonteCarlo; islog = method.islog
    )

Sample a density with one chain per starting row in `method.x0`. The target
accepts a one-row DataFrame and returns a one-element vector. `islog` specifies
whether the target returns log densities, defaulting to the method's setting.
Returns `(samples, rejection_fraction)` with `method.n` samples per chain.
"""
function sample(
        target::Function,
        sim::AbstractMarkovChainMonteCarlo;
        islog::Bool = sim.islog,
    )
    chains = [_sample_chain(target, sim, sim.x0[i:i, :]; islog = islog) for i in 1:nrow(sim.x0)]
    return reduce(vcat, first.(chains)), mean(last.(chains))
end

function Base.show(
        io::IO,
        sim::AbstractMarkovChainMonteCarlo,
    )
    print(io, nameof(typeof(sim)), "(")
    print(io, "n=", sim.n, ", burnin=", sim.burnin)
    print(io, ", chains=", nrow(sim.x0), ", parameters=", ncol(sim.x0))
    print(io, ", islog=", sim.islog, ")")
    return nothing
end

function Base.show(
        io::IO, ::MIME"text/plain",
        sim::AbstractMarkovChainMonteCarlo,
    )
    get(io, :compact, false) && return show(io, sim)

    println(io, nameof(typeof(sim)))
    println(io, "  Samples per chain: ", sim.n)
    println(io, "  Burn-in per chain: ", sim.burnin)
    println(io, "  Log density: ", sim.islog)
    if sim isa AdaptiveMetropolisHastings
        println(io, "  Covariance scaling (γ): ", sim.γ)
        println(io, "  Adaptation start: ", sim.adapt_start)
        println(io, "  Regularization (ε): ", sim.ε)
    end
    println(io, "Proposal: ", sim.proposal)
    print(io, "Starting points: ")
    for (i, name) in enumerate(names(sim.x0))
        i > 1 && print(io, ", ")
        print(io, name, " = ")
        show(io, sim.x0[!, name])
    end
    return nothing
end
