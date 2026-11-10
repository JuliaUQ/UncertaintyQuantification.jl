"""
    SingleComponentMetropolisHastings(proposal, x0, n, burnin, islog)

Passed to [`bayesianupdating`](@ref) to run the single-component Metropolis-Hastings algorithm starting from `x0` with univariate proposal distibution `proposal` (or vector of proposal distributions per dimension). Will generate `n` samples *after* performing `burnin` steps of the Markov chain and discarding the samples. The flag `islog` specifies whether the prior and likelihood functions passed to the [`bayesianupdating`](@ref) method are already given as logarithms.

Alternative constructor

```julia
    SingleComponentMetropolisHastings(proposal, x0, n, burnin)  # `islog` = true
```

"""
struct SingleComponentMetropolisHastings # !TODO: add type to inherit from
    proposal::Vector{<:UnivariateDistribution}
    x0::NamedTuple
    n::Int
    burnin::Int
    islog::Bool

    function SingleComponentMetropolisHastings(
            proposal::Union{UnivariateDistribution, Vector{<:UnivariateDistribution}},
            x0::NamedTuple,
            n::Int,
            burnin::Int,
            islog::Bool = true,
        )
        if n <= 0
            error("Number of samples `n` must be positive")
        end

        if !isa(proposal, Vector)
            proposal = fill(proposal, length(x0))
        else
            @assert length(x0) == length(proposal) "Number of proposal distribution must match the dimension."
        end
        return new(proposal, x0, n, burnin, islog)
    end
end

function sample(target::Function, sim::SingleComponentMetropolisHastings)
    number_of_dimensions = length(sim.x0)

    samples = DataFrame(collect(sim.x0)', collect(keys(sim.x0)))
    inputvariables = collect(keys(sim.x0))

    println("samples: ", samples)

    # is this necessary?
    logtarget = if sim.islog
        df -> target(df)
    else
        df -> log.(target(df))
    end

    rejection = 0.0

    for i in 1:(sim.n + sim.burnin - 1)
        current = DataFrame(samples[i, inputvariables])
        x = DataFrame(samples[i, inputvariables])

        for d in 1:number_of_dimensions
            x[1, d] += rand(sim.proposal[d])

            # safeguard for areas where the logprior is -Inf (prior = 0)
            while sim.islog ? isinf(target(x)[1]) : isinf(log.(target(x))[1])
                x[1, d] = current[1, d] + rand(sim.proposal[d])
            end

            α = min(0, logtarget(x)[1] - logtarget(current)[1])

            if α >= log(rand())
                current[1, d] = x[1, d]
            else
                x[1, d] = current[1, d]
                rejection += 1
            end
        end

        # !TODO: check if this is the best place to filter for input variables, or if it should be done in the likelihood function
        # Problem is that the likelihood function may use evaluate!, changing the DataFrame contents, but we only want to return the input variables in the samples
        push!(samples, x[1, inputvariables])
    end

    rejection /= ((sim.n + sim.burnin) * number_of_dimensions)

    # discard burnin samples during return
    return samples[(sim.burnin + 1):end, :], rejection
end
