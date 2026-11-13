using UncertaintyQuantification
using Plots
using DataFrames
using LinearAlgebra

Y = [
    1.51 0.33
    4.01 0.3
    3.16 0.27
    3.21 0.18
    2.19 0.33
    1.71 0.23
    2.73 0.21
    5.51 0.2
    1.95 0.11
    4.48 0.2
    1.43 0.16
    2.91 0.26
    3.81 0.23
    3.58 0.25
    2.62 0.25
]

λ1 = @. Model(df -> ((df.θ1 + 2 * df.θ2) + sqrt(df.θ1^2 + 4 * df.θ2^2)) / 2, :λ1)
λ2 = @. Model(df -> ((df.θ1 + 2 * df.θ2) - sqrt(df.θ1^2 + 4 * df.θ2^2)) / 2, :λ2)

σ = [1.0 0.1]
function likelihood(df)
    λ = [df.λ1 df.λ2]

    return log.(exp.([-0.5 * sum(((Y .- λ[n, :]') ./ σ) .^ 2) for n in axes(λ, 1)]))
end

prior = RandomVariable.(Uniform(0.01, 4), [:θ1, :θ2])

x0 = DataFrame(θ1 = [1.0, 3.0], θ2 = [1.0, 0.5])
Σ = 0.1 * I(2)
proposal = MvNormal([0, 0], Σ)
n, burnin = 1000, 1000
mh = MetropolisHastings(proposal, x0, n, burnin)

samples_mh, _ = bayesianupdating(prior, likelihood, mh, [λ1, λ2])
scatter(samples_mh.θ1, samples_mh.θ2; lim = [0, 4], label = "MH (σ=0.1)", xlabel = "θ1", ylabel = "θ2")

proposal_2 = MvNormal([0, 0], 2 * I(2))
mh_2 = MetropolisHastings(proposal_2, x0, n, burnin)

samples_mh_2, _ = bayesianupdating(prior, likelihood, mh_2, [λ1, λ2])
scatter(samples_mh_2.θ1, samples_mh_2.θ2; lim = [0, 4], label = "MH (σ=2)", xlabel = "θ1", ylabel = "θ2")

amh = AdaptiveMetropolisHastings(proposal, x0, n, burnin)

samples_amh, _ = bayesianupdating(prior, likelihood, amh, [λ1, λ2])
scatter(samples_amh.θ1, samples_amh.θ2; lim = [0, 4], label = "Adaptive MH (σ=0.1)", xlabel = "θ1", ylabel = "θ2")

amh_2 = AdaptiveMetropolisHastings(proposal_2, x0, n, burnin)
samples_amh_2, _ = bayesianupdating(prior, likelihood, amh_2, [λ1, λ2])
scatter(samples_amh_2.θ1, samples_amh_2.θ2; lim = [0, 4], label = "Adaptive MH (σ=2)", xlabel = "θ1", ylabel = "θ2")

n = 1000
burnin = 0
tmcmc = TransitionalMarkovChainMonteCarlo(n, burnin)

samples, _ = bayesianupdating(prior, likelihood, tmcmc, [λ1, λ2])

scatter(samples.θ1, samples.θ2; lim = [0, 4], label = "TMCMC", xlabel = "θ1", ylabel = "θ2")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
