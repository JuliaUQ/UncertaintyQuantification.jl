using UncertaintyQuantification
using Plots

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

prior = RandomVariable.(Uniform(.1, 10), [:θ1, :θ2])

burnin = 0
n = 1000

x0 = [[1., 1.],[3.,.5]]

tmcmc = TransitionalMarkovChainMonteCarlo(prior, n, burnin)
MAP = MaximumAPosterioriBayesian(prior, "LBFGS", x0)
MLE = MaximumLikelihoodBayesian(prior, "LBFGS", x0)

samples, evidence = bayesianupdating(likelihood, [λ1, λ2], tmcmc)
MapEstimate = bayesianupdating(likelihood, [λ1, λ2], MAP)
MLEstimate = bayesianupdating(likelihood, [λ1, λ2], MLE)

scatter(samples.θ1, samples.θ2; lim=[0, 4], label="TMCMC", xlabel="θ1", ylabel="θ2")
scatter!((MapEstimate.θ1, MapEstimate.θ2), label="MAP")
scatter!((MLEstimate.θ1, MLEstimate.θ2), label="MLE")

priorθ1 = RandomVariable(Normal(.5, .5), :θ1)
priorθ2 = RandomVariable(Normal(1.5, .5), :θ2)

prior = [priorθ1, priorθ2]

burnin = 0
n = 1000

x0 = [[1., 1.],[3.,.5]]

tmcmc = TransitionalMarkovChainMonteCarlo(prior, n, burnin)
MAP = MaximumAPosterioriBayesian(prior, "LBFGS", x0)
MLE = MaximumLikelihoodBayesian(prior, "LBFGS", x0)

samples, evidence = bayesianupdating(likelihood, [λ1, λ2], tmcmc)
MapEstimate = bayesianupdating(likelihood, [λ1, λ2], MAP)
MLEstimate = bayesianupdating(likelihood, [λ1, λ2], MLE)

scatter(samples.θ1, samples.θ2; lim=[0, 4], label="TMCMC", xlabel="θ1", ylabel="θ2")
scatter!((MapEstimate.θ1, MapEstimate.θ2), label="MAP")
scatter!((MLEstimate.θ1, MLEstimate.θ2), label="MLE")

println(exp.(MLEstimate[!,:maxval]))
println(exp.(MapEstimate[!,:maxval]))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
