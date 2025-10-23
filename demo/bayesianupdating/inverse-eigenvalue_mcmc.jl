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

prior = RandomVariable.(Uniform(0.01, 4), [:θ1, :θ2])

n = 1000
burnin = 0

x0 = [[1., 1.],[3.,.5],[2.,2.]]

tmcmc = TransitionalMarkovChainMonteCarlo(prior, n, burnin)

samples, evidence = bayesianupdating(likelihood, [λ1, λ2], tmcmc)

scatter(samples.θ1, samples.θ2; lim=[0, 4], label="TMCMC", xlabel="θ1", ylabel="θ2")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
