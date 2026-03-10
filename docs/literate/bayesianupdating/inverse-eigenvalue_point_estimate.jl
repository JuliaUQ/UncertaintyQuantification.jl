#===

## Inverse eigenvalue problem with maximum likelihood and maximum a posteriori point estimation

The inverse eigenvalue problem can also be solved with point estimation schemes, i.e. maximum likelihood estimate (MLE) and maximum a posteriori (MAP) estimate. Both find the maximum of either only the likelihood (MLE) or the posterior (MAP) using optimization. The main difference in both is that MLE does not use the prior information, it will only give an estimate of the most likely parameter values based on the measurements. MAP on the other hand takes into account the prior distribution and gives a weighted estimate of the most likely parameters. MAP thus can also be seen as regularization of MLE.

We will set up the problem the same way as in the MCMC example.

===#

#md using UncertaintyQuantification # hide
#md using Plots # hide

#jl using UncertaintyQuantification
#jl using Plots

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

# For MLE and MAP we need to define the prior, however note that in MLE the defined [`RandomVariable`](@ref) is only used to inform the updating process of which paramters to update. The distribution will not affect the updating, as only the likelihood is taken into account. For the optimization we also need to define starting point(s). Since we know that the problem is multi-modal, we can define multiple starting points to find both modes. We also have to specify the optimization procedure, in this case we will use LBFGS. To illustrate the results of MAP and MLE, we will also solve the problem with TMCMC.

prior = RandomVariable.(Uniform(.1, 10), [:θ1, :θ2])

burnin = 0
n = 1000

x0 = [[1., 1.],[3.,.5]]

tmcmc = TransitionalMarkovChainMonteCarlo(prior, n, burnin)
MAP = MaximumAPosterioriBayesian(prior, "LBFGS", x0)
MLE = MaximumLikelihoodBayesian(prior, "LBFGS", x0)

# With the prior, likelihood, models and  MCMC sampler defined, the last step is to call the [`bayesianupdating`](@ref) method.

samples, evidence = bayesianupdating(likelihood, [λ1, λ2], tmcmc)
MapEstimate = bayesianupdating(likelihood, [λ1, λ2], MAP)
MLEstimate = bayesianupdating(likelihood, [λ1, λ2], MLE)

scatter(samples.θ1, samples.θ2; lim=[0, 4], label="TMCMC", xlabel="θ1", ylabel="θ2")
scatter!((MapEstimate.θ1, MapEstimate.θ2), label="MAP")
scatter!((MLEstimate.θ1, MLEstimate.θ2), label="MLE")
#md savefig("stiffness-point-estimate-uniform.svg"); nothing # hide

# ![Resulting point estimates](stiffness-point-estimate-uniform.svg)
#  A scatter plot of the resulting samples shows convergence to two distinct regions. Since we used a uniform prior distribution, ML and MAP estimates find the same estimates. With a different prior distribution, i.e. a standard normal centered on one of the modes, we obtain a different result:

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
#md savefig("stiffness-point-estimate-normal.svg"); nothing # hide

# ![Resulting point estimates](stiffness-point-estimate-normal.svg)
# Some things to note: Results from MLE are the same as before, since the prior distribution is not taken into account. MCMC does not find the second mode, since it is much less likely than the first one, so the Markov chains do not converge there. MAP does find the mode since it uses optimization and therefore is able to find the local maximum. A look at the relative values between both modes show the differences in probability:

println(exp.(MLEstimate[!,:maxval]))
println(exp.(MapEstimate[!,:maxval]))

# The second mode is 4 orders of magnitude less probable than the first mode, which explains why the Markov chains do not converge there.
