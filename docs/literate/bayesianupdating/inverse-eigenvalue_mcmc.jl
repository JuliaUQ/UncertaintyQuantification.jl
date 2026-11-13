#===
# Bayesian Updating

## Inverse eigenvalue problem

The inverse eigenvalue problem is a classic engineering example. Here we will use Bayesian updating to sample from a bivariate posterior distribution describing unknown quantities of a matrix

```math
\begin{bmatrix}
 \theta_1 + \theta_2 & -\theta_2 \\
 -\theta_2 & \theta_2 \\
\end{bmatrix}
```

A matrix of this form can represent different problems, like the stiffness matrix describing a tuned mass damper system. In this example we assume the fixed values ``\theta_1 = 0.5`` and ``\theta_2 = 1.5`` for the variables.

The eigenvalues ``\lambda_1`` and ``\lambda_2`` of this matrix represent a physical measurable property corrupted by "noise" created for example due to environmental factors or measurement inaccuracy.

```math
\lambda_1^{noisy} = \frac{(\theta_1+2\theta_2)+\sqrt{\theta_1^2+4{\theta_2}^2}}{2} + \epsilon_1
```

```math
\lambda_2^{noisy} = \frac{(\theta_1+2\theta_2)-\sqrt{\theta_1^2+4{\theta_2}^2}}{2} + \epsilon_2
```

The "noise" terms ``\epsilon_1`` and ``\epsilon_2`` follow a Normal distribution with zero mean and standard deviations ``\sigma_1 = 1.0`` and ``\sigma_2 = 0.1``.

 The synthetic "noisy" data used for the Bayesian updating procedure is given in the following table.

|``\lambda_1``|``\lambda_2``|
|-----------|-----------|
|  1.51     |    0.33   |
|  4.01     |    0.30   |
|  3.16     |    0.17   |
|  3.21     |    0.18   |
|  2.19     |    0.32   |
|  1.71     |    0.23   |
|  2.73     |    0.21   |
|  5.51     |    0.20   |
|  1.95     |    0.11   |
|  4.48     |    0.20   |
|  1.43     |    0.16   |
|  2.91     |    0.26   |
|  3.91     |    0.23   |
|  3.58     |    0.25   |
|  2.62     |    0.25   |

The a priori knowledge of ``\theta_1`` and ``\theta_2`` is that they take values between 0.01 and 4. The likelihood function used for this problem is a bivariate Gaussian function with a covariance matrix ``\begin{bmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{bmatrix}``, with off-diagonal terms equal to 0 and the diagonal terms corresponding to the variances of the respective noise terms.

```math
P(\lambda|\theta) \propto \exp \left[-\frac{1}{2}\sum_{i=1}^2\sum_{n=1}^{15} {\left(\frac{\lambda_{i,n}^{data}-\lambda_i^{model}}{\sigma_i}\right)}^2\right]
```

To begin the Bayesian model updating procedure we start by defining the data, the models for the eigenvalues (without the noise term) and the likelihood function.

===#

#md using UncertaintyQuantification # hide
#md using Plots # hide
#md using DataFrames # hide
#md using LinearAlgebra #hide

#jl using UncertaintyQuantification
#jl using Plots
#jl using DataFrames
#jl using LinearAlgebra

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
#md nothing # hide

# For solving the Bayesian inverse problem, we use the MCMC methods implemented in the package.
# We setup the problem by defining the prior distributions using the marginal [`RandomVariable`](@ref).
# Afterwards, we define the setup of the MCMC methods.
prior = RandomVariable.(Uniform(0.01, 4), [:θ1, :θ2])

# ### Metropolis Hastings
# [`MetropolisHastings`](@ref) implements the Metropolis-Hastings algorithm with multivariate
# proposal. Here, we set the proposal to a multivariate Normal with zero mean and diagonal
# covariance matrix with ``\sigma = 0.1`` on the diagonal ``\mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma})``.
# We use two starting points of the chains ``x_0`` with a chain length of each `n = 1000`
# and `burnin = 1000`.
x0 = DataFrame(θ1 = [1.0, 3.0], θ2 = [1.0, 0.5])
Σ = 0.1 * I(2)
proposal = MvNormal([0, 0], Σ)
n, burnin = 1000, 1000
mh = MetropolisHastings(proposal, x0, n, burnin)
#md nothing #hide

# With the prior, likelihood, MCMC sampler and models defined, the last step is to call the
# [`bayesianupdating`](@ref) method to get the samples:

samples_mh, _ = bayesianupdating(prior, likelihood, mh, [λ1, λ2])
scatter(samples_mh.θ1, samples_mh.θ2; lim = [0, 4], label = "MH (σ=0.1)", xlabel = "θ1", ylabel = "θ2")
#md savefig("stiffness-samples-mh.svg"); nothing # hide

# ![Resulting MH](stiffness-samples-mh.svg)

# For comparison, we can run the same updating with a different proposal variance:
proposal_2 = MvNormal([0, 0], 2 * I(2))
mh_2 = MetropolisHastings(proposal_2, x0, n, burnin)

samples_mh_2, _ = bayesianupdating(prior, likelihood, mh_2, [λ1, λ2])
scatter(samples_mh_2.θ1, samples_mh_2.θ2; lim = [0, 4], label = "MH (σ=2)", xlabel = "θ1", ylabel = "θ2")
#md savefig("stiffness-samples-mh-2.svg"); nothing # hide

# ![Resulting MH](stiffness-samples-mh-2.svg)

# We see that the results are very sensitive to the chosen proposal covariance.


# ### Adaptive Metropolis Hastings
# [`AdaptiveMetropolisHastings`](@ref) implements an adaptation of the proposal covariance
# of a multivariate Gaussian, following [haarioAdaptiveMetropolisAlgorithm2001](@cite).
amh = AdaptiveMetropolisHastings(proposal, x0, n, burnin)

samples_amh, _ = bayesianupdating(prior, likelihood, amh, [λ1, λ2])
scatter(samples_amh.θ1, samples_amh.θ2; lim = [0, 4], label = "Adaptive MH (σ=0.1)", xlabel = "θ1", ylabel = "θ2")
#md savefig("stiffness-samples-amh.svg"); nothing # hide

# ![Resulting AMH](stiffness-samples-amh.svg)

# We can also investigate the results with a different proposal variance, e.g., with
# ``\sigma = 100``.
amh_2 = AdaptiveMetropolisHastings(proposal_2, x0, n, burnin)
samples_amh_2, _ = bayesianupdating(prior, likelihood, amh_2, [λ1, λ2])
scatter(samples_amh_2.θ1, samples_amh_2.θ2; lim = [0, 4], label = "Adaptive MH (σ=2)", xlabel = "θ1", ylabel = "θ2")
#md savefig("stiffness-samples-amh-new.svg"); nothing # hide

# ![Resulting AMH](stiffness-samples-amh-new.svg)

# We see that the results are less sensitive to the initially chosen values due to the adaptive refinement.

# ### Transitional Markov Chain Monte Carlo
# Finally, we use the [`TransitionalMarkovChainMonteCarlo`](@ref) algorithm proposed by
# [chingTransitionalMarkovChain2007](@cite). We set `n = 1000` and `burnin = 0`.
n = 1000
burnin = 0
tmcmc = TransitionalMarkovChainMonteCarlo(n, burnin)

samples, _ = bayesianupdating(prior, likelihood, tmcmc, [λ1, λ2])

scatter(samples.θ1, samples.θ2; lim = [0, 4], label = "TMCMC", xlabel = "θ1", ylabel = "θ2")
#md savefig("stiffness-samples.svg"); nothing # hide

# ![Resulting TMCMC](stiffness-samples.svg)
#  A scatter plot of the resulting samples shows convergence to two distinct regions. Unlike the transitional Markov Chain Monte Carlo algorithm, the standard Metropolis-Hastings algorithm would have only identified one of the two regions.
