
using UncertaintyQuantification
using Plots
using DataFrames

# # Bayesian Updating with Transport Maps
#
# This example demonstrates Bayesian updating using transport maps to approximate
# the posterior distribution. We'll compare the results with Transitional Markov
# Chain Monte Carlo (TMCMC) sampling.

# ## Define the Prior Distribution
#
# We define a bivariate prior distribution with independent standard normal components.

prior = [
    RandomVariable(Normal(), :θ1),
    RandomVariable(Normal(), :θ2)
]

# ## Forward Model
#
# The forward model maps the parameters θ₁ and θ₂ to observations at different times.
# Parameters A and B are transformed from the standard normal space using the CDF.

function forward_model(t, df)
    A = 0.4 .+ (1.2 - 0.4) * cdf(Normal(), df.θ1)
    B = 0.01 .+ (0.31 - 0.01) * cdf(Normal(), df.θ2)
    return A .* (1 .- exp.(-B * t))
end

# ## Observed Data
#
# Define the observation times and corresponding measurements with noise standard deviation.

t = [1, 2, 3, 4, 5]
D = [0.18, 0.32, 0.42, 0.49, 0.54]
σ = sqrt(1e-3)

# ## Likelihood Function
#
# The likelihood assumes normally distributed errors around the forward model predictions.

function loglikelihood(df)
    return sum([logpdf.(Normal.(forward_model(t[k], df), σ), D[k]) for k in 1:5])
end

# Wrap the likelihood in a Model
Likelihood = Model(df -> loglikelihood(df), :L)

# ## Bayesian Updating with TMCMC
#
# First, we perform Bayesian updating using Transitional Markov Chain Monte Carlo
# as a reference solution.

tmcmc = TransitionalMarkovChainMonteCarlo(prior, 1_000, 0)
tmcmc_samples, evidence = bayesianupdating(df -> df.L, [Likelihood], tmcmc)

# ## Bayesian Updating with Transport Maps
#
# Now we use transport maps to approximate the posterior. We construct a polynomial
# transport map with order 3 and optimize it using Gauss-Hermite quadrature.

tm = PolynomialMap(2, 3, Normal(), Softplus(), LinearizedHermiteBasis())
quadrature = GaussHermiteWeights(10, 2)
tm = bayesianupdating(prior, df -> df.L, Likelihood, tm, quadrature)

# ## Compare Results
#
# Sample from the transport map and compare with TMCMC samples.

tm_samples = sample(tm, 1000)

p = scatter(tm_samples.θ1, tm_samples.θ2, label="Transport Map", alpha=0.8)
scatter!(tmcmc_samples.θ1, tmcmc_samples.θ2, label="TMCMC", alpha=0.8)
#md savefig("samples-tm-tmcmc.svg"); nothing # hide
# ![Sample Comparison](samples-tm-tmcmc.svg)

# ## Transform Between Spaces
#
# Demonstrate transformation from standard normal space to physical space and back.
# This shows how the transport map can be used for sampling and inverse transformations.

# Generate samples from the reference (prior) distribution
ref = copy(prior)
df = sample(ref, 1000)
#md nothing # hide

# Transform from standard normal space to physical space
to_physical_space!(tm, df)

scatter!(df.θ1, df.θ2, label="Transported", alpha=0.8)
#md savefig("samples-tm-tmcmc-2.svg"); nothing # hide
# ![Sample Comparison](samples-tm-tmcmc-2.svg)

# Transform samples from TMCMC to standard normal space
to_standard_normal_space!(tm, tmcmc_samples)

scatter(tmcmc_samples.θ1, tmcmc_samples.θ2, label="SNS", alpha=0.8, aspect_ratio=1)
#md savefig("samples-sns.svg"); nothing # hide
# ![Samples in SNS](samples-sns.svg)
