#===

## Beam Example: Comparison of TMCMC and Transport Maps

This example demonstrates Bayesian model updating for a beam deflection problem using two different approaches: Transitional Markov Chain Monte Carlo (TMCMC) and Transport Maps.
Both methods are used to infer unknown parameters of a beam model from noisy displacement measurements.

The clamped beam has a rectangular cross-section with width ``b = 0.2`` m and height ``h = 0.1`` m, and length ``L = 10`` m. The Young's modulus is ``E = 210`` GPa.
A load ``F`` is applied at the position ``a L`` along the beam, as seen in the figure below.

![Beam](../assets/beam.svg)

The displacement at the end of the beam, denoted as ``s``,  is given by:

```math
s = \mathcal{M}(a, F) = \frac{F \cdot (a L)^2}{6 E I} (3 L - a L)
```

where ``I = \frac{b h^3}{12}`` is moment of inertia of the cross section.

We have 10 displacement measurements corrupted by Gaussian noise with standard deviation ``\sigma = 0.01`` m.
Our goal is to infer the unknown parameters ``a`` (relative position along the beam) and ``F`` (applied force) from these measurements using Bayesian inference.

===#

#md using UncertaintyQuantification # hide
#md using Plots # hide

#jl using UncertaintyQuantification
#jl using Plots

#===
### Define the Problem

First, we define the beam parameters and the displacement model:
===#

b = 0.2
h = 0.1
L = 10.
E = 210e9

A = b * h
I = b * h^3 / 12

s(a, F) = F .* (a * L) .^ 2 / (6 * E * I) .* (3 * L .- a * L)

#===
Next, we define the prior distributions for the unknown parameters. We assume that ``a`` follows a Beta distribution (constrained to [0, 1]) and ``F`` follows a Normal distribution centered at 1000 N with standard deviation 300 N.
===#

prior = [
    RandomVariable(Beta(10, 3), :a),
    RandomVariable(Normal(1000, 300), :F),
]

#===
The measurement noise standard deviation and the observed data are:
===#

σ = 0.01

data = [
    0.04700676518380301
    0.05567472107255563
    0.06033689503633009
    0.035675890874077895
    0.02094933145952007
    0.044602949523632154
    0.04886405043326749
    0.0456339834330763
    0.04457204918859585
    0.05735639275860812
]

#===
We define the forward model and the log-likelihood function. The likelihood assumes that the measurement errors are independent and normally distributed.
===#

M = Model(df -> s(df.a, df.F), :disp)
Like = df -> sum([logpdf.(Normal(y, σ), df.disp) for y in data])

#===
### Bayesian Updating with TMCMC

First, we apply the TMCMC algorithm to sample from the posterior distribution.
===#

tmcmc = TransitionalMarkovChainMonteCarlo(prior, 1_000, 3)
samples, evidence = bayesianupdating(Like, [M], tmcmc)

#===
To visualize the results, we compute the unnormalized posterior on a grid and plot it alongside the TMCMC samples:
===#

loglike(x) = sum([logpdf.(Normal(y, σ), s(x[1], x[2])) for y in data])
posterior(x) = exp(logpdf(prior[1], x[1]) + logpdf(prior[2], x[2]) + loglike(x))

x1_grid = 0.3:0.01:1
x2_grid = 0:10:2500

post = [posterior([x1, x2]) for x2 in x2_grid, x1 in x1_grid]

scatter(samples.a, samples.F, alpha=0.8, label="TMCMC Samples")
contour!(x1_grid, x2_grid, post)
xlabel!("a [-]")
ylabel!("F [N]")
title!("Unnormalized posterior and TMCMC samples")
#md savefig("beam-tmcmc-posterior.svg"); nothing # hide

# ![TMCMC posterior](beam-tmcmc-posterior.svg)

#===
We can also visualize the likelihood function alone:
===#

likelihood_vals = [exp.(loglike([x1, x2])) for x2 in x2_grid, x1 in x1_grid]
contour(x1_grid, x2_grid, likelihood_vals)
xlabel!("a [-]")
ylabel!("F [N]")
title!("Likelihood")
#md savefig("beam-likelihood.svg"); nothing # hide

# ![Likelihood function](beam-likelihood.svg)

#===
### Bayesian Updating with Transport Maps

Transport Maps provide an alternative approach to Bayesian inference. They construct a deterministic transformation that maps a standard normal distribution to the posterior distribution. This is achieved by learning a triangular map using polynomial basis functions.

We define a polynomial transport map of order 2 with 2 input dimensions:
===#
T = PolynomialMap(2, 2)
quadrature = GaussHermiteWeights(3, 2)
transportmap = TransportMapBayesian(prior, T, quadrature)

#===
We perform the Bayesian updating using automatic differentiation for gradient computation:
===#

tm = bayesianupdating(Like, [M], transportmap, nothing, AutoFiniteDiff())

#===
Now we can sample from the transport map posterior and compare the samples to those obtained from TMCMC:
===#

df = sample(tm, 1000)
scatter(df.a, df.F, alpha=0.8, label="TM Samples")
scatter!(samples.a, samples.F, alpha=0.8, label="TMCMC Samples")
xlabel!("a [-]")
ylabel!("F [N]")
title!("Comparison of TM and TMCMC samples")
#md savefig("beam-tm-tmcmc-comparison.svg"); nothing # hide

# ![Transport Map vs TMCMC](beam-tm-tmcmc-comparison.svg)

#===
We can also visualize the posterior density learned by the transport map:
===#

x1_grid = range(0.3, 1, 100)
x2_grid = range(0, 2500, 100)

post = [pdf(tm, [x1, x2]) for x2 in x2_grid, x1 in x1_grid]
scatter(samples.a, samples.F, alpha=0.8, label="TMCMC Samples")
contour!(x1_grid, x2_grid, post)
xlabel!("a [-]")
ylabel!("F [N]")
title!("TM-posterior and TMCMC samples")
#md savefig("beam-tm-posterior.svg"); nothing # hide

# ![Transport Map posterior](beam-tm-posterior.svg)

#===
Finally, we can compute a variance diagnostic to assess the quality of the transport map approximation:
===#

df = sample(prior, 1000)
to_standard_normal_space!(prior, df)
var_diag = variancediagnostic(tm, df)
println("Variance diagnostic: $var_diag")
