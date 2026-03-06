#===

## Beam Example: Comparison of TMCMC and Transport Maps

This example demonstrates Bayesian model updating for a beam deflection problem using two different approaches: Transitional Markov Chain Monte Carlo (TMCMC) and Transport Maps.
Both methods are used to infer unknown parameters of a clamped beam model from noisy displacement measurements.

The clamped beam has a rectangular cross-section with width ``b = 0.2`` m and height ``h = 0.1`` m, and length ``L = 10`` m. The Young's modulus is ``E = 210`` GPa.
A load ``F`` is applied at the position ``a L`` along the beam, as seen in the figure below.
This example is based on an [grashornEfficientDiagnostics2024](@cite).

![Beam](../assets/beam.svg)

The displacement at the end of the beam, denoted as ``s``, is obatined as a function of ``a`` and ``F``
through the following analyical expression:

```math
    \mathcal{M}(\theta) := s = \frac{F \cdot (a L)^2}{6 E I} (3 L - a L)
```

where ``I = \frac{b h^3}{12}`` is moment of inertia of the cross section and ``\theta = [a, F]``.

To perform Bayesian updating, a set of 10 displacement measurements corrupted by Gaussian noise with standard deviation ``\sigma = 0.01`` m are given.
The goal is to infer the unknown parameters ``a`` (relative position along the beam) and ``F`` (applied force) from these measurements using Bayesian inference.

===#

#md using UncertaintyQuantification # hide
#md using Plots # hide

#jl using UncertaintyQuantification
#jl using Plots

#===

First, we define the beam parameters and the displacement model:
===#

b = 0.2
h = 0.1
L = 10.
E = 210e9

A = b * h
I = b * h^3 / 12

s(a, F) = F .* (a * L) .^ 2 / (6 * E * I) .* (3 * L .- a * L)
#md nothing #hide

#===
Next, we define the prior distributions for the unknown parameters. We assume that ``a`` follows a Beta distribution and ``F`` follows a Normal distribution centered at 1000 N with standard deviation 300 N.
===#

prior = [
    RandomVariable(Beta(10, 3), :a),
    RandomVariable(Normal(1000, 300), :F),
]

#===
The measurement noise standard deviation and the observed data ``D`` are:
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
#md nothing #hide

#===
We define the forward model ``\mathcal{M}(\theta)`` as a [`Model`](@ref).
The likelihood assumes that the measurement errors are independent and normally distributed:

```math
    P(D | \theta) = \prod_{i=1}^{10} \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left[ - \frac{1}{2} \left(\frac{\mathcal{M}(\theta) - D_i}{\sigma}\right)^2 \right].
```
For better stability, we use the log-likelihood for the updating.
===#

M = Model(df -> s(df.a, df.F), :disp)
Like = df -> sum([logpdf.(Normal(y, σ), df.disp) for y in data])
#md nothing #hide

#===
### Bayesian Updating with TMCMC

First, we apply the TMCMC algorithm to sample from the posterior distribution.
We use these samples to compare with the results obtained with the transport map.
===#

tmcmc = TransitionalMarkovChainMonteCarlo(prior, 1_000, 3)
samples, evidence = bayesianupdating(Like, [M], tmcmc)
#md nothing #hide

#===
### Bayesian Updating with Transport Maps

Transport Maps provide an alternative approach to Bayesian inference. They construct a deterministic transformation that maps a standard normal distribution to the posterior distribution. This is achieved by learning a triangular map using polynomial basis functions.

We define a polynomial transport map of order 2 with 2 input dimensions.
Additionally, we define a quadrature scheme used to compute the KL-divergence, i.e., the target of the optimization.
Finally, we store these along with the `prior` in the [`TransportMapBayesian`](@ref) object:
===#
T = PolynomialMap(2, 2)
quadrature = GaussHermiteWeights(3, 2)
transportmap = TransportMapBayesian(prior, T, quadrature)

#===
We perform the Bayesian updating using finite differences for gradient computation in the optimization.
The map we defined is optimized by calling the [`bayesianupdating`](@ref) function.
===#

tm = bayesianupdating(Like, [M], transportmap)

#===
Once the map coefficients are optimized, we can sample from the transport map posterior and compare the samples to those obtained from TMCMC.
The TM-based samples are obtained from sampling in the reference space (i.e., standard normal space) and applying the mapping.
In the figure we see a good agreement of the samples obtained with TMCMC and from the transport map.
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
Further, transport maps provide a formulation of the poserior density in terms of the reference density and the map, as also outlined in [Variational Inference with Transport Maps](@ref).
We can evaluate the density by calling [`pdf(tm::TransportMap, x::AbstractVector{<:Real})`](@ref).
In the figure below, the TM-approximated pdf is plotted over the samples generated with TMCMC.
A good agreement of the samples and the pdf is observed.
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
Finally, we can compute the [`variancediagnostic`](@ref) to assess the quality of the transport map approximation.
The diagnostic measures the variance of the log-ratio between the pushforward density (mapping from target to reference density)
and the reference density:

```math
\varepsilon_\sigma = \frac{1}{2} \operatorname{Var}[\log \pi(T(z)) + \log |\operatorname{det} \nabla T(z)| - \rho(z)] .
```

A smaller variance indicates a better fit of the transport map.
===#

df = sample(prior, 1000)
to_standard_normal_space!(prior, df) # generate standard normal samples
var_diag = variancediagnostic(tm, df)
println("Variance diagnostic: $var_diag")
