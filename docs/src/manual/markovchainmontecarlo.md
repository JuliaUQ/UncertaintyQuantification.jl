# Sampling with Markov Chain Monte Carlo

Placeholder for MCMC samples of the Banana distribution


```math
p(x_1, x_2) = \phi(x_1) \cdot \phi(x_1^2 - x_2)
```

### Single-Component Metropolis Hastings

[`SingleComponentMetropolisHastings`](@ref)


### Metropolis Hastings

[`MetropolisHastings`](@ref)

```@example MCMC
using UncertaintyQuantification # hide
using LinearAlgebra # hide
using Plots # hide
using DataFrames # hide
logtarget = df -> logpdf.(Normal(), df.x1) + logpdf.(Normal(), df.x1 .^ 2 - df.x2)
```

```@example MCMC
Σ = 0.1 * I(2)
x0 = DataFrame(x1 = [4.0], x2 = [4.0])
mh = MetropolisHastings(MvNormal([0, 0], Σ), x0, 500, 0)
samples_mh, α = sample(logtarget, mh)
burnin = 100
plot(samples_mh.x1[1:burnin], samples_mh.x2[1:burnin], marker=:o, label="Burnin")
plot!(samples_mh.x1[burnin+1:end], samples_mh.x2[burnin+1:end], marker=:o, label="Samples")
scatter!(x0.x1, x0.x2, c=:black, label="x₀", m=:diamond, ms=10)
savefig("banana-mcmc-samples-burnin.svg"); nothing # hide
```
![Banana MCMC](banana-mcmc-samples-burnin.svg)

```@example MCMC
mh = MetropolisHastings(MvNormal([0, 0], Σ), x0, 5_000, 1_000)
samples_mh, α = sample(logtarget, mh)
scatter(samples_mh.x1, samples_mh.x2)
savefig("banana-mcmc-samples.svg"); nothing # hide
```

![Banana MCMC](banana-mcmc-samples.svg)


### Adaptive Metropolis Hastings

[`AdaptiveMetropolisHastings`](@ref)