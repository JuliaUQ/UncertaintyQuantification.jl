using UncertaintyQuantification
using Plots

b = 0.2
h = 0.1
L = 10.
E = 210e9

A = b * h
I = b * h^3 / 12

s(a, F) = F .* (a * L) .^ 2 / (6 * E * I) .* (3 * L .- a * L)

prior = [
    RandomVariable(Beta(10, 3), :a),
    RandomVariable(Normal(1000, 300), :F),
]

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

M = Model(df -> s(df.a, df.F), :disp)
Like = df -> sum([logpdf.(Normal(y, σ), df.disp) for y in data])

tmcmc = TransitionalMarkovChainMonteCarlo(prior, 1_000, 3)
samples, evidence = bayesianupdating(Like, [M], tmcmc)

T = PolynomialMap(2, 2)
quadrature = GaussHermiteWeights(3, 2)
transportmap = TransportMapBayesian(prior, T, quadrature)

tm = bayesianupdating(Like, [M], transportmap)

df = sample(tm, 1000)
scatter(df.a, df.F, alpha=0.8, label="TM Samples")
scatter!(samples.a, samples.F, alpha=0.8, label="TMCMC Samples")
xlabel!("a [-]")
ylabel!("F [N]")
title!("Comparison of TM and TMCMC samples")

x1_grid = range(0.3, 1, 100)
x2_grid = range(0, 2500, 100)

post = [pdf(tm, [x1, x2]) for x2 in x2_grid, x1 in x1_grid]
scatter(samples.a, samples.F, alpha=0.8, label="TMCMC Samples")
contour!(x1_grid, x2_grid, post)
xlabel!("a [-]")
ylabel!("F [N]")
title!("TM-posterior and TMCMC samples")

df = sample(prior, 1000)
to_standard_normal_space!(prior, df) # generate standard normal samples
var_diag = variancediagnostic(tm, df)
println("Variance diagnostic: $var_diag")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
