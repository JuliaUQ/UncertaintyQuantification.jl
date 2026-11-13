using UncertaintyQuantification
using QuasiMonteCarlo
using ImplicitPlots, Plots
using DataFrames
using Optim

x = RandomVariable.(Normal(), [:x1, :x2])

k = 6.0
g1 = Model(df -> 3 .+ 0.1 * (df.x1 .- df.x2) .^ 2 .- (df.x1 .+ df.x2) ./ sqrt(2), :g1)
g2 = Model(df -> 3 .+ 0.1 * (df.x1 .- df.x2) .^ 2 .+ (df.x1 .+ df.x2) ./ sqrt(2), :g2)
g3 = Model(df -> (df.x1 .- df.x2) .+ k / sqrt(2), :g3)
g4 = Model(df -> (df.x2 .- df.x1) .+ k / sqrt(2), :g4)
models = [g1, g2, g3, g4]
g = df -> min.(df.g1, df.g2, df.g3, df.g4)

optim = MaximumLikelihoodEstimation(Optim.Adam(alpha = 0.005), Optim.Options(; iterations = 10, show_trace = false))
akmcs = AdaptiveGaussianProcessMonteCarlo(
    1_000_000,
    QuasiMonteCarloSampling(10, LatinHypercubeSample()),
    DeviationNumber(),
    2.0,
    200;
    optimizer = optim
)

pf_akmcs, std_akmcs, samples_akmcs, adaptive_gp = probability_of_failure([g1, g2, g3, g4], g, x, akmcs)

mc = MonteCarlo(1.0e6)

pf_mc, std_mc, samples = probability_of_failure(models, g, x, mc)

lsf(x, y) = minimum(
    [
        3 + 0.1 * (x - y)^2 - (x + y) / sqrt(2),
        3 + 0.1 * (x - y)^2 + (x + y) / sqrt(2),
        (x - y) + k / sqrt(2),
        (y - x) + k / sqrt(2),
    ]
)

function lsf_GP_adapt(x, y)
    df = DataFrame([:x1, :x2] .=> [x, y])
    evaluate!(adaptive_gp, df)
    return only(df.g_mean)
end

implicit_plot(lsf; xlims = (-8, 8), ylims = (-8, 8), linecolor = :gray)
implicit_plot!(lsf_GP_adapt; xlims = (-8, 8), ylims = (-8, 8), linecolor = :black)
scatter!(adaptive_gp.gp.data.x1, adaptive_gp.gp.data.x2)

println("Probability of failure (Adaptive GP):  $pf_akmcs ± $std_akmcs")
println("Probability of failure (Standard MC):  $pf_mc ± $std_mc")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
