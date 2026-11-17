using UncertaintyQuantification
using QuasiMonteCarlo
using Plots
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

optim = MaximumLikelihoodEstimation(
    Optim.Adam(alpha = 0.005), Optim.Options(; iterations = 10, show_trace = false)
)
akmcs = AdaptiveGaussianProcessMonteCarlo(
    1_000_000,
    QuasiMonteCarloSampling(10, LatinHypercubeSample()),
    DeviationNumber(),
    2.0,
    200;
    optimizer = optim
)

pf_akmcs, std_akmcs, samples_akmcs, adaptive_gp =
    probability_of_failure([g1, g2, g3, g4], g, x, akmcs)

mc = MonteCarlo(1.0e6)

pf_mc, std_mc, samples = probability_of_failure(models, g, x, mc)

grid = -7.5:0.05:7.5
n_grid = length(grid)

contour_data = DataFrame(x1 = repeat(grid, inner = n_grid), x2 = repeat(grid, outer = n_grid))

evaluate!(models, contour_data)
contour_data.g = g(contour_data)
evaluate!(adaptive_gp, contour_data)

true_values = reshape(contour_data.g, n_grid, n_grid)
gp_values = reshape(contour_data.g_mean, n_grid, n_grid)

contour(
    grid, grid, true_values; levels = [0.0], linewidth = 1, label = "g",
    xlabel = "x₁", ylabel = "x₂", cbar = false, xlims = (-8, 8), ylims = (-8, 8),
    aspect_ratio = :equal, color = :gray,
)
contour!(grid, grid, gp_values; levels = [0.0], linewidth = 2, label = "GP mean", cbar = false, color = :black)

n_initial = akmcs.initial_design.n
n_added = size(adaptive_gp.gp.data, 1) - n_initial

scatter!(
    adaptive_gp.gp.data.x1[1:n_initial], adaptive_gp.gp.data.x2[1:n_initial],
    label = "$(n_initial) Initial"
)
scatter!(
    adaptive_gp.gp.data.x1[(n_initial + 1):end], adaptive_gp.gp.data.x2[(n_initial + 1):end],
    label = "$(n_added) Added"
)

println("Probability of failure (Adaptive GP):  $pf_akmcs ± $std_akmcs")
println("Probability of failure (Standard MC):  $pf_mc ± $std_mc")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
