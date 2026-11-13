using UncertaintyQuantification
using QuasiMonteCarlo
using DataFrames
using QuadGK
using Random
using Statistics
using Plots

# Hong et al. (2023), Eq. (34): g(x) = 7 - (x1 + x3)^2 + x2; failure is g < 0.
# https://doi.org/10.1016/j.cma.2023.116410
seed = 20261112
σ = 2.0 # Normal(μ, σ) takes standard deviation as its second argument.

# Inputs; epistemic coordinates follow input order: (x3, μ).
x1 = RandomVariable(Normal(), :x1)
x3 = IntervalVariable(-1, 2, :x3)
x2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-2, 1), :σ => σ)), :x2)
inputs = [x1, x3, x2]

# Model and performance function
model = Model(df -> 7 .- (df.x1 .+ df.x3) .^ 2 .+ df.x2, :g)
performance(df) = df.g

# Shared initial design and aleatory samples for the two methods.
# Design columns are unit-cube coordinates for (x1 quantile, x2 quantile, x3, μ).
design = permutedims(QuasiMonteCarlo.sample(29, 4, LatinHypercubeSample(rng = MersenneTwister(seed))))
q = QuasiMonteCarlo.sample(1024, 2, SobolSample())
q = mod.(q .+ rand(MersenneTwister(seed + 1), 2), 1)
aleatory_design = permutedims(clamp.(q, eps(), 1 - eps()))
grid = Matrix(crossjoin(DataFrame(x3 = collect(-1.0:0.5:2.0)), DataFrame(μ = collect(-2.0:0.5:1.0))))

# CABO: 12 initial evaluations, 13 targeting the minimum, then 4 the maximum.
method = CABO(
    initial_points = 12,
    lower_bound_iterations = 13,
    upper_bound_iterations = 4,
    aleatory_samples = 1024,
    posterior_draws = 128,
    epistemic_candidates = grid
)

println("CABO runtime:")
@time pf, xmin, xmax, diagnostics = probability_of_failure(
    model, performance, inputs, method;
    rng = MersenneTwister(seed + 2), return_diagnostics = true,
    initial_design = design[1:12, :], aleatory_design = aleatory_design,
)

# Fixed-design GP: use all 29 design points and perform no adaptive iterations.
fixed_method = CABO(
    initial_points = 29,
    lower_bound_iterations = 0,
    upper_bound_iterations = 0,
    aleatory_samples = 1024,
    posterior_draws = 128,
    epistemic_candidates = grid
)

println("Fixed-design GP runtime:")
@time pf_fixed, xmin_fixed, xmax_fixed, fixed_diagnostics = probability_of_failure(
    model, performance, inputs, fixed_method;
    rng = MersenneTwister(seed + 3), return_diagnostics = true,
    initial_design = design, aleatory_design = aleatory_design,
)

# Independent quadrature reference; symmetry and ordering locate the extrema.
reference_probability(x3, μ) = quadgk(
    z -> pdf(Normal(), z) * cdf(Normal(μ, σ), (z + x3)^2 - 7),
    -Inf, Inf; rtol = 1.0e-10,
)[1]
reference = [reference_probability(0, 1), reference_probability(2, -2)]

println("CABO pf = $pf; minimum at $xmin, maximum at $xmax")
println("CABO model calls = $(diagnostics.calls)")
println("CABO absolute endpoint errors = $(abs.([pf.lb, pf.ub] .- reference))\n")

println("Fixed-design GP pf = $pf_fixed; minimum at $xmin_fixed, maximum at $xmax_fixed")
println("Fixed-design GP model calls = $(fixed_diagnostics.calls)")
println("Fixed-design GP absolute endpoint errors = $(abs.([pf_fixed.lb, pf_fixed.ub] .- reference))\n")

println("Quadrature pf = $reference; minimum at (0, 1), maximum at (2, -2)")
println("Table 2 CABO: 0.0076 at (-0.0353, 1), 0.4203 at (2, -2), using 29 calls.")
println("Table 2 reference: [0.0071, 0.4296], which differs from quadrature with σ = 2.")

# Figure 5 in Hong et al.: reference probability with projected training locations.
# These plots use the existing diagnostics and quadrature; no new simulator calls.
x3_axis = range(-1, 2; length = 101)
μ_axis = range(-2, 1; length = 101)
reference_surface = [reference_probability(x3, μ) for μ in μ_axis, x3 in x3_axis]
epistemic = diagnostics.training_data[:, [:_cabo_e1, :_cabo_e2]]
map_style = (
    xlabel = "x₃", ylabel = "μ", color = :viridis, clims = (0, 0.45),
    colorbar_title = "Failure probability", aspect_ratio = :equal,
    xticks = -1:0.5:2, yticks = -2:0.5:1, framestyle = :box,
    titlefontsize = 11, guidefontsize = 10, tickfontsize = 8,
)

training_panels = Plots.Plot[]
for (phase, title, available, optimum, exact, marker) in (
        (:minimum, "Lower-bound search", method.initial_points, xmin, [0.0, 1.0], :circle),
        (
            :maximum, "Upper-bound search", method.initial_points + method.lower_bound_iterations,
            xmax, [2.0, -2.0], :rect,
        ),
    )
    added = diagnostics.history.call[diagnostics.history.phase .== phase]
    panel = heatmap(
        x3_axis, μ_axis, reference_surface; map_style...,
        title = "$title: $available available + $(length(added)) added",
        xlims = (-1.15, 2.15), ylims = (-2.15, 1.15),
        legend = :outerbottom, legend_columns = 2, legendfontsize = 8,
    )
    scatter!(
        panel, epistemic[1:available, 1], epistemic[1:available, 2];
        marker = :diamond, markersize = 4, color = :deepskyblue,
        markerstrokecolor = :white, markerstrokewidth = 0.5, label = "Available samples"
    )
    scatter!(
        panel, epistemic[added, 1], epistemic[added, 2];
        marker = :cross, markersize = 13, color = :orangered,
        markerstrokewidth = 1.5, label = "Added samples"
    )
    scatter!(
        panel, [exact[1]], [exact[2]];
        marker = marker, markersize = 10, markercolor = :white,
        markerstrokecolor = :red, markerstrokewidth = 2, label = "Quadrature optimum"
    )
    # These are the final endpoints, using all 29 observations in both panels.
    scatter!(
        panel, [optimum[1]], [optimum[2]];
        marker = :star5, markersize = 6, color = :royalblue,
        markerstrokecolor = :white, label = "Final CABO optimum"
    )
    push!(training_panels, panel)
end
training_plot = plot(
    training_panels...; layout = (1, 2), size = (1200, 600), margin = 5Plots.mm,
    plot_title = "CABO evaluations in epistemic space (markers may overlap)",
    plot_titlefontsize = 14,
)

# Compare the final probability estimates on the same candidate grid and colour scale.
# Index by coordinates so the plot does not depend on the candidate row ordering.
x3_grid, μ_grid = sort(unique(grid[:, 1])), sort(unique(grid[:, 2]))
grid_index = Dict(Tuple(row) => i for (i, row) in enumerate(eachrow(grid)))
reference_grid = [reference_probability(x3, μ) for μ in μ_grid, x3 in x3_grid]
comparison_panels = [heatmap(x3_grid, μ_grid, reference_grid; map_style..., title = "Quadrature reference")]
for (title, info) in (("CABO (29 calls)", diagnostics), ("Fixed-design GP (29 calls)", fixed_diagnostics))
    estimates = vec(mean(info.probability_draws; dims = 2))
    surface = [estimates[grid_index[(x3, μ)]] for μ in μ_grid, x3 in x3_grid]
    push!(comparison_panels, heatmap(x3_grid, μ_grid, surface; map_style..., title = title))
end
comparison_plot = plot(
    comparison_panels...; layout = (1, 3), size = (1500, 450), margin = 5Plots.mm,
    xlims = (-1, 2), ylims = (-2, 1),
    plot_title = "Failure probability on the 7 × 7 candidate grid", plot_titlefontsize = 14,
)

plot_directory = joinpath(@__DIR__, "hong-cabo-plots")
mkpath(plot_directory)
for (name, figure) in (("training", training_plot), ("comparison", comparison_plot))
    display(figure)
    savefig(figure, joinpath(plot_directory, "$name.png"))
    savefig(figure, joinpath(plot_directory, "$name.svg"))
end
println("Plots saved to $plot_directory")
