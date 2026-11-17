#===

## Adaptive Gaussian Process Monte Carlo simulation

This method estimates the probability of failure of a performance function by building
a Gaussian process (GP) surrogate from an initial design and iteratively enriching it
with one point at a time, chosen from a set of candidate samples by an acquisition
function, until the stopping criterion or the sample budget is reached.
It is also known in literature as AK-MCS (Adaptive Kriging Monte Carlo Simulation).

### Four-Branch Function (Example 1 from Echard et al. (2011))

We consider the following series system of performance functions, as in Example 1
of [echard2011akmcs](@cite):

```math
G(x_1, x_2) = \min \left\{
\begin{array}{l}
3 + 0.1(x_1 - x_2)^2 - \frac{x_1 + x_2}{\sqrt{2}} \,; \\[6pt]
3 + 0.1(x_1 - x_2)^2 + \frac{x_1 + x_2}{\sqrt{2}} \,; \\[6pt]
(x_1 - x_2) + \frac{k}{\sqrt{2}} \,; \\[6pt]
(x_2 - x_1) + \frac{k}{\sqrt{2}}
\end{array}
\right\}
```
where ``k=6.0`` or ``k=7.0`` usually. The input random variables are independent and normally distributed, with
``x_1, x_2 ∼ \mathcal{N}(0, 1)``.

===#

#md using UncertaintyQuantification # hide
#md using QuasiMonteCarlo # hide
#md using Plots # hide
#md using DataFrames # hide
#md using Optim # hide

#jl using UncertaintyQuantification
#jl using QuasiMonteCarlo
#jl using Plots
#jl using DataFrames
#jl using Optim

#===
First, define the probabilistic input and the series of performances.
===#

x = RandomVariable.(Normal(), [:x1, :x2])

k = 6.0
g1 = Model(df -> 3 .+ 0.1 * (df.x1 .- df.x2) .^ 2 .- (df.x1 .+ df.x2) ./ sqrt(2), :g1)
g2 = Model(df -> 3 .+ 0.1 * (df.x1 .- df.x2) .^ 2 .+ (df.x1 .+ df.x2) ./ sqrt(2), :g2)
g3 = Model(df -> (df.x1 .- df.x2) .+ k / sqrt(2), :g3)
g4 = Model(df -> (df.x2 .- df.x1) .+ k / sqrt(2), :g4)
models = [g1, g2, g3, g4]
g = df -> min.(df.g1, df.g2, df.g3, df.g4)
#md nothing # hide

#===
We now construct the adaptive Gaussian process Monte Carlo simulation.
We use a small number of initial samples to initially train the GP surrogate. The initial
samples are generated using a quasi-Monte Carlo method. Very few samples are needed to
initially train the GP surrogate. The acquisition function is set to `DeviationNumber()`,
which selects the next point to evaluate based on the deviation number, i.e., the points
with a high potential risk of crossing the predicted separator limit state function,
approximated by GP. The enrichment stops when the deviation number is below a threshold
of 2.0, or when the maximum number of samples is reached. The GP surrogate is fitted with
a constant mean function and a squared exponential kernel, and the hyperparameters are
optimized after each enrichment.
===#

#md using Random #hide
#md Random.seed!(67) #hide
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
#md nothing # hide

#===
Next, we run the simulation and estimate the probability of failure. The result is the
estimated probability of failure, the standard deviation of the estimate, the samples
evalauted, and the fitted GP surrogate. The fitted GP surrogate can be used to visualize
the limit state function and the uncertainty of the prediction.
===#

pf_akmcs, std_akmcs, samples_akmcs, adaptive_gp =
    probability_of_failure([g1, g2, g3, g4], g, x, akmcs)
#md nothing # hide

#===
Finally, we visualize the limit state function. We will also plot the GP approximation
of the limit state function, and the samples evaluated by the adaptive GP Monte Carlo simulation.
We also compare the results with a standard Monte Carlo simulation with 1 million samples.
===#

mc = MonteCarlo(1.0e6)

pf_mc, std_mc, samples = probability_of_failure(models, g, x, mc)
#md nothing # hide

grid = -7.5:0.05:7.5
n_grid = length(grid)

contour_data = DataFrame(x1 = repeat(grid, inner = n_grid), x2 = repeat(grid, outer = n_grid))
#md nothing # hide

# Evaluate the `models` and compute the final limit state value ``g``
evaluate!(models, contour_data)
contour_data.g = g(contour_data)
evaluate!(adaptive_gp, contour_data)
#md nothing # hide

# Reshape the data and contour the limit states
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
#md savefig("agp-lsf-and-samples.svg"); nothing # hide

# ![Exact limit state function and fitted GP, and samples from the GP](agp-lsf-and-samples.svg)

# Finally, the probability of failure estimates are printed.
println("Probability of failure (Adaptive GP):  $pf_akmcs ± $std_akmcs")
println("Probability of failure (Standard MC):  $pf_mc ± $std_mc")
