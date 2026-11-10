using UncertaintyQuantification

## Set-up the reliability problem

x = RandomVariable.(Normal(), [:x1, :x2])

k = 6.0
g1 = Model(df -> 3 .+ 0.1 * (df.x1 .- df.x2) .^ 2 .- (df.x1 .+ df.x2) ./ sqrt(2), :g1)
g2 = Model(df -> 3 .+ 0.1 * (df.x1 .- df.x2) .^ 2 .+ (df.x1 .+ df.x2) ./ sqrt(2), :g2)
g3 = Model(df -> (df.x1 .- df.x2) .+ k / sqrt(2), :g3)
g4 = Model(df -> (df.x2 .- df.x1) .+ k / sqrt(2), :g4)
models = [g1, g2, g3, g4]
g = df -> min.(df.g1, df.g2, df.g3, df.g4)

mc = MonteCarlo(1.0e6)

pf_mc, std_mc, samples = probability_of_failure(models, g, x, mc)

# just some plotting of the full problem
using ImplicitPlots, Plots

lsf(x, y) = minimum(
    [
        3 + 0.1 * (x - y)^2 - (x + y) / sqrt(2),
        3 + 0.1 * (x - y)^2 + (x + y) / sqrt(2),
        (x - y) + k / sqrt(2),
        (y - x) + k / sqrt(2),
    ]
)

implicit_plot(lsf; xlims = (-8, 8), ylims = (-8, 8), linecolor = :gray)
scatter!(samples.x1[g(samples) .< 0], samples.x2[g(samples) .< 0], m = 2, markercolor = :red, markerstrokewidth = 0.5)

## Here I start with the algorithm

# What do I need?
# 1. The model(s) for the reliability
#    i. The performance function
# 2. A simulation method
# 3. An initial GP
#    i. A sampling strategy
#   ii. An initial number of samples
# 4. A learning function (or a set of learning functions)

using QuasiMonteCarlo, Optim, DataFrames

num_starting_samples = 10;  # REQUIRED_INPUT
# create a starting GP of the performance function
design = QuasiMonteCarloSampling(num_starting_samples, LatinHypercubeSample())
mean_f = ConstMean(0.0)
kernel = SqExponentialKernel()

optimizer = MaximumLikelihoodEstimation(Optim.Adam(alpha = 0.005), Optim.Options(; iterations = 10, show_trace = false))  # REQUIRED_INPUT

# append the preformance function to the GP as a model
push!(models, Model(df -> g(df), :g))

initial_gp = GaussianProcess(
    x, # REQUIRED_INPUT
    models, # REQUIRED_INPUT
    design, # REQUIRED_INPUT
    :g;
    mean = mean_f,
    kernel = kernel, # REQUIRED_INPUT
    normalize = true, # should the user pass this?
    optimizer = optimizer # REQUIRED_INPUT
) # REQUIRED_INPUT

learning_function = DeviationNumber() # REQUIRED_INPUT
candidate_sampling = MonteCarlo(1.0e6) # this is hard-coded. Should the user secify this? Maybe as an optional

adaptive_gp = AdaptiveGaussianProcess(
    deepcopy(initial_gp),
    x, # same as initial GP
    models, # same as initial GP
    learning_function,  # REQUIRED_INPUT
    candidate_sampling; # should the user pass this? Maybe have a optional input
    optimizer = optimizer # same as initial GP
)
refine!(adaptive_gp, 120)

g = df -> df.g_mean # duplicate the output of the mean of the GP as the performance for use in the reliability
pf_akmc, std_akmc, samples_akmc = probability_of_failure(adaptive_gp, g, x, mc)

# some plots
function lsf_GP_adapt(x, y)
    df = DataFrame([:x1, :x2] .=> [x, y])
    evaluate!(adaptive_gp, df)
    return only(df.g_mean)
end
implicit_plot(lsf; xlims = (-8, 8), ylims = (-8, 8), linecolor = :gray)
implicit_plot!(lsf_GP_adapt; xlims = (-8, 8), ylims = (-8, 8), linecolor = :black)
scatter!(adaptive_gp.gp.data.x1, adaptive_gp.gp.data.x2)
