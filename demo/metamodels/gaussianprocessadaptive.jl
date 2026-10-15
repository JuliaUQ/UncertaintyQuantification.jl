using UncertaintyQuantification
using QuasiMonteCarlo
using Plots
using DataFrames
using Optim

x = RandomVariable.(Uniform(-5, 5), [:x1, :x2])
himmelblau = Model(
    df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2, :y
)

design = QuasiMonteCarloSampling(80, LatinHypercubeSample())
mean_f = ConstMean(0.0)
kernel = SqExponentialKernel()

optimizer = MaximumLikelihoodEstimation(Optim.Adam(alpha = 0.005), Optim.Options(; iterations = 10, show_trace = false))

initial_gp = GaussianProcess(
    x,
    himmelblau,
    design,
    :y;
    mean = mean_f,
    kernel = kernel,
    normalize = true,
    optimizer = optimizer
)

learning_function = MaximinDistance()
n_added_points = 20
candidate_sampling = MonteCarlo(1000)

adaptive_gp = AdaptiveGaussianProcess(
    deepcopy(initial_gp),
    x,
    himmelblau,
    learning_function,
    n_added_points;
    candidate_sampling = candidate_sampling,
    optimizer = optimizer
)

test_data = UncertaintyQuantification.sample(x, QuasiMonteCarloSampling(1000, LatinHypercubeSample()))
test_data_adaptive = deepcopy(test_data)
evaluate!(initial_gp, test_data; mode = :mean)
evaluate!(himmelblau, test_data)

mse = mean((test_data.y .- test_data.y_mean) .^ 2)
println("MSE (initial GP):  $mse")

evaluate!(adaptive_gp, test_data_adaptive; mode = :mean)
evaluate!(himmelblau, test_data_adaptive)

mse_adap = mean((test_data_adaptive.y .- test_data_adaptive.y_mean) .^ 2)
println("MSE (adaptive GP):  $mse_adap")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
