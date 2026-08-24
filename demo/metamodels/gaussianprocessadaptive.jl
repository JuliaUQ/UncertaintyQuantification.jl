using UncertaintyQuantification
using Plots
using DataFrames
using Optim # hide

x = RandomVariable.(Uniform(-5, 5), [:x1, :x2])
himmelblau = Model(
    df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2, :y
)

design = LatinHypercubeSampling(80)
mean_f = ConstMean(0.0)
kernel = SqExponentialKernel()

gp_prior = GP(mean_f, kernel)
input_transform = ZScoreTransformChoice()
optimizer = MaximumLikelihoodEstimation(Optim.Adam(alpha = 0.005), Optim.Options(; iterations = 10, show_trace = false))

initial_gp = GaussianProcess(
    gp_prior,
    x,
    himmelblau,
    :y;
    experimental_design = design,
    input_transform = input_transform,
    optimizer = optimizer
)

learning_function = MaximinDistance()
n_added_points = 20

adaptive_gp = AdaptiveGaussianProcess(
    deepcopy(initial_gp),
    x,
    himmelblau,
    learning_function,
    n_added_points;
    optimizer = optimizer
)

test_data = sample(x, LatinHypercubeSampling(1000))
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
