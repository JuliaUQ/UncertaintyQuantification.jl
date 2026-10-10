using UncertaintyQuantification
using QuasiMonteCarlo

x = RandomVariable.(Uniform(-5, 5), [:x1, :x2])

himmelblau = Model(
    df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2, :y
)

design = QuasiMonteCarloSampling(80, LatinHypercubeSample())

mean_f = ConstMean(0.0)
kernel = SqExponentialKernel()

using Optim

optimizer = MaximumLikelihoodEstimation(Optim.Adam(alpha = 0.005), Optim.Options(; iterations = 10, show_trace = false))

gp_model = GaussianProcess(
    x,
    himmelblau,
    design,
    :y;
    mean = mean_f,
    kernel = kernel,
    normalize = true,
    optimizer = optimizer
)

test_data = UncertaintyQuantification.sample(x, 1000)
evaluate!(gp_model, test_data; mode = :mean_and_var)
evaluate!(himmelblau, test_data)
mse = mean((test_data.y .- test_data.y_mean) .^ 2)
println("MSE (GP):  $mse")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
