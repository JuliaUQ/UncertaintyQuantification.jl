using UncertaintyQuantification

x = RandomVariable.(Uniform(-5, 5), [:x1, :x2])

himmelblau = Model(
    df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2, :y
)

design = LatinHypercubeSampling(80)

mean_f = ConstMean(0.0)
kernel = SqExponentialKernel() ∘ ARDTransform([1.0, 1.0])
σ² = 1e-5

gp_prior = with_gaussian_noise(GP(mean_f, kernel), σ²)

using Optim

optimizer = MaximumLikelihoodEstimation(Optim.Adam(alpha=0.005), Optim.Options(; iterations=10, show_trace=false))

input_transform = ZScoreTransform()

gp_model = GaussianProcess(
    gp_prior,
    x,
    himmelblau,
    :y,
    design;
    input_transform=input_transform,
    optimization=optimizer
)

test_data = sample(x, 1000)
evaluate!(gp_model, test_data; mode=:mean_and_var)

test_data = sample(x, 1000)
evaluate!(gp_model, test_data)
evaluate!(himmelblau, test_data)

mse = mean((test_data.y .- test_data.y_mean) .^ 2)
println("MSE is:  $mse")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
