using AbstractGPs
using UncertaintyQuantification
using DataFrames
using Random

Random.seed!(42)

x = RandomVariable(Uniform(0, 5), :x)

σ = 0.1
f = 3.0
noisy_sinus = Model(
    df -> sin.(f .* df.x) .+ σ .* randn(size(df.x)), :y
)
experimentaldesign = LatinHypercubeSampling(10)

σ² = σ^2
kernel = SqExponentialKernel()
kernel = SqExponentialKernel() + PeriodicKernel(; r=[f])
gp = with_gaussian_noise(GP(0.0, kernel), σ²)
gp = GP(0.0, kernel)

gpr = GaussianProcess(
    gp, x, noisy_sinus, :y, experimentaldesign;
    #optimization=NoHyperparameterOptimization()
) 
    
using Plots

x_plot = collect(range(0, 5, 500))
y_true = sin.(f .* x_plot)

prediction = DataFrame(:x => x_plot)
mean_and_var!(gpr, prediction)

prediction_mean = prediction[!, :y_mean]
prediction_std = sqrt.(prediction[!, :y_var])

plot(x_plot, prediction_mean, color=:blue, label="Mean prediction")
plot!(x_plot, prediction_mean, ribbon=2 .* prediction_std, color=:grey, alpha=0.5, label="Confidence band")

# Optionally add ground truth function
plot!(x_plot, y_true, color=:red, label="Ground truth")