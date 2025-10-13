using AbstractGPs
using UncertaintyQuantification
using DataFrames
using Random
using DisplayAs

Random.seed!(42)

x = collect(range(0, 10, 10))
noise_var = 0.1
y = sin.(x) + 0.3 * cos.(2 .* x) # + noise_var .* randn(length(x))
df = DataFrame(x = x, y = y)

σ² = 1e-5
kernel = SqExponentialKernel() ∘ ScaleTransform(3.0)# ∘ ScaleTransform(1.0)
gp = with_gaussian_noise(GP(0.0, kernel), σ²)

gpr = GaussianProcess(gp, df, :y) # ; optimization=MaximumLikelihoodEstimation()
    
using Plots

x_plot = collect(range(0, 5, 500))
y_true = sin.(x_plot) + 0.3 * cos.(2 .* x_plot)

prediction = DataFrame(:x => x_plot)
evaluate!(gpr, prediction; mode=:mean_and_var)

prediction_mean = prediction[!, :y_mean]
prediction_std = sqrt.(prediction[!, :y_var])

plot(x_plot, prediction_mean, color=:blue, label="Mean prediction")
plot!(x_plot, prediction_mean, ribbon=2 .* prediction_std, color=:grey, alpha=0.5, label="Confidence band")

# Optionally add ground truth function
plot!(x_plot, y_true, color=:red, label="Ground truth")