using UncertaintyQuantification
using AbstractGPs
using Random
using Optim


# Setup Himmelblau example
x = RandomVariable.(Uniform(-5, 5), [:x1, :x2])
himmelblau = Model(
    df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2, :y
)
design = LatinHypercubeSampling(100)
training_data = sample(x, design)
evaluate!(himmelblau, training_data)

# Setup the GP
σ² = 1e-5
kernel = SqExponentialKernel() ∘ ARDTransform([0.5, 0.5])
gp = with_gaussian_noise(GP(0.0, kernel), σ²)

optimizer = MaximumLikelihoodEstimation(Optim.Adam(alpha=0.01), Optim.Options(; iterations=1000, show_trace=false))
# optimizer = MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations=10, show_trace=false))

gpr = GaussianProcess(
    gp, x, himmelblau, :y, design; input_transform=ZScoreTransform(), output_transform=ZScoreTransform(), optimization=optimizer
)

test_data = sample(x, 1000)
evaluate!(gpr, test_data)

p_data = test_data[:, [:x1, :x2]]
evaluate!(himmelblau, p_data)

mse = mean((p_data.y .- test_data.y) .^ 2)
println("MSE is:  $mse")

using Plots
using DataFrames
# SNSInputTransform will crash the plotting routine on -5 and 5 values
a = range(-5, 5; length=1000) 
b = range(5, -5; length=1000)
himmelblau_f(x1, x2) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2
function gpr_f(x, y)
    df = DataFrame(x1 = x, x2 = y)
    evaluate!(gpr, df)
    return only(df[:, :y])
end

s1 = surface(a, b, himmelblau_f; plot_title="Himmelblau's function")
s2 = surface(a, b, gpr_f; plot_title="Gaussian process regression")
plot(s1, s2, layout = (1, 2), legend = false)