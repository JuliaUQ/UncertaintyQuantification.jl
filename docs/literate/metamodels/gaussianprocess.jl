#===
# Gaussian Process Regression

## Himmelblau's Function

In this example, we will model the following test function (known as Himmelblau's function) in the range ``x_1, x_2 ∈ [-5, 5]`` with a Gaussian process (GP) regression model.

It is defined as:

 ```math
f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2.
```
===#
# ![](himmelblau.svg)
#===
Analogue to the response surface example, we create an array of random variables, that will be used when evaluating the points that our experimental design produces.
===#

using UncertaintyQuantification

x = RandomVariable.(Uniform(-5, 5), [:x1, :x2])

himmelblau = Model(
    df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2, :y
)
#md nothing # hide

#===
Next, we choose an experimental design. Here, `LatinHypercubeSampling(80)` specifies both the sampling method and the number of training points:
===#

design = LatinHypercubeSampling(80)

#===
Next, we choose the prior mean and kernel, which we will pass directly to the constructor. Here we use a constant mean of 0.0 and a squared exponential kernel.
The constructor adds a small observation-noise variance (`σ² = 1.0e-10` by default) for numerical stability:
===#

mean_f = ConstMean(0.0)
kernel = SqExponentialKernel()

#===
Next, we set up an optimizer used in the log marginal likelihood maximization to find the optimal hyperparameters of our GP model. Here we use the Adam optimizer from the `Optim.jl` package with a learning rate of 0.005 and run it for 10 iterations:
===#
using Optim

optimizer = MaximumLikelihoodEstimation(Optim.Adam(alpha = 0.005), Optim.Options(; iterations = 10, show_trace = false))
#md nothing # hide

#===
The constructor takes the input random variables, model, design, and output symbol as positional arguments. The prior is specified with the `mean` and `kernel` keywords.
With `normalize = true` (the default), this constructor maps the inputs to standard normal space using their distributions. Predictions apply the same transformation automatically, while training data and outputs remain in physical space.
Set `normalize = false` to use physical-space inputs directly. When constructing a GP from a `DataFrame` instead, `normalize = true` standardizes input columns using their training-data means and standard deviations.

Here, both input variables are used as GP features. An optional vector of input names after `:y` can select the feature columns explicitly, for example `[:x1, :x2]`.
===#
#md using Random #hide
#md Random.seed!(42) #hide

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
#md nothing # hide

#===
The constructor optimizes the hyperparameters by maximizing the log marginal likelihood, then fits the posterior used for predictions. Set `learn_hyperparameters = false` to keep the specified prior hyperparameters fixed.
===#

#===
To evaluate the `GaussianProcess`, use `evaluate!(gp::GaussianProcess, data::DataFrame)` with the `DataFrame` containing the points you want to evaluate.
We can evaluate the predictive mean, variance, or both. The predictions are written to columns named after the output, here `:y_mean` and `:y_var`.
The default is to evaluate the mean prediction.
We can specify the evaluation mode via the `mode` keyword argument. Supported options are:
- `:mean` - predictive mean (default)
- `:var` - predictive variance
- `:mean_and_var` - both mean and variance
===#

test_data = sample(x, 1000)
evaluate!(gp_model, test_data; mode = :mean_and_var)
evaluate!(himmelblau, test_data)
mse = mean((test_data.y .- test_data.y_mean) .^ 2)
println("MSE (GP):  $mse")

#===
The plots below compare the predicted surface with the original function. We use a grid inside the input domain, excluding the endpoints of the uniform distributions because they map to infinite values in standard normal space:
===#

#md using Plots #hide
#md using DataFrames #hide
#md a = range(-5, 5; length=202)[2:end-1] #hide
#md b = range(-5, 5; length=202)[2:end-1] #hide
#md A = repeat(collect(a)', length(b), 1) #hide
#md B = repeat(collect(b), 1, length(a)) #hide
#md df = DataFrame(x1 = vec(A), x2 = vec(B)) #hide
#md evaluate!(gp_model, df; mode=:mean_and_var) #hide
#md evaluate!(himmelblau, df) #hide
#md gp_mean = reshape(df[:, :y_mean], length(b), length(a)) #hide
#md gp_var = reshape(df[:, :y_var], length(b), length(a)) #hide
#md himmelblau_values = reshape(df[:, :y], length(b), length(a)) #hide
#md s1 = surface(a, b, himmelblau_values; plot_title="Himmelblau's function")
#md s2 = surface(a, b, gp_mean; plot_title="GP posterior mean")
#md plot(s1, s2, layout = (1, 2), legend = false)
#md savefig("gp-mean-comparison.svg") # hide
#md s3 = surface(a, b, gp_var; plot_title="GP posterior variance") # hide
#md plot(s3, legend = false) #hide
#md savefig("gp-variance.svg"); nothing # hide

# ![](gp-mean-comparison.svg)

#===
The MSE depends on the experimental design and fitted hyperparameters. The GP also provides predictive variance, including observation noise, as a measure of uncertainty:
===#

# ![](gp-variance.svg)
