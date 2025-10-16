#===
# Gaussian Process Regression 

## Himmelblau's Function

In this example, we will model the following test function (known as Himmelblau's function) in the range ``x1, x2 ∈ [-5, 5]`` with a Gaussian process (GP) regression model.

It is defined as:

 ```math
f(x1, x2) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2.
```
===#
# ![](himmelblau.svg)
#===
Aanalogue to the response surface example, we create an array of random variables, that will be used when evaluating the points that our experimental design produces.
===#

using UncertaintyQuantification

x = RandomVariable.(Uniform(-5, 5), [:x1, :x2])

himmelblau = Model(
    df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2, :y
)

#===
Next, we chose a experimental design. In this example, we are using a `LatinHyperCube` design from which we draw 80 samples to train our model:
===#

design = LatinHypercubeSampling(80)

#===
After that, we construct a prior GP model. Here we assume a constant mean of 0.0 and a squared exponential kernel with automatic relevance determination (ARD). 
We also assume a small Gaussian noise term in the observations for numerical stability:
===#

mean_f = ConstMean(0.0)
kernel = SqExponentialKernel() ∘ ARDTransform([1.0, 1.0])
σ² = 1e-5

gp_prior = with_gaussian_noise(GP(mean_f, kernel), σ²)

#===
Next, we set up an optimizer used in the log marginal likelihood maximization to find the optimal hyperparameters of our GP model. Here we use the Adam optimizer from the `Optim.jl` package with a learning rate of 0.005 and run it for 10 iterations.:
===#
using Optim

optimizer = MaximumLikelihoodEstimation(Optim.Adam(alpha=0.005), Optim.Options(; iterations=10, show_trace=false))

#===
Finally, we define an input standardization (here a z-score transform). While not strictly necessary for this example, standardization can help finding good hyperparameters. 
Note that we can also define an output transform to scale the output for training the GP. When evaluating the GP model, the input will be automatically transformed with the fitted standardization.
The output will be transformed back to the original scale automatically as well.
===#

input_transform = ZScoreTransform()

#===
The GP regression model is now constructed by calling the `GaussianProcess` constructor with the prior GP, the input random variables, the model, the output symbol, the experimental design, and the optional input and output transforms as well as the hyperparameter optimization method.
The construction then samples the experimental design, evaluates the model at the sampled points, standardizes the input and output data, optimizes the hyperparameters of the GP, and constructs the posterior GP.
===#
#md using Random #hide
#md Random.seed!(42) #hide

gp_model = GaussianProcess(
    gp_prior, 
    x, 
    himmelblau, 
    :y, 
    design; 
    input_transform=input_transform, 
    optimization=optimizer
)

#===
To evaluate the `GaussianProcess`, use `evaluate!(gp::GaussianProcess, data::DataFrame)` with the `DataFrame` containing the points you want to evaluate. 
The evaluation of a GP is not unique, and we can choose to evaluate the mean prediction, the prediction variance, a combination of both, or draw samples from the posterior distribution.
The default is to evaluate the mean prediction.
We can specify the evaluation mode via the `mode` keyword argument. Supported options are:
- `:mean` - predictive mean (default)
- `:var` - predictive variance
- `:mean_and_var` - both mean and variance
- `:sample` - random samples from the predictive distribution
===#

test_data = sample(x, 1000)
evaluate!(gp_model, test_data; mode=:mean_and_var)

#===
The mean prediction of our model in this case has an mse of about 65 and looks like this in comparison to the original:
===#

#md using Plots #hide
#md using DataFrames #hide
#md a = range(-5, 5; length=200) #hide
#md b = range(-5, 5; length=200) #hide
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
Note that the mse in comparison to the response surface model (with an mse of about 1e-26) is significantly higher.
However, the GP model also provides a measure of uncertainty in its predictions via the predictive variance.
===#

# ![](gp-variance.svg)

#jl test_data = sample(x, 1000)
#jl evaluate!(gp_model, test_data)
#jl evaluate!(himmelblau, test_data)

#jl mse = mean((test_data.y .- test_data.y_mean) .^ 2)
#jl println("MSE is:  $mse")
