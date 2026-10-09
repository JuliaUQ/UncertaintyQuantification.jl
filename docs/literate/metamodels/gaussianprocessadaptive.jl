#===
# Adaptive Gaussian Process Regression

Adaptive Gaussian process regression enriches an initial surrogate model with
new evaluations selected by a learning function. At every iteration, the
algorithm samples candidate points, selects the most informative one according
to the learning function, evaluates the expensive model there, and refits the
Gaussian process.

## Himmelblau's Function

As in the (non-adaptive) GP example, we consider the Himmelblau function in ``x1, x2 ∈ [-5, 5]``
as a test function.
===#

#md using UncertaintyQuantification # hide
#md using Plots # hide
#md using DataFrames # hide
#md using Optim # hide

#jl using UncertaintyQuantification
#jl using Plots
#jl using DataFrames
#jl using Optim # hide

#===
First, define the probabilistic input and the expensive model to approximate.
===#

x = RandomVariable.(Uniform(-5, 5), [:x1, :x2])
himmelblau = Model(
    df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2, :y
)
#md nothing # hide

#===
We start with the same initial Gaussian process surrogate as in the *regular* GP regression
example. We pass the design positionally and specify `mean`, `kernel`, and
`normalize` as keywords. With `normalize = true`, the input distributions define
the transformation to standard normal space.
===#

design = LatinHypercubeSampling(80)
mean_f = ConstMean(0.0)
kernel = SqExponentialKernel()

optimizer = MaximumLikelihoodEstimation(Optim.Adam(alpha = 0.005), Optim.Options(; iterations = 10, show_trace = false))

#md using Random #hide
#md Random.seed!(42) #hide
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
#md nothing # hide

#===
Next, we update the initial GP using a selected learning function and a set number of
additional points used to refine the initial GP. We use the [`MaximinDistance`](@ref)
acquisition function and select `20` additional points, drawing `1000` candidates
at each iteration.
===#

learning_function = MaximinDistance()
n_added_points = 20
candidate_sampling = MonteCarlo(1000)
#md nothing # hide

#===
We pass the fitted GP, inputs, model, acquisition function, and number of adaptive
iterations to [`AdaptiveGaussianProcess`](@ref). The returned model is a regular
`GaussianProcess` and retains the initial input transformation. We use
`deepcopy(initial_gp)` because refinement appends rows to the GP's training data,
and we want to preserve the initial model for comparison.
===#

adaptive_gp = AdaptiveGaussianProcess(
    deepcopy(initial_gp),
    x,
    himmelblau,
    learning_function,
    n_added_points;
    candidate_sampling = candidate_sampling,
    optimizer = optimizer
)
#md nothing # hide

#===
To assess the fitted surrogate, we compute the MSE between GP mean and the reference model.
We compare the MSE of the initial GP and the refined GP.

We start with the initial GP:
===#

test_data = sample(x, LatinHypercubeSampling(1000))
test_data_adaptive = deepcopy(test_data)
evaluate!(initial_gp, test_data; mode = :mean)
evaluate!(himmelblau, test_data)

mse = mean((test_data.y .- test_data.y_mean) .^ 2)
println("MSE (initial GP):  $mse")

# Then, we also evaluate the adaptively refined GP at the same test set:
evaluate!(adaptive_gp, test_data_adaptive; mode = :mean)
evaluate!(himmelblau, test_data_adaptive)

mse_adap = mean((test_data_adaptive.y .- test_data_adaptive.y_mean) .^ 2)
println("MSE (adaptive GP):  $mse_adap")
