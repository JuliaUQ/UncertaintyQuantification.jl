# Metamodels

## Linear Basis Function Models 

Linear basis function models are a simple class of metamodels that express the predicted output as a linear combination of basis functions evaluated for the input variables defined as

```math
y(\mathbf{x}) = \sum_{i=1}^{n} \beta_i \varphi_i(\mathbf{x}),
```

where ``\varphi_i(x)`` are nonlinear basis functions that map the input space into an intermediate feature space and ``\mathbf{\beta}`` represents the adjustable weights. Some commonly used basis functions are introduced next.

### Monomial Basis

Monomial basis functions are defined as the powers, or products of powers in the multivariate case, of the input variables with a total degree of less than or equal to ``d``. For example, the monomial basis in two variables of degree ``d=3`` in graded reverse lexicographic order is given by

```math
\varphi(x) = \left[1, x_2, x_1, x_2^2, x_1x_2, x_1^2, x_2^3, x_1x_2^2, x_1^2x_2, x_1^3\right]
```

The construction of this [`MonomialBasis`](@ref) is presented next.

```@example monomials
using UncertaintyQuantification # hide
φ = MonomialBasis(2, 3)
```

By default the [`MonomialBasis`](@ref) includes the constant (zero degree) term. This behaviour can be changed by passing the `include_zero=false` keyword.

### Radial Basis

A radial basis function (RBF) is a real-valued function ``\varphi(||\mathbf{x}-\mathbf{c}||)`` that depends on the distance between the input ``\mathbf{x}`` and a fixed point ``\mathbf{c}``, called a *center*. These functions are multivariate but reduce to scalar functions of the radius ``r = ||\mathbf{x} - \mathbf{c}||``, hence the name **radial** basis function. The distance used is most commonly the euclidian norm. *UncertaintyQuantification* provides two types of RBFs.

#### Gaussian

```math

\varphi(r) = \exp(-\epsilon r)

```

here, ``\epsilon`` is a shape parameter. Gaussian radial basis functions are exposed through the [`GaussianRadialBasis`](@ref) struct.

#### Polyharmonic

```math
\varphi(r) = \begin{cases}
    r^k & \text{with} & k = 1,3,5,\ldots, \\
    r^k\ln(r) & \text{with} & k = 2,4,6,\ldots
\end{cases}
```

Note, that polyharmonic radial basis functions do not require a shape parameter. These RBs can be constructed using the [`PolyharmonicRadialBasis`](@ref) type.

### Least Squares

Despite the nonlinearity of the basis functions, the model remains linear in its parameters, which allows efficient estimation of the weights using ordinary least squares. Given ``m`` observations ``(\mathbf{x}_i, y_i)``, for ``i = 1,\ldots,m`` we construct the design matrix ``\Phi`` where each row contains the evaluated basis functions for one input:

```math
\Phi_{ij} = \varphi_j(\mathbf{x}_i), \text{ for } i=1,\ldots,n \text{ and } j=1,\ldots,m
```

The optimal weight vector is found by minimizing the sum-of-squares error

```math
E(\mathbf{\beta}) = \sum_{i=1}^{m} (y(\mathbf{x}_i, \mathbf{\beta}) - y_i)^2
```

yielding the closed-form solution via

```math
\mathbf{\beta} = (\Phi^{\top}\Phi)^{-1}\Phi^{\top}\mathbf{y},
```

where ``\mathbf{y}`` is the output vector.

### Example

Consider the function

```math
    y(x) = x\cos(x) ,
```

where ``x \sim U(-5, 5)``

We use [`HaltonSampling`](@ref) to sample 150 data points from the input variable, evaluate the model, and fit a [`LinearBasisFunctionModel`](@ref) using a [`MonomialBasis`](@ref) of degree ``d=9``.

```@example lbfm
using UncertaintyQuantification # hide
x = RandomVariable(Uniform(-5, 5), :x)
y = Model(
        df -> df.x .* cos.(df.x),
        :y,
    )
data = sample(x, HaltonSampling(150))
evaluate!(y, data)
lbfm = LinearBasisFunctionModel(data, :y, MonomialBasis(1, 9))
```

A plot comparing the resulting model to the data points is presented next.

```@example lbfm
using Plots # hide
using DataFrames # hide
scatter(data.x, data.y; markershape=:xcross, color=:red, label="data") # hide
plot_data = DataFrame(x = collect(range(-5.5, 5.5; length=1000))) # hide
evaluate!(lbfm, plot_data) # hide
plot!(plot_data.x, plot_data.y, color=:blue, label="model") # hide
savefig("lbfm.svg"); nothing # hide
```

![Linear Basis Function Model Plot](lbfm.svg)

### Response Surface

A linear basis function model constructed from a [`MonomialBasis`](@ref) is also known as a polynomial *Response Surface* [khuriResponseSurfaceMethodology2010](@cite). For this reason we provide a convenient alias [`ResponseSurface`](@ref). Using this alias the previous example can be adapted as follows.

```@example lbfm
rs = ResponseSurface(data, :y, 9)
```

### Design Of Experiments

Several experimental designs have been developed to efficiently estimate [`ResponseSurface`](@ref) models [khuriResponseSurfaceMethodology2010](@cite). Although designed for response surface methodology these designs can be used to fit any metamodel. However, for more complex models we suggest using [Quasi Monte Carlo](@ref) sampling schemes instead.

The designs implemented in *UncertaintyQuantification* are `TwoLevelFactorial`, `FullFactorial`, `FractionalFactorial`, `CentralComposite`, `BoxBehnken`, and `PlackettBurman`.

## Interval Predictor Model

An interval predictor model (IPM)[crespoIntervalPredictorModels2016](@cite) is a function that returns an interval instead of a precise value for the dependent variable given as

```math
    I_y(x,P) = \{y = p^T \varphi (x), p \in P\},
```

where ``\varphi(x)`` is an arbitrary basis and the uncertainty set ``P`` is defined as

```math
    P = \{p : \underline{p} \leq p \leq \overline{p} \}.
```

Using the *defining vertices*  of P ``\underline{p}`` and ``\overline{p}`` the IPM results as

```math
I_y(x,P) = \left[\underline{y}(x,\overline{p},\underline{p}), \overline{y}(x,\overline{p},\underline{p})\right],
```

with

```math
    \underline{y}(x,\overline{p},\underline{p}) = \overline{p}^T\left(\frac{\varphi(x) - |\varphi(x)|}{2}\right) + \underline{p}^T\left(\frac{\varphi(x) + |\varphi(x)|}{2}\right)
```

and

```math
    \overline{y}(x,\overline{p},\underline{p}) = \overline{p}^T\left(\frac{\varphi(x) + |\varphi(x)|}{2}\right) + \underline{p}^T\left(\frac{\varphi(x) - |\varphi(x)|}{2}\right).
```

Here, ``\underline{y}`` and ``\overline{y}`` are the lower and upper bounds of the IPM.

The distance between the lower and upper bound given by

```math
\delta_y(x,\overline{p},\underline{p}) = (\overline{p} - \underline{p})^T|\varphi(x)|
```

is known as the *spread* of the IPM. The optimal defining vertices for a given data set are found by minimizing the average spread such that all data points fall into the IPM by solving the following convex constrained optimization problem.

```math
\{\underline{p},\overline{p}\} = \underset{u,v}{\operatorname{argmax}}\{\mathbb{E}_x\left[\delta_y(x,u,v\right] : \underline{y}(x_i,v,u)\leq y_i \leq\overline{y}(x_i,v,u),u \leq v\}
```

### Example

Consider the function

```math
    y(x) = x^2\cos(x) - \sin(3x)\exp(-x^2) - x - \cos(x^2) +xg,
```

where ``x \sim U(-5.5, 5.5`` and ``g \sim N(0,1)``. We generate a data sequence of ``N = 150`` points and fit an `IntervalPredictorModel` using a [`MonomialBasis`](@ref) of sixth degree.

```@example ipm
using UncertaintyQuantification # hide

x = RandomVariable(Uniform(-5.5, 5.5), :x)
data = sample(x, HaltonSampling(150))

m = Model(
        df ->
            df.x .^ 2 .* cos.(df.x) .- sin.(3 * df.x) .* exp.(-df.x .^ 2) .- df.x .-
            cos.(df.x .^ 2) .+ df.x .* randn(size(df, 1)),
        :y,
    )

evaluate!(m, data)

b = MonomialBasis(1,6)
ipm = IntervalPredictorModel(data, :y, b)
```

The following figure presents the bounds of the resulting IPM and the corresponding least squares solution. Note, that the least squares solution is not guaranteed to be between the bounds of the IPM.

```@example ipm
using Plots # hide
using DataFrames # hide
ls = LinearBasisFunctionModel(data, :y, b) # hide

ipm_data = DataFrame(x = range(-5.5, 5.5; length=1000)) # hide
ls_data = copy(ipm_data) # hide

evaluate!(ipm, ipm_data) # hide
evaluate!(ls, ls_data) # hide

scatter(data.x, data.y; markershape=:xcross, color=:red, label="data") # hide
plot!(ipm_data.x, getproperty.(ipm_data.y, :lb), linestyle=:dash, color=:black, label="IPM bounds") # hide
plot!(ipm_data.x, getproperty.(ipm_data.y, :ub), linestyle=:dash, color=:black, label="") # hide
plot!(ls_data.x, ls_data.y; color=:blue, label ="LS") # hide

savefig("ipm.svg"); nothing # hide
```

![IPM Plot](ipm.svg)

### IPM reliability

The reliability of the IPM, that is the probability that and unobserved data point ``(x,y)`` will fall in the interval ``I_y(x,P)`` can be assessed using the [`reliability`](@ref) function. The function `reliability(ipm, ϵ)` returns the confidence parameter ``\beta``. Then, the reliability of the IPM is no less than  ``1 - \epsilon`` with confidence ``1 - \beta``.

```@example ipm
1 - reliability(ipm, 0.1548)
```

### Reliability analysis

As the IPM is an imprecise model, it can only be applied in a reliability analysis using the [`DoubleLoop`](@ref) or [`RandomSlicing`](@ref). For more information, see [Imprecise Reliability Analysis](@ref).

## Gaussian Process Regression

### Theoretical Background

A Gaussian Process (GP) is a collection of random variables, any finite subset of which has a joint Gaussian distribution. It is fully specified by a mean function $m(x)$ and a covariance (kernel) function $k(x, x')$. In GP regression, we aim to model an unknown function $f(x)$. Before observing any data, we assume that the function $f(x)$ is distributed according to a GP:

```math
f(x) \sim \mathcal{G}\mathcal{P}\left( m(x), k(x, x')  \right).
```

This prior GP specifies that any finite collection of function values follows a multivariate normal distribution.

To define a prior GP we use [`AbstractGPs.jl`](https://juliagaussianprocesses.github.io/AbstractGPs.jl/stable/) for the GP interface and mean function, and [`KernelFunctions.jl`](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/) for the definition of a covariance kernel. Below, we construct a simple prior GP with a constant zero mean function and a scaled squared exponential kernel:

```@example gaussianprocess
using UncertaintyQuantification

kernel = SqExponentialKernel() ∘ ScaleTransform(3.0)
gp = GP(0.0, kernel); nothing # hide
```

Note that the definition of a prior GP is handled by `UncertaintyQuantification` if no prior GP is specified. The construction of a `GaussianProcess` is flexible. Mean functions, kernels and many other parameters can be specified later directly in the constructor of the `GaussianProcess`.

#### Posterior Gaussian Process

The posterior GP represents the distribution of functions after incorporating observed data. We denote the observation data as:

```math
\mathcal{D} = \lbrace (\hat{x}_i, \hat{f}_i) \mid i=1, \dots, N \rbrace,
```

where $\hat{f}_i = f(\hat{x}_i)$ in the noise-free observation case, and $\hat{f}_i = f(\hat{x}_i) + e_i$ in the noisy case, with independent noise terms $e_i \sim \mathcal{N}(0, \sigma_e^2)$. Let $\hat{X} = [\hat{x}_1, \dots, \hat{x}_N]$ denote the collection of observation data locations. The corresponding mean vector and covariance matrix are:

```math
\mu(\hat{X}) = [m(\hat{x}_1), \dots, m(\hat{x}_N)], \quad K(\hat{X}, \hat{X}) \text{ with entries } K_{ij} = k(\hat{x}_i, \hat{x}_j).
```

For a new input location $x^*$ we are interested at the unknown function value $f^* = f(x^*)$. By the definition of a GP, the joint distribution of observed outputs $\hat{f}_i$ and the unknown $f^*$ is multivariate Gaussian:

```math
\begin{bmatrix} \hat{f}\\ f^* \end{bmatrix} = \mathcal{N}\left( \begin{bmatrix} \mu(\hat{X}) \\ m(x^*) \end{bmatrix},  \begin{bmatrix} K(\hat{X}, \hat{X}) & K(\hat{X}, x^*)\\ K(x^*, \hat{X}) & K(x^*, x^*) \end{bmatrix} \right),
```

where:

- $K(\hat{X}, \hat{X})$ is the covariance matrix with entries $K_{ij} = k(\hat{x}_i, \hat{x}_j)$,
- $K(\hat{X}, x^*)$ is the covariance matrix with entries $K_{i1} = k(\hat{x}_i, x^*)$,
- and $K(x^*, x^*)$ is the variance at the unknown input location.

We can then obtain the posterior distribution of $f^*$ from the properties of multivariate Gaussian distributions (see, e.g. Appendix A.2 in [rasmussen2005gaussian](@cite)), by conditioning the joint Gaussian on the observed outputs $\hat{f}_i$:

```math
f^* \mid \hat{X}, \hat{f}, x^* \sim \mathcal{N}(\mu^*(x^*), \Sigma^*(x^*)),
```

with

```math
\mu^*(x^*) = m(x^*) + K(x^*, \hat{X})K(\hat{X}, \hat{X})^{-1}(\hat{f} - \mu(\hat{X})), \\
\Sigma^*(x^*) = K(x^*, x^*) - K(x^*, \hat{X})K(\hat{X}, \hat{X})^{-1}K(\hat{X}, x^*).
```

In the noisy observation case, the covariance between training points is adjusted by adding the noise variance:

```math
K(\hat{X}, \hat{X}) \rightarrow K(\hat{X}, \hat{X}) + \sigma^2_{e}I.
```

The computation of the posterior predictive distribution generalizes straightforwardly to multiple input locations, providing both the posterior mean, which can serve as a regression estimate of the unknown function, and the posterior variances, which quantify the uncertainty at each point. Because the posterior is multivariate Gaussian, one can also sample function realizations at specified locations to visualize possible functions consistent with the observed data.

To construct a posterior GP, we need to define training data in form of a `DataFrame`. Constructing a `GaussianProcess` model will then automatically compute the posterior GP to predict requested the modeled output $y$ and by default it will also optimize the hyperparameters. If this is not desired, the input `learn_hyperparameters=false` can be set.

The following creates a standard GP with mean function `ConstMean()`, kernel `SqExponentialKernel()`, and directly optimizes the hyperparameters. Note that while `ConstMean(0.0)` and `ZeroMean()` provide the same zero-mean prior GP, using `ConstMean()` also allows for optimization of the mean.
We also equip the GP with small observation noise $\sigma^2$, which has implications on the numerical stability and allows the GP to handle imprecise data. The noise can also be optimized as part of the hyperparameter optimization, but it is not optimized by default.
To specify different mean functions and/or kernels, either construct a GP manually beforehand, or use them as inputs.

```@example gaussianprocess
using DataFrames # hide
x = collect(range(0, 10, 10))
y = sin.(x) + 0.3 * cos.(2 .* x)
df = DataFrame(x = x, y = y)

mean_fct = ConstMean(0.0)
kernel = SqExponentialKernel() ∘ ScaleTransform(3.0)

gp_prior = GP(mean_fct, kernel)

σ² = 1e-5

# these are equivalent
gp_model = GaussianProcess(gp_prior, df, :y; σ²=σ²)
gp_model = GaussianProcess(df, :y; σ²=σ², mean_fct=mean_fct, kernel=kernel)
# providing the input learn_noise=true also optimizes the data noise
gp_model = GaussianProcess(df, :y; σ²=σ², mean_fct=mean_fct, kernel=kernel, learn_noise=true); nothing # hide
```

Now we can use our GP model to predict at new input locations `x_test`:

```@example gaussianprocess
using Plots # hide
x_test = collect(range(0, 5, 500))
prediction = DataFrame(:x => x_test)

evaluate!(gp_model, prediction; mode=:mean_and_var)

prediction_mean = prediction[!, :y_mean] # hide
prediction_std = sqrt.(prediction[!, :y_var]) # hide

p = plot(x_test, prediction_mean, color=:blue, label="Mean prediction") # hide
plot!(
    x_test, prediction_mean, ribbon=2 .* prediction_std,
    color=:grey, alpha=0.5, label="Confidence band"
) # hide

y_true = sin.(x_test) + 0.3 * cos.(2 .* x_test) # hide
plot!(x_test, y_true, color=:red, label="True function") # hide

savefig(p, "posterior-gp.svg"); nothing # hide
```

![Fitted Gaussian process](posterior-gp.svg)

#### Hyperparameter optimization

GP models typically contain hyperparameters in their mean functions $m(x; \theta_m)$ and covariance kernel functions $k(x, x'; \theta_k)$. The observation noise variance $\sigma^2_{e}$ is also considered a hyperparameter related to the kernel. The choice of hyperparameters strongly affects the quality of the posterior GP.

A common approach to selecting hyperparameters is maximum likelihood estimation (MLE) (see, e.g. [rasmussen2005gaussian](@cite)), where we maximize the likelihood of observing the training data $\mathcal{D}$ under the chosen GP prior.

The marginal likelihood of the observed training outputs $\hat{f}$ is:

```math
p(\hat{f} \mid \hat{X}, \theta_m, \theta_k, \sigma^2_{e}) = \mathcal{N}(\hat{f} \mid \mu_{\theta_m}(\hat{X}), K_{\theta_k}(\hat{X}, \hat{X}) + \sigma^2_{e}I),
```

where $\mu_{\theta_m}(\hat{X})$ and $K_{\theta_k}(\hat{X}, \hat{X})$ denote the parameter dependent versions of the previously defined quantities.

For numerical reasons, the logarithm of the marginal likelihood is typically used. Maximizing the log marginal likelihood with respect to the hyperparameters then yields the parameters that best explain the observed data. After obtaining the optimal hyperparameters, the posterior GP can be constructed as described above.

`UncertaintyQuantification.jl` provides a default optimizer for the hyperparameters based on the [`MaximumLikelihoodEstimation`](@ref) constructor.

```
optimizer::AbstractHyperparameterOptimization=MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations=100, show_trace=false))
```

If other options are desired, a different optimizer can be constructed based on [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/). The script below shows the difference between an optimized and unoptimized GP.

```@example gaussianprocess
using Optim

optimization = MaximumLikelihoodEstimation(
                Optim.LBFGS(),
                Optim.Options(; iterations=10, show_trace=false)
            )

gp_model = GaussianProcess(df, :y;
                           σ²=σ²,
                           mean_fct=mean_fct,
                           kernel=kernel,
                           optimizer=optimization
                           )

gp_model_unoptimized = GaussianProcess(df, :y;
                            σ²=σ²,
                            mean_fct=mean_fct,
                            kernel=kernel,
                            learn_hyperparameters=false
                           )

prediction = DataFrame(:x => x_test)
prediction_unopt = DataFrame(:x => x_test)
evaluate!(gp_model, prediction; mode=:mean_and_var)
evaluate!(gp_model_unoptimized, prediction_unopt; mode=:mean_and_var)

prediction_mean = prediction[!, :y_mean] # hide
prediction_std = sqrt.(prediction[!, :y_var]) # hide

prediction_mean_unopt = prediction_unopt[!, :y_mean] #hide
prediction_std_unopt = sqrt.(prediction_unopt[!, :y_var]) # hide

p = plot(x_test, prediction_mean, ribbon=2 .* prediction_std, color=:blue, alpha=0.5, label="Optimized") # hide
plot!(x_test, prediction_mean_unopt, ribbon=2 .* prediction_std_unopt, color=:grey, alpha=0.5, label="Not Optimized") # hide

plot!(x_test, y_true, color=:red, label="True function") # hide

savefig(p, "posterior-gp-opt.svg"); nothing # hide
```

![Optimized Gaussian process](posterior-gp-opt.svg)

Internally, `MaximumLikelihoodEstimation()` defaults to using [`LBFGS`](https://julianlsolvers.github.io/Optim.jl/stable/algo/lbfgs/) optimizer that performs 100 optimization steps with standard optimization hyperparameters as defined [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/). Note that any other first-order optimizer supported by [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/), along with its corresponding hyperparameters, can also be used when constructing [`MaximumLikelihoodEstimation`](@ref).

During optimization, GP hyperparameters $\theta_m, \theta_k$ and $\sigma^2_{e}$ are automatically extracted and updated.

We support the automatic extraction of hyperparameters from mean functions provided by [`AbstractGPs.jl`](https://juliagaussianprocesses.github.io/AbstractGPs.jl/stable/api/#Mean-functions), with the exception of:

- Custom mean functions [`CustomMean`](https://juliagaussianprocesses.github.io/AbstractGPs.jl/stable/api/#AbstractGPs.CustomMean). These are defined with a custom function that itself could depend on hyperparameters. These additional hyperparameters are ignored in the optimization.

Kernel functions are defined with the kernels and transformations provided by [`KernelFunctions.jl`](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/). For similar reasons as with `CustomMean`, we do not extract potential function hyperparameters from the following kernels or transforms:

- Transforms defined with custom functions [`FunctionTransform`](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/transform/#KernelFunctions.FunctionTransform),
- The [`GibbsKernel`](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/kernels/#KernelFunctions.GibbsKernel), which models a kernel lengthscale parameter with the help of a function.

Further, GP models containing the following kernels are not supported for hyperparameter optimization currently:

- Multi-output kernels [`MOKernel`](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/kernels/#Multi-output-Kernels),
- Neural kernel networks [`NeuralKernelNetwork`].
