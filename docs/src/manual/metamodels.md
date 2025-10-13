# Metamodels

## Design Of Experiments

Design Of Experiments (DOE) offers various designs that can be used for creating a model of a given system. The core idea is to evaluate significant points of the system in order to obtain a sufficient model while keeping the effort to achieve this relatively low. Depending on the parameters, their individual importance and interconnections, different designs may be adequate.

The ones implemented here are `TwoLevelFactorial`, `FullFactorial`, `FractionalFactorial`, `CentralComposite`, `BoxBehnken` and `PlackettBurman`.

## Response Surface

A Response Surface is a simple polynomial surrogate model. It can be trained by providing it with evaluated points of a function or any of the aforementioned experimental designs.

## Gaussian Process Regression

### Theoretical Background
A Gaussian Process (GP) is a collection of random variables, any finite subset of which has a joint Gaussian distribution. It is fully specified by a mean function $m(x)$ and a covariance (kernel) function $k(x, x')$. In GP regression, we aim to model an unknown function $f(x)$. Before observing any data, we assume that the function $f(x)$ is distributed according to a GP:

```math
f(x) \sim \mathcal{G}\mathcal{P}\left( m(x), k(x, x')  \right).
```

This prior GP specifies that any finite collection of function values follows a multivariate normal distribution. 

To define a prior GP we use [`AbstractGPs.jl`](https://juliagaussianprocesses.github.io/AbstractGPs.jl/stable/) for the GP interface and mean function, and [`KernelFunctions.jl`](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/) for the definition of a covariance kernel. Below, we construct a simple prior GP with a constant zero mean function and a scaled squared exponential kernel:

```@example gaussianprocess
using AbstractGPs
kernel = SqExponentialKernel() ∘ ScaleTransform(3.0)
gp = GP(0.0, kernel)
return nothing # hide
```

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

To construct a posterior GP from our previously defined prior GP, we need to define training data in form of a `DataFrame`. Constructing a `GaussianProcess` model will then automatically compute the posterior GP to predict requested the modeled output $y$. In this example, we equip the prior GP with a small Gaussian observation noise with zero mean and variance $\sigma^2_{e}=\sigma^2$, which improves the numerical stability of the covariance matrix.

```@example gaussianprocess
using DataFrames # hide
x = collect(range(0, 10, 10))
y = sin.(x) + 0.3 * cos.(2 .* x)
df = DataFrame(x = x, y = y)

σ² = 1e-5 
gp = with_gaussian_noise(gp, σ²)
posterior_gp = GaussianProcess(gp, df, :y)
return nothing # hide
```

```@example gaussianprocess
using Plots # hide
x_plot = collect(range(0, 5, 500)) # hide
y_true = sin.(x_plot) + 0.3 * cos.(2 .* x_plot) # hide

prediction = DataFrame(:x => x_plot) # hide
evaluate!(posterior_gp, prediction; mode=:mean_and_var) # hide
prediction_mean = prediction[!, :y_mean] # hide
prediction_std = sqrt.(prediction[!, :y_var]) # hide

p = plot(x_plot, prediction_mean, color=:blue, label="Mean prediction") # hide
plot!(
    x_plot, prediction_mean, ribbon=2 .* prediction_std, 
    color=:grey, alpha=0.5, label="Confidence band"
) # hide
plot!(x_plot, y_true, color=:red, label="True function") # hide

savefig(p, "posterior-gp.svg"); # hide
return nothing # hide
```
![](posterior-gp.svg)

#### Hyperparameter optimization
Gaussian process models typically contain hyperparameters in their mean functions $m(x; \theta_m)$ and covariance kernel functions $k(x, x'; \theta_k)$. The observation noise variance $\sigma^2_{e}$ is also considered a hyperparameter related to the kernel. The choice of hyperparameters strongly affects the quality of the posterior GP. A common approach to selecting hyperparameters is maximum likelihood estimation (MLE) (see, e.g. [rasmussen2005gaussian](@cite)), where we maximize the likelihood of observing the training data $\mathcal{D}$ under the chosen GP prior.

The marginal likelihood of the observed training outputs $\hat{f}$ is:

```math
p(\hat{f} \mid \hat{X}, \theta_m, \theta_k, \sigma^2_{e}) = \mathcal{N}(\hat{f} \mid \mu_{\theta_m}(\hat{X}), K_{\theta_k}(\hat{X}, \hat{X}) + \sigma^2_{e}I),
```

where $\mu_{\theta_m}(\hat{X})$ and $K_{\theta_k}(\hat{X}, \hat{X})$ denote the parameter dependent versions of the previously defined quantities. For numerical reasons, the logarithm of the marginal likelihood is typically used. Maximizing the log marginal likelihood with respect to the hyperparameters then yields the parameters that best explain the observed data. After obtaining the optimal hyperparamters, the posterior GP can be constructed as described above.

- Custom mean functions [`CustomMean`](https://juliagaussianprocesses.github.io/AbstractGPs.jl/stable/api/#AbstractGPs.CustomMean)
- Multi-output kernels [`MOKernel`](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/kernels/#Multi-output-Kernels)
- Neural kernel networks [`NeuralKernelNetwork`]
- FunctionTransform
- GibbsKernel

