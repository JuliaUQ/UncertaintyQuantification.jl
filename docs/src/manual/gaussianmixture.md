# Gaussian Mixture Models

Gaussian Mixture Models (GMMs) are a probabilistic model that assumes all data points are generated from a mixture of ``K`` Gaussian distributions with unknown parameters.
A GMM is characterized by its probability density function (PDF), which is expressed as a weighted sum of ``K`` component Gaussian densities:

```math
p(\boldsymbol{x}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k),
```

where:

- ``\boldsymbol{x} \in \mathbb{R}^d`` is a ``d``-dimensional observation vector
- ``\pi_k`` are the mixing coefficients (weights) satisfying ``\pi_k \geq 0`` and ``\sum_{k=1}^K \pi_k = 1``
- ``\mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)`` is the PDF of the ``k``-th multivariate Gaussian component with mean vector ``\boldsymbol{\mu}_k \in \mathbb{R}^d`` and covariance matrix ``\boldsymbol{\Sigma}_k \in \mathbb{R}^{d \times d}``

In practice, GMMs are widely applied for multivariate density estimation, clustering, and dimensionality reduction from data [hastieElementsStatiscialLearning2009](@cite).
For univariate density estimation, we refer to [Kernel Density Estimation](@ref).

## Expectation-Maximization Algorithm for GMMs

One way to find the parameters of a GMM from a set of samples is to use the Expectation-Maximization (EM) algorithm.
Here, we show the basic steps to fit a GMM to data using the EM algorithm based on Ref.[hastieElementsStatiscialLearning2009](@cite).
The EM algorithm iteratively refines the parameters of the GMM by alternating between two steps:

1. **Expectation Step (E-step)**: Calculate the expected value of the latent variables given the current parameters.
2. **Maximization Step (M-step)**: Update the parameters to maximize the expected log-likelihood found in the E-step.

Given a dataset ``\mathcal{D} = \{\boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_N\}`` of ``N`` independent observations, the goal is to estimate the parameters ``\boldsymbol{\theta} = \{\pi_1, \ldots, \pi_K, \boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K, \boldsymbol{\Sigma}_1, \ldots, \boldsymbol{\Sigma}_K\}`` that maximize the log-likelihood:

```math
\ell(\boldsymbol{\theta}) = \sum_{n=1}^{N} \log p(\boldsymbol{x}_n \mid \boldsymbol{\theta}) = \sum_{n=1}^{N} \log \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right).
```

The EM algorithm introduces latent variables ``z_{nk} \in \{0, 1\}`` indicating whether observation ``n`` belongs to component ``k``, where ``\sum_{k=1}^K z_{nk} = 1`` for each ``n``. The algorithm iteratively maximizes the expected log-likelihood by alternating between two steps:

### Expectation Step

Compute the posterior probabilities (responsibilities) ``\gamma_{nk}`` for each observation-component pair:

```math
\gamma_{nk}^{(t+1)} = \frac{\pi_k^{(t)} \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}{\sum_{j=1}^{K} \pi_j^{(t)} \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}.
```

### Maximization Step

Update the parameters using the computed responsibilities, where ``N_k = \sum_{n=1}^{N} \gamma_{nk}^{(t+1)}``:

```math
\pi_k^{(t+1)} = \frac{N_k}{N}, \quad \boldsymbol{\mu}_k^{(t+1)} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma_{nk}^{(t+1)} \boldsymbol{x}_n, \quad \boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma_{nk}^{(t+1)} (\boldsymbol{x}_n - \boldsymbol{\mu}_k^{(t+1)})(\boldsymbol{x}_n - \boldsymbol{\mu}_k^{(t+1)})^T.
```

### Algorithm Convergence

The algorithm terminates when the change in log-likelihood between iterations falls below a predefined threshold ``\epsilon``:

```math
|\ell(\boldsymbol{\theta}^{(t+1)}) - \ell(\boldsymbol{\theta}^{(t)})| < \epsilon.
```

## Implementation

In *UncertaintyQuantification.jl*, a GMM can be fitted to data using the [`GaussianMixtureModel`](@ref) function, which implements the EM algorithm described above. The function takes a `DataFrame` containing the samples and the number of components `k` as input. Optionally, one can set the maximum number of iterations and tolerance.
The GMM is constructed as:

```@example GaussianMixture
using DataFrames # hide
using UncertaintyQuantification # hide
# Generate sample data with two clusters
df = DataFrame(x1=randn(100), x2=2*randn(100))
k = 2
gmm = GaussianMixtureModel(df, k) # maximum_iterations = 100, tolerance=1e-4
nothing # hide
```

This returns a `MultivariateDistribution` object. The fitted mixture model, constructed using the EM algorithm, is stored as a `Distributions.MixtureModel` from [`Distributions.jl`](https://juliastats.org/Distributions.jl/stable/mixture/) in the field `gmm.d`:

```@example GaussianMixture
gmm.d
```

Since the GMM is returned as a `MultivariateDistribution`, we can perform sampling and evaluation of the
PDF the same way as for other (multivariate) random variables.
For a more detailed explanation, we refer to the [Gaussian Mixture Model Example](@ref).

!!! note "Alternative mixture model construction"
    (Gaussian) Mixture models constructed with other packages can also be used to construct a
    `MultivariateDistribution`, as long as they return a `Distributions.MixtureModel`.