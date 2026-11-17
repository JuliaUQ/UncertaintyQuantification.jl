"""
    AdaptiveGaussianProcessMonteCarlo(n, initial_design, acquisition_function, stopping, maximum_samples; kwargs...)

Creates an adaptive Gaussian process (GP) Monte Carlo simulation for reliability analysis.
A GP surrogate of the performance function is built from an initial design and iteratively
enriched with one point at a time, chosen from a set of candidate samples by an
acquisition function, until the stopping criterion or the sample budget is reached.

# Arguments
- `n::Integer`: Number of samples to estimate the probability of failure with the refined gaussian process. Must be greater than zero.
- `initial_design::AbstractMonteCarlo`: Sampling method for the initial training set. Defaults to `QuasiMonteCarloSampling(10, LatinHypercubeSample())`.
- `acquisition_function::AbstractReliabilityAcquisitionFunction`: Criterion used to select new training points. Defaults to `DeviationNumber()`.
- `stopping::Float64`: Threshold of the acquisition criterion used to stop the enrichment. Defaults to `2.0`.
- `maximum_samples::Integer`: Maximum number of performance function evaluations (training samples). Defaults to `1000`.

# Keyword Arguments
- `σ²::Float64`: Observation noise variance of the GP. Defaults to `1.0e-10`.
- `learn_noise::Bool`: Whether the noise variance is optimized together with the hyperparameters. Defaults to `false`.
- `learn_hyperparameters::Bool`: Whether the GP hyperparameters are optimized after each enrichment. Defaults to `true`.
- `mean::AbstractGPs.MeanFunction`: Mean function of the GP. Defaults to `ConstMean(0.0)`.
- `kernel::Kernel`: Covariance kernel of the GP. Defaults to `SqExponentialKernel()`.
- `normalize::Bool`: Whether inputs/outputs are normalized before fitting the GP. Defaults to `true`.
- `optimizer::AbstractHyperparameterOptimization`: Hyperparameter optimization method. Defaults to `MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false))`.
- `candidate_sampling::AbstractMonteCarlo`: Sampling method generating the candidate pool, also used for the failure probability estimate. Defaults to `MonteCarlo(100_000)`.

# Examples
```julia
akmcs = AdaptiveGaussianProcessMonteCarlo(5)

akmcs = AdaptiveGaussianProcessMonteCarlo(
    5,
    QuasiMonteCarloSampling(20, LatinHypercubeSample()),
    ExpectedImprovement(),
    0.001,
    200;
    kernel = Matern52Kernel(),
    candidate_sampling = MonteCarlo(10^6),
)
```
"""
struct AdaptiveGaussianProcessMonteCarlo
    n::Integer
    initial_design::AbstractMonteCarlo
    acquisition_function::AbstractReliabilityAcquisitionFunction
    stopping::Float64
    maximum_samples::Integer
    σ²::Float64
    mean::AbstractGPs.MeanFunction
    kernel::Kernel
    normalize::Bool
    candidate_sampling::AbstractMonteCarlo
    learn_noise::Bool
    learn_hyperparameters::Bool
    optimizer::UncertaintyQuantification.AbstractHyperparameterOptimization

    function AdaptiveGaussianProcessMonteCarlo(
            n::Integer,
            initial_design::AbstractMonteCarlo = QuasiMonteCarloSampling(10, LatinHypercubeSample()),
            acquisition_function::AbstractReliabilityAcquisitionFunction = DeviationNumber(),
            stopping::Float64 = 2.0,
            maximum_samples::Integer = 1000;
            σ²::Float64 = 1.0e-10,
            learn_noise::Bool = false,
            learn_hyperparameters::Bool = true,
            mean::AbstractGPs.MeanFunction = ConstMean(0.0),
            kernel::Kernel = SqExponentialKernel(),
            normalize::Bool = true,
            optimizer::AbstractHyperparameterOptimization = MaximumLikelihoodEstimation(Optim.LBFGS(), Optim.Options(; iterations = 100, show_trace = false)),
            candidate_sampling::AbstractMonteCarlo = MonteCarlo(100_000)
        )
        @assert n > 0 "n must be greater than zero"
        return new(n, initial_design, acquisition_function, stopping, maximum_samples, σ², mean, kernel, normalize, candidate_sampling, learn_noise, learn_hyperparameters, optimizer)
    end
end
