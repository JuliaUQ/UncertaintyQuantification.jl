struct AdaptiveGaussianProcessMonteCarlo
    n::Integer
    initial_design::AbstractMonteCarlo
    acquisition_function::AbstractGaussianProcessAcquisitionFunction
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
            acquisition_function::AbstractGaussianProcessAcquisitionFunction = DeviationNumber(),
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
        return new(n, initial_design, acquisition_function, stopping, maximum_samples, σ², mean, kernel, normalize, candidate_sampling, learn_noise, learn_hyperparameters, optimizer)
    end
end
