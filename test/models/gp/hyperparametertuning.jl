@testitem "GaussianProcess 1D Hyperparameter Tuning" setup = [
    GPTestSetup,
] begin
    x = RandomVariable(Uniform(0, 10), :x)
    m = Model(df -> sin.(df.x) .+ df.x, :y)

    mean = ConstMean(0.0)
    kernel = SqExponentialKernel() ∘ ScaleTransform(10.0)
    σ² = 1.0e-9
    gp_opt = GaussianProcess(x, m, QuasiMonteCarloSampling(10, HaltonSample()), :y; mean = mean, kernel = kernel)
    gp_no_opt = GaussianProcess(x, m, QuasiMonteCarloSampling(10, HaltonSample()), :y; mean = mean, kernel = kernel, learn_hyperparameters = false)
    test = sample(x, QuasiMonteCarloSampling(20, LatinHypercubeSample()))
    evaluate!(m, test)
    likelihood_opt = logpdf(gp_opt, test)
    likelihood_no_opt = logpdf(gp_no_opt, test)

    @test likelihood_opt > likelihood_no_opt
end

@testitem "Gaussian Process 2DHyperparameter Tuning" setup = [
    GPTestSetup,
] begin
    x = RandomVariable.(Uniform(0, 5), [:x1, :x2])
    m = Model(df -> sin.(df.x1) .+ cos.(df.x2), :y)

    samples = sample(x, QuasiMonteCarloSampling(100, HaltonSample()))
    data = copy(samples[1:20, :])
    evaluate!(m, data)

    mean = ConstMean(0.0)
    kernel = Matern52Kernel() ∘ ARDTransform([5.0, 5.0])
    σ² = 1.0e-9
    gp_opt = GaussianProcess(data, :y; mean = mean, kernel = kernel)
    gp_no_opt = GaussianProcess(data, :y; mean = mean, kernel = kernel, learn_hyperparameters = false)
    test = copy(samples[21:end, :])
    evaluate!(m, test)
    likelihood_opt = logpdf(gp_opt, test)
    likelihood_no_opt = logpdf(gp_no_opt, test)

    @test likelihood_opt > likelihood_no_opt
end
