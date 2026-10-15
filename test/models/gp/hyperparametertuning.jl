@testitem "1D GP Hyperparameter Tuning" begin
    x = collect(range(0, stop = 10, length = 5))
    fct(x) = sin.(x) .+ x
    y = fct(x)
    σ² = 1.0e-9
    kernel = SqExponentialKernel() ∘ ScaleTransform(10.0)
    prior_gp = GP(ConstMean(0.0), kernel)
    gp_opt, _ = UncertaintyQuantification._fit_gp(prior_gp, permutedims(x), y; σ² = σ²)
    gp_nonopt, _ = UncertaintyQuantification._fit_gp(prior_gp, permutedims(x), y; σ² = σ², learn_hyperparameters = false)
    x_test = collect(range(0, stop = 10, length = 50))
    y_test = fct(x_test)
    likelihood_no_opt = logpdf(gp_nonopt(x_test), y_test)
    likelihood_opt = logpdf(gp_opt(x_test), y_test)

    @test likelihood_opt > likelihood_no_opt

end

@testitem "2D GP Hyperparameter Tuning" begin
    x = [collect(range(0, stop = 5, length = 10)) collect(range(0, stop = 5, length = 10))]
    y = sin.(x[:, 1]) + cos.(x[:, 2])

    σ² = 1.0e-9
    kernel = Matern52Kernel() ∘ ARDTransform([5.0, 5.0])
    prior_gp = GP(ConstMean(0.0), kernel)
    gp_opt, _ = UncertaintyQuantification._fit_gp(prior_gp, permutedims(x), y; σ² = σ²)
    gp_nonopt, _ = UncertaintyQuantification._fit_gp(prior_gp, permutedims(x), y; σ² = σ², learn_hyperparameters = false)

    x_test = [collect(range(0, stop = 5, length = 50)) collect(range(0, stop = 5, length = 50))]
    y_test = sin.(x_test[:, 1]) + cos.(x_test[:, 2])
    likelihood_no_opt = logpdf(gp_nonopt(permutedims(x_test); obsdim = 2), y_test)
    likelihood_opt = logpdf(gp_opt(permutedims(x_test); obsdim = 2), y_test)
    @test likelihood_opt > likelihood_no_opt
end
