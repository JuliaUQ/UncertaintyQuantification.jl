@testset "Hyperparameter tuning" begin

    @testset "1D Input" begin
        x = collect(range(0, stop=10, length=5))
        fct(x) = sin.(x) .+ x
        y = fct(x)
        data = DataFrame(:x => x, :y => y)

        σ² = 0.0
        kernel = SqExponentialKernel() ∘ ScaleTransform(10.0)
        prior_gp = GP(ConstMean(0.0), kernel)
        gp_opt = GaussianProcess(prior_gp, data, :y; σ²=σ²)
        gp_nonopt = GaussianProcess(prior_gp, data, :y; σ²=σ², learn_hyperparameters=false)
        x_test = collect(range(0, stop=10, length=50))
        y_test = fct(x_test)
        likelihood_no_opt = logpdf(gp_nonopt.posterior(x_test), y_test)
        likelihood_opt = logpdf(gp_opt.posterior(x_test), y_test)
        
        @test likelihood_opt > likelihood_no_opt
    end
    
    @testset "2D Input" begin
        x = [collect(range(0, stop=5, length=10)) collect(range(0, stop=5, length=10))]
        y = sin.(x[:, 1]) + cos.(x[:, 2])
        data = DataFrame(:x1 => x[:, 1], :x2 => x[:, 2], :y => y)

        σ² = 0.0
        kernel = Matern52Kernel() ∘ ARDTransform([5.0, 5.0])
        prior_gp = GP(ConstMean(0.0), kernel)
        gp_opt = GaussianProcess(prior_gp, data, :y; σ²=σ²)
        gp_nonopt = GaussianProcess(prior_gp, data, :y; σ²=σ², learn_hyperparameters=false)

        x_test = [collect(range(0, stop=5, length=50)) collect(range(0, stop=5, length=50))]
        y_test = sin.(x_test[:, 1]) + cos.(x_test[:, 2])
        likelihood_no_opt = logpdf(gp_nonopt.posterior(RowVecs(x_test)), y_test)
        likelihood_opt = logpdf(gp_opt.posterior(RowVecs(x_test)), y_test)
        @test likelihood_opt > likelihood_no_opt
    end
end