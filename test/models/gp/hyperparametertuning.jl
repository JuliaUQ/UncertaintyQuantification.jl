@testset "GaussianProcessRegressionHyperparameterTuning" begin

    @testset "OneDimensionalInput" begin
        x = collect(range(0, stop=5, length=10))
        y = sin.(x)
        data = DataFrame(:x => x, :y => y)

        σ² = 1e-5
        kernel = SqExponentialKernel() ∘ ScaleTransform(10.0)

        gp = with_gaussian_noise(GP(0.0, kernel), σ²)
        gpr_no_opt = GaussianProcess(
            gp,
            data,
            :y;
            optimization=NoHyperparameterOptimization()
        )

        gpr_opt = GaussianProcess(
            gp,
            data,
            :y;
            optimization=MaximumLikelihoodEstimation()
        )
        
        likelihood_no_opt = logpdf(gpr_no_opt.gp(x), y)
        likelihood_opt = logpdf(gpr_opt.gp(x), y)
        
        @test likelihood_opt > likelihood_no_opt
    end
    
    @testset "TwoDimensionalInput" begin
        x = [collect(range(0, stop=5, length=10)) collect(range(0, stop=5, length=10))]
        y = sin.(x[:, 1]) + cos.(x[:, 2])
        data = DataFrame(:x1 => x[:, 1], :x2 => x[:, 2], :y => y)

        σ² = 1e-5
        kernel = SqExponentialKernel() ∘ ARDTransform([5.0, 5.0])

        gp = with_gaussian_noise(GP(0.0, kernel), σ²)
        gpr_no_opt = GaussianProcess(
            gp,
            data,
            :y;
            optimization=NoHyperparameterOptimization()
        )

        gpr_opt = GaussianProcess(
            gp,
            data,
            :y;
            optimization=MaximumLikelihoodEstimation()
        )

        likelihood_no_opt = logpdf(gpr_no_opt.gp(RowVecs(x)), y)
        likelihood_opt = logpdf(gpr_opt.gp(RowVecs(x)), y)

        @test likelihood_opt > likelihood_no_opt
    end
end