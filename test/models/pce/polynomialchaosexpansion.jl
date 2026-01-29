@testset "PolynomialChaosExpansion" begin
    x = RandomVariable.([Uniform(-2, 0), Normal(-1, 0.5), Uniform(0, 1)], [:x1, :x2, :x3])

    model1 = Model(df -> begin
        return df.x1 .+ df.x2 .* df.x3
    end, :y1)

    model2 = Model(df -> begin
        return df.y1
    end, :y)

    model = [model1, model2]

    p = 8
    Ψ = PolynomialChaosBasis([LegendreBasis(), HermiteBasis(), LegendreBasis()], p)

    @testset "LeastSquares" begin
        ls = LeastSquares(SobolSampling(1000))
        pce, samples, mse = polynomialchaos(x, model, Ψ, :y, ls)

        new_samples = samples[:, Not(:y1, :y)]
        evaluate!(pce, new_samples)
        ϵ = mean((new_samples.y .- samples.y) .^ 2)

        @test mean(pce) ≈ -1.5 rtol = 1e-10
        @test var(pce) ≈ 0.5 rtol = 1e-10
        @test mse ≈ ϵ atol = eps()
    end

    @testset "WeightedApproximateFetekePoints" begin
        wafp = WeightedApproximateFetekePoints(SobolSampling(1000))
        pce, samples, mse = polynomialchaos(x, model, Ψ, :y, wafp)

        new_samples = samples[:, Not(:y1, :y)]
        evaluate!(pce, new_samples)
        ϵ = mean((new_samples.y .- samples.y) .^ 2)

        @test mean(pce) ≈ -1.5 rtol = 1e-10
        @test var(pce) ≈ 0.5 rtol = 1e-10
        @test mse ≈ ϵ atol = eps()
    end

    @testset "GaussQuadrature" begin
        gq = GaussQuadrature()
        pce, _ = polynomialchaos(x, model, Ψ, :y, gq)

        @test mean(pce) ≈ -1.5 rtol = 1e-10
        @test var(pce) ≈ 0.5 rtol = 1e-10
    end

    @testset "MultipleOutputs" begin
        x1 = RandomVariable(Uniform(-2, 0), :x1)
        x2 = RandomVariable(Uniform(-2, 0), :x2)

        model_a = Model(df -> begin
            return df.x1 .^ 2
        end, :ya)

        model_b = Model(df -> begin
            return df.ya .* 2
        end, :yb)

        Ψ0 = PolynomialChaosBasis([LegendreBasis()], p)

        ls = LeastSquares(SobolSampling(1000))
        wafp = WeightedApproximateFetekePoints(SobolSampling(1000))
        gq = GaussQuadrature()

        pces_ls, _, mses_ls = polynomialchaos(x1, [model_a, model_b], Ψ0, [:ya,:yb], ls)
        pces_wafp, _, mses_wafp = polynomialchaos(x1, [model_a, model_b], Ψ0, [:ya,:yb], wafp)
        pces_gq, _ = polynomialchaos(x1, [model_a, model_b], Ψ0, [:ya,:yb], gq)

        @test isa(pces_ls, Vector{PolynomialChaosExpansion})
        @test isa(mses_ls, Vector{<:Real})
        @test isa(pces_wafp, Vector{PolynomialChaosExpansion})
        @test isa(mses_wafp, Vector{<:Real})
        @test isa(pces_gq, Vector{PolynomialChaosExpansion})

        @test mean(pces_ls[1]) ≈ 4/3 rtol = 1e-10
        @test mean(pces_ls[2]) ≈ 8/3 rtol = 1e-10
        @test mean(pces_wafp[1]) ≈ 4/3 rtol = 1e-10
        @test mean(pces_wafp[2]) ≈ 8/3 rtol = 1e-10
        @test mean(pces_gq[1]) ≈ 4/3 rtol = 1e-10
        @test mean(pces_ls[2]) ≈ 8/3 rtol = 1e-10
    end

    @testset "Convenience Functions" begin
        x1 = RandomVariable(Uniform(-2, 0), :x1)
        x2 = RandomVariable(Uniform(-2, 0), :x2)

        model_a = Model(df -> begin
            return df.x1 .^ 2
        end, :ya)

        model_b = Model(df -> begin
            return df.ya .* 2
        end, :yb)

        Ψ1 = PolynomialChaosBasis([LegendreBasis()], p)
        Ψ2 = PolynomialChaosBasis([LegendreBasis(), LegendreBasis()], p)

        ls = LeastSquares(SobolSampling(1000))
        gq = GaussQuadrature()

        pce_ls_11, _, _ = polynomialchaos(x1, model_a, Ψ1, :ya, ls)
        pce_gq_11, _ = polynomialchaos(x1, model_a, Ψ1, :ya, gq)

        pce_ls_12, _, _ = polynomialchaos(x1, [model_a, model_b], Ψ1, :yb, ls)
        pce_gq_12, _ = polynomialchaos(x1, [model_a, model_b], Ψ1, :yb, gq)

        pce_ls_21, _, _ = polynomialchaos([x1, x2], model_a, Ψ2, :ya, ls)
        pce_gq_21, _ = polynomialchaos([x1, x2], model_a, Ψ2, :ya, gq)

        @test isa(pce_ls_11, PolynomialChaosExpansion)
        @test isa(pce_gq_11, PolynomialChaosExpansion)
        @test isa(pce_ls_12, PolynomialChaosExpansion)
        @test isa(pce_gq_12, PolynomialChaosExpansion)
        @test isa(pce_ls_21, PolynomialChaosExpansion)
        @test isa(pce_gq_21, PolynomialChaosExpansion)
    end

    @testset "evaluate" begin
        gq = GaussQuadrature()
        pce, samples = polynomialchaos(x, model, Ψ, :y, gq)

        data = copy(samples)
        evaluate!(pce, data)

        @test sum((samples.y .- data.y) .^ 2) ≈ 0.0 atol = 0.01
    end

    @testset "sample" begin
        gq = GaussQuadrature()
        pce, _ = polynomialchaos(x, model, Ψ, :y, gq)

        samples = sample(pce, 100)
        data = copy(samples)

        evaluate!(model, data)

        @test sum((samples.y .- data.y) .^ 2) ≈ 0.0 atol = 0.01
    end
end
