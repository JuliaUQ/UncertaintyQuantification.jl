@testset "GP Acquisition Functions" begin
    # Small 1D training set: y = x^2 * sin(x)
    input = RandomVariable(Uniform(-2, 12), :x1)
    n_input_samples = 10
    design = LatinHypercubeSampling(n_input_samples)
    model = Model(df -> df.x1 .^ 2 .* sin.(df.x1), :y)

    data = sample(input, design)
    evaluate!(model, data)

    gp = GaussianProcess(
        data, :y;
        mean_fct = ConstMean(0.0),
        kernel = SqExponentialKernel(),
        learn_hyperparameters = true,
    )

    candidates = DataFrame(x1 = collect(range(-2, 12, 25)))

    function mean_and_std(gp, candidates)
        df = copy(candidates)
        evaluate!(gp, df; mode = :mean_and_var)
        return df[:, :y_mean], sqrt.(df[:, :y_var])
    end

    @testset "MaximumVariance" begin
        μ, σ = mean_and_std(gp, candidates)
        expected = candidates[[argmax(σ .^ 2)], [:x1]]

        result = UncertaintyQuantification._find_next_point(gp, candidates, MaximumVariance())
        @test result == expected
    end

    @testset "ExpectedImprovement" begin
        ei = ExpectedImprovement(; ξ = 0.0)
        μ, σ = mean_and_std(gp, candidates)
        f_best = minimum(data.y)

        z = (f_best .- μ) ./ σ
        ei_values = (f_best .- μ) .* cdf.(Normal(), z) .+ σ .* pdf.(Normal(), z)
        expected = candidates[[argmax(ei_values)], [:x1]]

        result = UncertaintyQuantification._find_next_point(gp, candidates, ei)
        @test result == expected
    end

    @testset "ProbabilityOfImprovement" begin
        poi = ProbabilityOfImprovement(; ξ = 0.0)
        μ, σ = mean_and_std(gp, candidates)
        f_best = minimum(data.y)

        z = (f_best .- μ) ./ σ
        pi_values = cdf.(Normal(), z)
        expected = candidates[[argmax(pi_values)], [:x1]]

        result = UncertaintyQuantification._find_next_point(gp, candidates, poi)
        @test result == expected
    end

    @testset "UpperConfidenceBound" begin
        ucb = UpperConfidenceBound(; κ = 2.0)
        μ, σ = mean_and_std(gp, candidates)

        lcb = μ .- ucb.κ .* σ
        expected = candidates[[argmin(lcb)], [:x1]]

        result = UncertaintyQuantification._find_next_point(gp, candidates, ucb)
        @test result == expected

        # larger κ should not decrease the selected lower confidence bound value
        ucb_explore = UpperConfidenceBound(; κ = 10.0)
        result_explore = UncertaintyQuantification._find_next_point(gp, candidates, ucb_explore)
        @test result_explore isa DataFrame
    end

    @testset "DeviationNumber" begin
        dn = DeviationNumber(; threshold = 0.0, stopping = 2.0)
        μ, σ = mean_and_std(gp, candidates)

        u = abs.(μ .- dn.threshold) ./ σ
        expected = candidates[[argmin(u)], [:x1]]

        result = UncertaintyQuantification._find_next_point(gp, candidates, dn)
        @test result == expected

        # stopping criterion is true once min(U) exceeds the threshold
        _, converged = UncertaintyQuantification._find_next_point_stopping(gp, candidates, dn)
        @test converged == (minimum(u) >= dn.stopping)
    end

    @testset "ExpectedFeasibility" begin
        eff = ExpectedFeasibility(; threshold = 0.0, epsilon_factor = 2.0, stopping = 0.001)
        μ, σ = mean_and_std(gp, candidates)
        ε = eff.epsilon_factor .* σ

        z = (eff.threshold .- μ) ./ σ
        z_lower = (eff.threshold .- ε .- μ) ./ σ
        z_upper = (eff.threshold .+ ε .- μ) ./ σ

        eff_values =
            (μ .- eff.threshold) .*
            (2 .* cdf.(Normal(), z) .- cdf.(Normal(), z_lower) .- cdf.(Normal(), z_upper)) .-
            σ .* (2 .* pdf.(Normal(), z) .- pdf.(Normal(), z_lower) .- pdf.(Normal(), z_upper)) .+
            ε .* (cdf.(Normal(), z_upper) .- cdf.(Normal(), z_lower))

        expected = candidates[[argmax(eff_values)], [:x1]]

        result = UncertaintyQuantification._find_next_point(gp, candidates, eff)
        @test result == expected

        _, converged = UncertaintyQuantification._find_next_point_stopping(gp, candidates, eff)
        @test converged == (maximum(eff_values) <= eff.stopping)
    end

    @testset "MaximinDistance" begin
        # doesn't depend on the GP posterior: purely geometric
        X = Matrix(data[:, [:x1]])
        Xc = Matrix(candidates[:, [:x1]])
        distances = [minimum(norm(Xc[i, :] - X[j, :]) for j in axes(X, 1)) for i in axes(Xc, 1)]
        expected = candidates[[argmax(distances)], [:x1]]

        result = UncertaintyQuantification._find_next_point(gp, candidates, MaximinDistance())
        @test result == expected
    end

    @testset "ExpectedImprovementForGlobalFit" begin
        μ, σ = mean_and_std(gp, candidates)
        σ² = σ .^ 2

        X = Matrix(data[:, [:x1]])
        Xc = Matrix(candidates[:, [:x1]])
        y = data.y

        nearest = [argmin([norm(Xc[i, :] - X[j, :]) for j in axes(X, 1)]) for i in axes(Xc, 1)]
        eigf = abs2.(μ .- y[nearest]) .+ σ²
        expected = candidates[[argmax(eigf)], [:x1]]

        result = UncertaintyQuantification._find_next_point(gp, candidates, ExpectedImprovementForGlobalFit())
        @test result == expected
    end
end
