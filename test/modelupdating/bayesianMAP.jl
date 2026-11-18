@testitem "MAP and MLE have different targets" begin
    using DataFrames, Optim, Statistics

    data = [1.0, 3.0, 5.0]
    prior = JointDistribution([RandomVariable(Normal(), :x)])
    expected = sum(data) / (4 + length(data))
    for islog in (true, false)
        calls = Ref(0)
        likelihood_calls = Ref(0)
        model = Model(
            df -> begin
                calls[] += 1
                copy(df.x)
            end, :response
        )
        loglikelihood = df -> begin
            likelihood_calls[] += 1
            [sum(logpdf.(Normal(x, 2), data)) for x in df.response]
        end
        likelihood = islog ? loglikelihood : df -> exp.(loglikelihood(df))
        starts = DataFrame(x = [0.0, 3.0])
        estimate = MaximumAPosterioriBayesian(starts; islog = islog)
        result = bayesianupdating(prior, likelihood, estimate, [model])
        @test nrow(result) == 1
        @test result.x[1] ≈ expected atol = 1.0e-5
        @test calls[] == likelihood_calls[]
        @test starts == DataFrame(x = [0.0, 3.0])
        expected_value = logpdf(Normal(), expected) + sum(logpdf.(Normal(expected, 2), data))
        @test result[1, islog ? :logMAP : :MAP] ≈ (islog ? expected_value : exp(expected_value)) atol = 1.0e-9
        mle = maximumlikelihood(likelihood, MaximumLikelihoodEstimate(starts; islog = islog), [model])
        @test mle.x[1] ≈ mean(data) atol = 1.0e-5
    end
end

@testitem "MAP bounded binomial and multivariate estimates" begin
    using DataFrames, Optim

    prior = JointDistribution([RandomVariable(Beta(2, 3), :p)])
    estimate = MaximumAPosterioriBayesian(DataFrame(p = [0.5]); lowerbounds = [0.0], upperbounds = [1.0])
    result = bayesianupdating(prior, df -> logpdf.(Binomial.(15, df.p), 10), estimate)
    @test result.p[1] ≈ 11 / 18 atol = 1.0e-5

    target = MvNormal([1.0, -2.0], [1.0 0.3; 0.3 2.0])
    prior = JointDistribution([RandomVariable(Normal(0, 5), :x), RandomVariable(Normal(0, 5), :y)])
    estimate = MaximumAPosterioriBayesian(DataFrame(y = [0.0], x = [0.0]))
    likelihood = df -> [logpdf(target, [row.x, row.y]) for row in eachrow(df)]
    result = bayesianupdating(prior, likelihood, estimate)
    covariance = inv(inv(cov(target)) + [1 / 25 0.0; 0.0 1 / 25])
    μ = covariance * inv(cov(target)) * mean(target)
    @test names(result) == ["y", "x", "logMAP"]
    @test [result.y[1], result.x[1]] ≈ reverse(μ) atol = 1.0e-5
    @test_throws ArgumentError bayesianupdating(prior, likelihood, MaximumAPosterioriBayesian(DataFrame(z = [0.0])))

    prior = JointDistribution([RandomVariable(Normal(), :x)])
    estimate = MaximumAPosterioriBayesian(
        DataFrame(x = [5.0]); optimizer = Optim.NelderMead(),
        optimoptions = Optim.Options(; iterations = 0)
    )
    result = bayesianupdating(prior, df -> zeros(nrow(df)), estimate)
    @test result.x[1] > 4
end

@testitem "MAP and MLE retain distinct modes" begin
    using DataFrames

    target = MixtureModel([MvNormal([-3.0], [0.1;;]), MvNormal([3.0], [0.1;;])], [0.2, 0.8])
    likelihood = df -> [logpdf(target, [x]) for x in df.x]
    starts = DataFrame(x = [-3.0, 3.0, 3.0])
    prior = JointDistribution([RandomVariable(Uniform(-5, 5), :x)])
    map = bayesianupdating(prior, likelihood, MaximumAPosterioriBayesian(starts))
    mle = maximumlikelihood(likelihood, MaximumLikelihoodEstimate(starts))
    @test map.x ≈ [-3.0, 3.0] atol = 1.0e-5
    @test mle.x ≈ map.x atol = 1.0e-5
    @test map.logMAP ≈ mle.logMLE .- log(10) atol = 1.0e-5
end
