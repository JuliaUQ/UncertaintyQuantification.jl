@testset "TM BMU - Transformations" begin
    prior = RandomVariable.([Uniform(-10, 10), Uniform(-10, 10)], [:x1, :x2])

    loglikelihood = (x1, x2) -> logpdf(Normal(), x1) + logpdf(Normal(), x2 - x1^2)

    logL = ParallelModel(df -> loglikelihood(df.x1, df.x2), :L)

    map = PolynomialMap(2, 2, Normal(), Softplus())
    quadrature = GaussHermiteWeights(3, 2)

    @testset "With density transform" begin
        tm_log = TransportMapBayesian(prior, deepcopy(map), quadrature, true)

        tm_result = bayesianupdating(df -> df.L, [logL], tm_log, nothing, AutoFiniteDiff())
        @test !iszero(getcoefficients(tm_result.map))

        @test tm_result isa TransportMap
        @test length(tm_result.names) == 2

        posterior_samples = sample(tm_result, 10)
        @test nrow(posterior_samples) == 10
    end

    @testset "No density transform" begin
        tm_log = TransportMapBayesian(prior, deepcopy(map), quadrature, true, false)

        tm_result = bayesianupdating(df -> df.L, [logL], tm_log, nothing, AutoFiniteDiff())
        @test !iszero(getcoefficients(tm_result.map))
    end
end

@testset "TM BMU - Benchmark" begin
    N_binom = 15
    data_binom = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1] # p = 0.8

    N_normal = 25
    data_normal_var = [
        3.1170072530410713,
        5.793425513286137,
        3.019644523423345,
        4.198492885643784,
        4.3404411932408005,
        5.338786752835615,
        1.6563692415479587,
        3.496696575234516,
        5.093428444721792,
        5.091321607109399,
        4.994437068835541,
        4.285932709130585,
        1.6751627811539134,
        5.51537861306012,
        2.4998958266295026,
        7.065078073070958,
        5.005937831600564,
        3.9349403146434914,
        3.9340063190556913,
        4.400921498325048,
        5.276032641042012,
        4.034478885301326,
        7.5811254216258215,
        6.468279762464751,
        4.416669261547403,
    ] # μ = 4, σ² = 5

    data_normal_mean = [
        3.2447586269995603,
        7.31570639905421,
        2.9583720841640524,
        2.7006645805494793,
        -5.390670487961957,
        13.222284857030616,
        4.274588755848634,
        6.92516681771173,
        -14.259515860959095,
        2.8938242850671676,
        8.702867817886007,
        17.60466794732554,
        -3.791069613048381,
        3.8733459676840587,
        5.010069312798485,
        -1.8534537575475099,
        10.843101672796852,
        -1.7035437372547804,
        -1.8363450654785087,
        -2.929811781116549,
        -1.6820361524479646,
        -1.5074002658326515,
        1.2143899492909962,
        2.9977426550531416,
        3.9469743716484778,
    ] # μ = 4, σ = 5

    function binomialinferencebenchmark(
        transportmap::TransportMapBayesian, prior::Beta{Float64}
    )
        alpha0 = prior.α
        beta0 = prior.β

        alpha_posterior = alpha0 + sum(data_binom)
        beta_posterior = beta0 + N_binom - sum(data_binom)

        analytic_mean = alpha_posterior / (alpha_posterior + beta_posterior)
        analytic_var =
            alpha_posterior * beta_posterior /
            ((alpha_posterior + beta_posterior)^2 * (alpha_posterior + beta_posterior + 1))

        function loglikelihood(df)
            return [
                sum(logpdf.(Binomial.(N_binom, df_i.x), sum(data_binom))) for
                df_i in eachrow(df)
            ]
        end

        logprior(df) = logpdf.(prior, df.x)

        tm = bayesianupdating(
            loglikelihood, UQModel[], transportmap, nothing, AutoFiniteDiff()
        )

        return tm, analytic_mean, sqrt(analytic_var)
    end

    function normalmeanbenchmark(transportmap::TransportMapBayesian, prior::Normal{Float64})
        std_fixed = 5     # Fixed

        prior_mean = prior.μ
        prior_std = prior.σ

        analytic_std = 1 / ((1 / prior_std^2) + (N_normal / std_fixed^2))
        analytic_mean =
            analytic_std * (prior_mean / prior_std^2 + sum(data_normal_mean) / std_fixed^2)

        function loglikelihood(df)
            return [
                sum(logpdf.(Normal.(df_i.x, std_fixed), data_normal_mean)) for
                df_i in eachrow(df)
            ]
        end

        tm = bayesianupdating(
            loglikelihood, UQModel[], transportmap, nothing, AutoFiniteDiff()
        )

        return tm, analytic_mean, analytic_std
    end

    function normalvarbenchmark(transportmap::TransportMapBayesian, prior::InverseGamma)
        mean_fixed = 4     # Fixed

        prior_shape = prior.invd.α
        prior_scale = prior.θ

        posterior_shape = prior_shape + N_normal / 2
        posterior_scale = prior_scale + sum((data_normal_var .- mean_fixed) .^ 2) / 2

        posterior_exact = InverseGamma(posterior_shape, posterior_scale)

        function loglikelihood(df)
            return [
                sum(logpdf.(Normal.(mean_fixed, sqrt(df_i.x)), data_normal_var)) for
                df_i in eachrow(df)
            ]
        end

        function logprior1(df)  # Required because inverse gamma throws error for negative values
            if df.x < 0
                return -Inf
            end
            return logpdf.(prior, df.x)
        end

        logprior(df) = [logprior1(df_i) for df_i in eachrow(df)]

        tm = bayesianupdating(
            loglikelihood, UQModel[], transportmap, logprior, AutoFiniteDiff()
        )

        return tm, mean(posterior_exact), std(posterior_exact)
    end

    @testset "TM normalmeanbenchmark" begin
        prior = Normal(2, 10)
        prior_RV = RandomVariable(prior, :x)

        tm = PolynomialMap(1, 1)
        quad = GaussHermiteWeights(5, 1)
        map = TransportMapBayesian([prior_RV], tm, quad)

        tm_opt, analytic_mean, analytic_std = normalmeanbenchmark(map, prior)

        tm_mean = only(mean(tm_opt))

        df = sample(tm_opt, 1000)
        tm_mean_samples = mean(df.x)

        @test tm_mean ≈ analytic_mean rtol = 0.05
        @test tm_mean_samples ≈ analytic_mean rtol = 0.05
        @test std(df.x) ≈ analytic_std rtol = 0.05
    end

    @testset "TM binomialinferencebenchmark" begin
        prior = Beta(1, 1)
        prior_RV = RandomVariable(prior, :x)

        tm = PolynomialMap(1, 1)
        quad = GaussHermiteWeights(5, 1)
        map = TransportMapBayesian([prior_RV], tm, quad)

        tm_opt, analytic_mean, analytic_std = binomialinferencebenchmark(map, prior)

        tm_mean = only(mean(tm_opt))

        df = sample(tm_opt, 1000)
        tm_mean_samples = mean(df.x)

        @test tm_mean ≈ analytic_mean rtol = 0.05
        @test tm_mean_samples ≈ analytic_mean rtol = 0.05
        @test std(df.x) ≈ analytic_std rtol = 0.05
    end

    @testset "TM " begin
        prior = InverseGamma(30, 100)
        prior_RV = RandomVariable(prior, :x)

        tm = PolynomialMap(1, 2)
        quad = GaussHermiteWeights(5, 1)
        map = TransportMapBayesian([prior_RV], tm, quad)

        tm_opt, analytic_mean, analytic_std = normalvarbenchmark(map, prior)

        tm_mean = only(mean(tm_opt))

        df = sample(tm_opt, 1000)
        tm_mean_samples = mean(df.x)

        @test tm_mean ≈ analytic_mean rtol = 0.05
        @test tm_mean_samples ≈ analytic_mean rtol = 0.05
        @test std(df.x) ≈ analytic_std rtol = 0.05
    end
end
