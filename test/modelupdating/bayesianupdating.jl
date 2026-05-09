
@testset "Bayesian Updating" begin
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
        sampler::AbstractBayesianMethod, prior::Beta{Float64}
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

        mcmc_samples, _ = bayesianupdating(logprior, loglikelihood, UQModel[], sampler)

        return mcmc_samples, analytic_mean, sqrt(analytic_var)
    end

    function normalmeanbenchmark(sampler::AbstractBayesianMethod, prior::Normal{Float64})
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

        logprior(df) = logpdf.(prior, df.x)

        mcmc_samples, _ = bayesianupdating(logprior, loglikelihood, UQModel[], sampler)

        return mcmc_samples, analytic_mean, analytic_std
    end

    function normalvarbenchmark(sampler::AbstractBayesianMethod, prior::InverseGamma)
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

        mcmc_samples, _ = bayesianupdating(logprior, loglikelihood, UQModel[], sampler)

        return mcmc_samples, mean(posterior_exact), std(posterior_exact)
    end

    function bivariategaussian(sampler::AbstractBayesianMethod, prior::Uniform)
        analytic_mean = [-0.5, 0.5]
        analytic_cov = [1 0.8; 0.8 1]

        function logprior(df)
            return logpdf.(prior, df.x) .+ logpdf.(prior, df.y)
        end

        function loglikelihood(df)
            return [
                logpdf(MvNormal(analytic_mean, analytic_cov), collect(x)) for
                x in eachrow(df)
            ]
        end

        mcmc_samples, _ = bayesianupdating(logprior, loglikelihood, UQModel[], sampler)

        return mcmc_samples, analytic_mean, analytic_cov
    end

    @testset "Single Component MH binomal inference analytical" begin
        proposal = Normal()
        x0 = (x=0.5,)
        n = 4000
        burnin = 100

        prior = Beta(1, 1)

        mh = SingleComponentMetropolisHastings(proposal, x0, n, burnin)

        mc_samples, analytic_mean, analytic_std = binomialinferencebenchmark(mh, prior)

        @test mean(mc_samples.x) ≈ analytic_mean rtol = 0.1
        @test std(mc_samples.x) ≈ analytic_std rtol = 0.1
    end

    @testset "Single Component MH normal mean analytical" begin
        proposal = Normal()
        x0 = (x=2.5,)
        n = 4000
        burnin = 100

        prior_mean = 2
        prior_std = 10

        prior = Normal(prior_mean, prior_std)

        mh = SingleComponentMetropolisHastings(proposal, x0, n, burnin)

        mc_samples, analytic_mean, analytic_std = normalmeanbenchmark(mh, prior)

        @test mean(mc_samples.x) ≈ analytic_mean rtol = 0.1
        @test std(mc_samples.x) ≈ analytic_std rtol = 0.1
    end

    @testset "Single Component MH normal var analytical" begin
        proposal = Normal()
        x0 = (x=3.0,)
        n = 4000
        burnin = 100

        prior_shape = 30
        prior_scale = 100

        prior = InverseGamma(prior_shape, prior_scale)

        mh = SingleComponentMetropolisHastings(proposal, x0, n, burnin)

        mc_samples, analytic_mean, analytic_std = normalvarbenchmark(mh, prior)

        @test mean(mc_samples.x) ≈ analytic_mean rtol = 0.1
        @test std(mc_samples.x) ≈ analytic_std rtol = 0.1
    end

    @testset "Single Component MH proposal" begin
        proposal = Normal()
        x0 = (x=3.0, y=1.0)
        mh = SingleComponentMetropolisHastings(proposal, x0, 1000, 100)

        @test isa(mh.proposal, Vector{<:UnivariateDistribution})
        @test mh.proposal[1] == proposal
        @test mh.proposal[2] == proposal

        proposal_vec = [Uniform(-1, 1), Normal(0, 0.1)]
        mh_vec = SingleComponentMetropolisHastings(proposal_vec, x0, 1000, 100)
        @test mh_vec.proposal[1] == proposal_vec[1]
        @test mh_vec.proposal[2] == proposal_vec[2]

        @test_throws AssertionError SingleComponentMetropolisHastings(
            proposal_vec, (; x=3.0), 1000, 100
        )
    end

    @testset "Single Component MH bivariategaussian" begin
        n = 50_000
        burnin = 10_000

        x0 = (x=0.0, y=0.0)

        proposal = Normal(0, 0.6)

        mh = SingleComponentMetropolisHastings(proposal, x0, n, burnin)
        mcmc_samples, analytic_mean, analytic_cov = bivariategaussian(mh, Uniform(-10, 10))

        @test mean(mcmc_samples.x) ≈ analytic_mean[1] rtol = 0.1
        @test mean(mcmc_samples.y) ≈ analytic_mean[2] rtol = 0.1

        analytic_std_1 = sqrt(analytic_cov[1, 1])
        analytic_std_2 = sqrt(analytic_cov[2, 2])
        analytic_cor = analytic_cov[1, 2] / (analytic_std_1 * analytic_std_2)

        @test std(mcmc_samples.x) ≈ analytic_std_1 rtol = 0.1
        @test std(mcmc_samples.y) ≈ analytic_std_2 rtol = 0.1
        @test cor(mcmc_samples.x, mcmc_samples.y) ≈ analytic_cor rtol = 0.1
    end

    @testset "TMCMC binomal inference analytical" begin
        n = 4000
        burnin = 100

        prior = Beta(1, 1)

        prior_sample_ = RandomVariable(prior, :x)

        tmcmc = TransitionalMarkovChainMonteCarlo([prior_sample_], n, burnin)

        mc_samples, analytic_mean, analytic_std = binomialinferencebenchmark(tmcmc, prior)

        @test mean(mc_samples.x) ≈ analytic_mean rtol = 0.1
        @test std(mc_samples.x) ≈ analytic_std rtol = 0.1
    end

    @testset "TMCMC normal mean analytical" begin
        n = 4000
        burnin = 100

        prior_mean = 2
        prior_std = 10

        prior = Normal(prior_mean, prior_std)

        prior_sample_ = RandomVariable(prior, :x)

        tmcmc = TransitionalMarkovChainMonteCarlo([prior_sample_], n, burnin)

        mc_samples, analytic_mean, analytic_std = normalmeanbenchmark(tmcmc, prior)

        @test mean(mc_samples.x) ≈ analytic_mean rtol = 0.1
        @test std(mc_samples.x) ≈ analytic_std rtol = 0.1
    end

    @testset "TMCMC normal var analytical" begin
        n = 4000
        burnin = 100

        prior_shape = 30
        prior_scale = 100

        prior = InverseGamma(prior_shape, prior_scale)

        prior_sample_ = RandomVariable(prior, :x)

        tmcmc = TransitionalMarkovChainMonteCarlo([prior_sample_], n, burnin)

        mc_samples, analytic_mean, analytic_std = normalvarbenchmark(tmcmc, prior)

        @test mean(mc_samples.x) ≈ analytic_mean rtol = 0.1
        @test std(mc_samples.x) ≈ analytic_std rtol = 0.1
    end

    @testset "TMCMC bivariategaussian" begin
    n = 10_000
    burnin = 5

    prior_dist = Uniform(-10, 10)

    prior = RandomVariable.(prior_dist, [:x, :y])

    mh = TransitionalMarkovChainMonteCarlo(prior, n, burnin)
    mcmc_samples, analytic_mean, analytic_cov = bivariategaussian(mh, prior_dist)

    @test mean(mcmc_samples.x) ≈ analytic_mean[1] rtol = 0.1
    @test mean(mcmc_samples.y) ≈ analytic_mean[2] rtol = 0.1

    analytic_std_1 = sqrt(analytic_cov[1, 1])
    analytic_std_2 = sqrt(analytic_cov[2, 2])
    analytic_cor = analytic_cov[1, 2] / (analytic_std_1 * analytic_std_2)

    @test std(mcmc_samples.x) ≈ analytic_std_1 rtol = 0.1
    @test std(mcmc_samples.y) ≈ analytic_std_2 rtol = 0.1
    @test cor(mcmc_samples.x, mcmc_samples.y) ≈ analytic_cor rtol = 0.1
end
end
