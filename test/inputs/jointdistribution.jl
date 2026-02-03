using Random, DataFrames

rv1 = RandomVariable(Exponential(1), :x)
rv2 = RandomVariable(Exponential(1 / 2), :y)

marginals = [rv1, rv2]
copula = GaussianCopula([1 0.8; 0.8 1])

@testset "JointDistribution" begin
    @testset "Copulas" begin
        @testset "Constructor" begin
            @test isa(JointDistribution(copula, marginals), JointDistribution)

            jd = JointDistribution(copula, marginals)
            @test jd.m == marginals
            @test jd.d == copula

            @test_throws ArgumentError("Dimension mismatch between copula and marginals.") JointDistribution(
                GaussianCopula([1 0 0; 0 1 0; 0 0 1]), marginals
            )
        end

        @testset "sample" begin
            jd = JointDistribution(copula, marginals)
            @test size(sample(jd, 10)) == (10, 2)
            @test size(sample(jd)) == (1, 2)
        end

        @testset "names" begin
            jd = JointDistribution(copula, marginals)
            @test names(jd) == [:x, :y]
        end

        @testset "mean" begin
            jd = JointDistribution(copula, marginals)
            @test mean(jd) == [1.0, 0.5]
        end

        @testset "to_standard_normal_space" begin
            jd = JointDistribution(copula, marginals)

            samples = sample(jd, 10^6)

            @test isapprox(mean(samples.x), 1.0; atol=0.01)
            @test isapprox(mean(samples.y), 0.5; atol=0.01)

            @test cor(samples.x, samples.y) ≈ 0.77 atol = 0.01

            to_standard_normal_space!(jd, samples)

            @test isapprox(abs(mean(samples.x)), 0.0; atol=0.01)
            @test isapprox(abs(mean(samples.y)), 0.0; atol=0.01)

            @test isapprox(std(samples.x), 1.0; atol=0.01)
            @test isapprox(std(samples.y), 1.0; atol=0.01)

            @test cor(samples.x, samples.y) ≈ 0.0 atol = 0.01
        end

        @testset "to_physical_space" begin
            jd = JointDistribution(copula, marginals)

            samples = DataFrame(:x => rand(Normal(), 10^5), :y => rand(Normal(), 10^5))

            to_physical_space!(jd, samples)

            @test isapprox(mean(samples.x), 1.0; atol=0.01)
            @test isapprox(mean(samples.y), 0.5; atol=0.01)

            @test round(cor(samples.x, samples.y); digits=2) == 0.77
        end
    end

    @testset "MultivariateDistribution" begin
        dist = MvNormal([1.0 0.71; 0.71 1.0])
        m = [:x, :y]
        jd = JointDistribution(dist, m)
        @testset "Constructor" begin
            @test jd.d == dist
            @test jd.m == m

            @test_throws ArgumentError(
                "Dimension mismatch between distribution and names."
            ) JointDistribution(dist, [:x, :y, :z])
        end

        @testset "sample" begin
            samples = sample(jd, 10^6)

            @test size(samples) == (10^6, 2)

            @test cor(samples.x, samples.y) ≈ 0.71 atol = 0.01
        end

        @testset "functions" begin
            @test mean(jd) == [0.0, 0.0]
            @test dimensions(jd) == 2
            @test names(jd) == [:x, :y]

            samples = sample(jd, 10^6)
            @test_throws ErrorException(
                "Cannot map ZeroMeanFullNormal{Tuple{Base.OneTo{Int64}}} to standard normal space.",
            ) to_standard_normal_space!(jd, samples)
            @test_throws ErrorException(
                "Cannot map ZeroMeanFullNormal{Tuple{Base.OneTo{Int64}}} to physical space."
            ) to_physical_space!(jd, samples)
        end
    end
end
