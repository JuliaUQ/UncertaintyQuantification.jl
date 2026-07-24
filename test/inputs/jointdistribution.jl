using Random, DataFrames

rv1 = RandomVariable(Exponential(1), :x)
rv2 = RandomVariable(Exponential(1 / 2), :y)

pbox1 = RandomVariable(ProbabilityBox{Exponential}(Interval(1 / 2, 1)), :x)
pbox2 = RandomVariable(ProbabilityBox{Exponential}(Interval(3 / 4, 5 / 4)), :y)

marginals = [rv1, rv2]

imprecise_marginals = [pbox1, pbox2]

copula = GaussianCopula([1 0.8; 0.8 1])

@testset "JointDistribution" begin
    @testset "Copulas" begin
        @testset "Constructor" begin
            @test isa(JointDistribution(copula, marginals), JointDistribution)
            @test isa(JointDistribution(copula, imprecise_marginals), JointDistribution)

            jd = JointDistribution(copula, marginals)
            @test jd.m == marginals
            @test jd.d == copula

            jd = JointDistribution(copula, imprecise_marginals)
            @test jd.m == imprecise_marginals
            @test jd.d == copula

            @test_throws ArgumentError("Dimension mismatch between copula and marginals.") JointDistribution(
                GaussianCopula([1 0 0; 0 1 0; 0 0 1]), marginals
            )
            @test_throws ArgumentError("Dimension mismatch between copula and marginals.") JointDistribution(
                GaussianCopula([1 0 0; 0 1 0; 0 0 1]), imprecise_marginals
            )
            @test_throws ArgumentError("Marginal names must be unique.") JointDistribution(
                copula, [rv1, rv1]
            )
        end

        @testset "sample" begin
            jd = JointDistribution(copula, marginals)
            @test size(sample(jd, 10)) == (10, 2)
            @test size(sample(jd)) == (1, 2)
            jd = JointDistribution(copula, imprecise_marginals)
            @test size(sample(jd, 10)) == (10, 2)
            @test size(sample(jd)) == (1, 2)

            conditioning_df = DataFrame(:x => [0.25, 0.5])
            conditioned_samples = sample(JointDistribution(copula, marginals), conditioning_df)

            @test size(conditioned_samples) == (2, 2)
            @test conditioned_samples.x == conditioning_df.x
        end

        @testset "sample with mixed columns" begin
            jd = JointDistribution(copula, marginals)
            
            mixed_df = DataFrame(:x => [0.25, 0.5], :a => [1.0, 2.0], :b => [10, 20])
            
            conditioned_samples = sample(jd, mixed_df[:, [:x, :a, :b]])
            
            @test size(conditioned_samples) == (2, 4)  # x, y (resampled), a, b
            @test Set(Symbol.(names(conditioned_samples))) == Set([:x, :y, :a, :b])
            
            @test conditioned_samples.x == mixed_df.x
            
            @test conditioned_samples.a == mixed_df.a
            @test conditioned_samples.b == mixed_df.b
            
            @test eltype(conditioned_samples.y) == Float64
            @test all(isfinite.(conditioned_samples.y))
        end

        @testset "condition" begin
            jd = JointDistribution(copula, marginals)
            conditioning_df = DataFrame(:x => [0.25, 0.5])
            inputs = UncertaintyQuantification.condition(jd, conditioning_df[1, :])
        
            @test length(inputs) == 2
            @test inputs[1] isa JointDistribution
            @test inputs[2] isa Parameter
            @test inputs[2].name == :x
            @test inputs[2].value == 0.25
            @test names(inputs[1]) == [:y]
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

            jd = JointDistribution(copula, imprecise_marginals)

            samples = sample(jd, 10^6)

            @test eltype(samples.x) == Interval
            @test eltype(samples.y) == Interval

            to_standard_normal_space!(jd, samples)

            @test eltype(samples.x) <: Real
            @test eltype(samples.y) <: Real

            @test isapprox(abs(mean(samples.x)), 0.0; atol=0.01)
            @test isapprox(abs(mean(samples.y)), 0.0; atol=0.01)
            @test cor(samples.x, samples.y) ≈ 0.0 atol = 0.01
        end

        @testset "densities" begin
            jd = JointDistribution(
                copula,
                [RandomVariable(Uniform(-1.0, 1.0), :x), RandomVariable(Uniform(), :y)],
            )

            @test hcubature(x -> pdf(jd, x), [-1.0, 0.0], [0.5, 0.5])[1] ≈
                cdf(jd, [0.5, 0.5]) atol = 1e-4
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

            @test_throws ArgumentError("Dimension mismatch between distribution and names.") JointDistribution(
                dist, [:x, :y, :z]
            )
            @test_throws ArgumentError("Marginal names must be unique.") JointDistribution(
                dist, [:x, :x]
            )
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
            @test_throws ["Cannot map", "to standard normal space."] to_standard_normal_space!(
                jd, samples
            )
            @test_throws ["Cannot map", "to physical space."] to_physical_space!(
                jd, samples
            )
        end
    end
end
