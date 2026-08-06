@testset "GaussianMixtureModel" begin
    # Test fitting with a DataFrame and initialization
    df = DataFrame(x1=randn(100), x2=randn(100))
    gmm = GaussianMixtureModel(df, 3; maximum_iterations=1, tolerance=1e-3)

    @test gmm isa JointDistribution
    @test length(gmm.m) == 2
    @test length(gmm.d.components) == 3
    @test dimensions(gmm) == 2
    @test all(names(gmm) .== [:x1, :x2])

    # Test sample generation
    samples = sample(gmm, 10)
    @test nrow(samples) == 10
    @test all(names(samples) .== ["x1", "x2"])

    # Test properties after fitting
    @test all(mean(gmm) .== mean(gmm.d))
    @test all(var(gmm) .== var(gmm.d))
    @test all(minimum(gmm) == minimum(gmm.d))
    @test all(maximum(gmm) == maximum(gmm.d))
    @test insupport(gmm, [0.0, 0.0]) == insupport(gmm.d, [0.0, 0.0])
    @test pdf(gmm, [0.0, 0.0]) == pdf(gmm.d, [0.0, 0.0])
    @test logpdf(gmm, [0.0, 0.0]) == logpdf(gmm.d, [0.0, 0.0])
|
    # Test error handling
    @test_throws ErrorException to_standard_normal_space!(gmm, samples)
    @test_throws ArgumentError GaussianMixtureModel(DataFrame(x1=randn(100)), 3)  # Mismatched dimensions
    @test_throws ArgumentError GaussianMixtureModel(df, 0)  
end
