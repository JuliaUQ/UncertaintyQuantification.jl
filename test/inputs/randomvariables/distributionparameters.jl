@testitem "Distribution Parameters" begin
    # Beta
    d = Beta(0.5, 0.5)
    μ = mean(d)
    σ = std(d)
    @test [distribution_parameters(μ, σ, Beta)...] ≈ [0.5, 0.5]
    μ = -0.1
    @test_throws ErrorException(
        "provided mean value -0.1 is not compatible with Beta distribution support: (0; 1)"
    ) distribution_parameters(μ, σ, Beta)

    # Gamma
    d = Gamma(3.0, 2.0)
    μ = mean(d)
    σ = std(d)
    @test [distribution_parameters(μ, σ, Gamma)...] ≈ [3.0, 2.0]

    # Gumbel
    d = Gumbel(1.0, 2.0)
    μ = mean(d)
    σ = std(d)
    @test [distribution_parameters(μ, σ, Gumbel)...] ≈ [1.0, 2.0]

    # Logistic
    d = Logistic(5.0, 2.0)
    μ = mean(d)
    σ = std(d)
    @test [distribution_parameters(μ, σ, Logistic)...] ≈ [5.0, 2.0]

    # LogNormal
    d = LogNormal(10.0, 5.0)
    μ = mean(d)
    σ = std(d)
    @test [distribution_parameters(μ, σ, LogNormal)...] ≈ [10.0, 5.0]

    # Rayleigh
    d = Rayleigh(3.0)
    μ = mean(d)
    σ = std(d)
    @test distribution_parameters(μ, Rayleigh) ≈ 3.0
    @test distribution_parameters(σ, Rayleigh; std=true) ≈ 3.0

    # Uniform
    d = Uniform(-1.0, 3.0)
    μ = mean(d)
    σ = std(d)
    @test [distribution_parameters(μ, σ, Uniform)...] ≈ [-1.0, 3.0]
end
