@testset "Transport Maps" begin

    @testset "TransportMap from density" begin
        # Define Target
        μ = [1.0, 2.0]
        Σ = [1.0 0.5; 0.5 2.0]
        target_dist = MvNormal(μ, Σ)

        log_dens = x -> logpdf(target_dist, x)
        grad_log_dens = x -> -Σ \ (x - μ)

        target = MapTargetDensity(log_dens, grad_log_dens)

        map = PolynomialMap(2, 1)
        quadrature = GaussHermiteWeights(3, 2)
        var_names = [:x1, :x2]

        tm = mapfromdensity(map, target, quadrature, var_names, nothing)

        @test tm isa TransportMap
        @test length(tm.names) == 2
        @test tm.names == var_names
        @test tm.target == target
        @test tm.quadrature == quadrature

        samples = sample(tm, 100)
        @test nrow(samples) == 100
        @test all(names(samples) .== ["x1", "x2"])

        # Test pdf and logpdf
        x_test = [1.0, 2.0]
        p = pdf(tm, x_test)
        @test isapprox(p, pdf(target_dist, x_test); atol=1e-8)

        X_test = hcat(samples.x1, samples.x2)
        p_vec = pdf(tm, X_test)
        @test length(p_vec) == 100
        @test all(p_vec .> 0)

        lp = logpdf(tm, x_test)
        @test lp isa Float64
        @test lp ≈ log(p)

        # Test transformations
        Z = DataFrame(; x1=randn(10), x2=randn(10))
        Z_copy = copy(Z)
        to_physical_space!(tm, Z)

        X = copy(Z)
        to_standard_normal_space!(tm, X)
        @test isapprox(Matrix(X), Matrix(Z_copy); atol=1e-8)

        Z_diag = DataFrame(; x1=randn(10), x2=randn(10))
        vd = variancediagnostic(tm, Z_diag)
        @test isapprox(vd, 0; atol=1e-8)

        # Test show methods
        io = IOBuffer()
        show(io, tm)
        @test !isempty(String(take!(io)))

        show(io, MIME("text/plain"), tm)
        @test !isempty(String(take!(io)))
    end

    @testset "TransportMap from samples" begin
        # Generate samples from a known distribution
        μ = [0.5, 1.5]
        Σ = [1.5 0.3; 0.3 1.0]
        target_dist = MvNormal(μ, Σ)
        X_samples = rand(target_dist, 500)'
        X_df = DataFrame(; x1=X_samples[:, 1], x2=X_samples[:, 2])

        # Create transport map from samples
        map = PolynomialMap(2, 1)
        tm_samples = mapfromsamples(map, X_df)

        @test tm_samples isa TransportMapFromSamples
        @test length(tm_samples.names) == 2
        @test tm_samples.names == [:x1, :x2]
        @test nrow(tm_samples.samples) == 500

        # Test sampling
        new_samples = sample(tm_samples, 50)
        @test nrow(new_samples) == 50
        @test all(names(new_samples) .== ["x1", "x2"])

        # Test pdf evaluation
        x_test = [0.5, 1.5]
        p = pdf(tm_samples, x_test)
        @test p isa Float64
        @test p > 0

        # Test pdf with matrix
        X_test = X_samples[1:10, :]
        p_vec = pdf(tm_samples, X_test)
        @test length(p_vec) == 10
        @test all(p_vec .> 0)

        # Test logpdf
        lp = logpdf(tm_samples, x_test)
        @test lp isa Float64
        @test lp ≈ log(p)

        # Test transformations
        Z = DataFrame(; x1=randn(10), x2=randn(10))
        Z_copy = copy(Z)
        to_physical_space!(tm_samples, Z)

        X = copy(Z)
        to_standard_normal_space!(tm_samples, X)
        @test isapprox(Matrix(X), Matrix(Z_copy); atol=1e-6)

       # Test show methods
        io = IOBuffer()
        show(io, tm_samples)
        @test !isempty(String(take!(io)))

        show(io, MIME("text/plain"), tm_samples)
        @test !isempty(String(take!(io)))
    end

    @testset "TransportMap with transform_density" begin
        # Prior to transform density
        prior = RandomVariable.([Uniform(0, 5), Uniform(-2, 3)], [:x1, :x2])

        # Target in standard normal space
        target_density = x -> -0.5 * sum(x .^ 2)
        grad_target_density = x -> -x

        target = MapTargetDensity(target_density, grad_target_density)

        map = PolynomialMap(2, 1)
        quadrature = GaussHermiteWeights(3, 2)

        tm = mapfromdensity(map, target, quadrature, names(prior), prior)

        @test !isnothing(tm.transform_density)
        @test length(tm.transform_density) == 2

        # Test sampling in target space
        samples = sample(tm, 100)
        @test all(0 .<= samples.x1 .<= 5)
        @test all(-2 .<= samples.x2 .<= 3)
    end
end
