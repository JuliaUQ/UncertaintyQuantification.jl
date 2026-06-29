@testset "MonteCarlo" begin
    mc = MonteCarlo(1000)

    @test isa(mc, MonteCarlo)
    @test mc.n == 1000
end

@testset "QuasiMonteCarlo" begin

    # Make sure that all QMC Types implement qmc_samples
    for qmc in subtypes(AbstractQuasiMonteCarlo)
        @test hasmethod(qmc_samples, (qmc, Integer))
    end

    @testset "SobolSampling" begin
        sobol = SobolSampling(1000, :none)

        @test isa(sobol, SobolSampling)
        @test sobol.n == 1024

        @testset "sample" begin
            inputs = [RandomVariable.(Uniform(), [:a, :b]); Parameter(1, :c)]

            samples = sample(inputs, SobolSampling(4, :none))

            @test isapprox(samples.a, [0.375, 0.875, 0.625, 0.125])
            @test isapprox(samples.b, [0.375, 0.875, 0.125, 0.625])
            @test samples.c == [1.0, 1.0, 1.0, 1.0]
        end
    end

    @testset "FaureSampling" begin
        faure = FaureSampling(1000, :none)

        @test isa(faure, FaureSampling)

        @testset "sample" begin
            inputs = [RandomVariable.(Uniform(), [:a, :b]); Parameter(1, :c)]

            samples = sample(inputs, FaureSampling(4, :none))

            @test isapprox(samples.a, [0.0625, 0.5625, 0.3125, 0.8125])
            @test isapprox(samples.b, [0.0625, 0.5625, 0.8125, 0.3125])
            @test samples.c == [1.0, 1.0, 1.0, 1.0]

            @test_logs (
                :warn,
                "n must be a power of the base (here 2), automatically increased to 8 for these samples.",
            ) sample(inputs, FaureSampling(7, :none))
        end
    end

    @testset "HaltonSampling" begin
        halton = HaltonSampling(1000, :none)

        @test isa(halton, HaltonSampling)

        @testset "sample" begin
            inputs = [
                RandomVariable.([Uniform(-1, 0), Uniform()], [:a, :b])
                Parameter(1, :c)
            ]

            samples = sample(inputs, halton)

            @test mean(samples.a) ≈ -0.5 rtol = 0.05
            @test mean(samples.b) ≈ 0.5 rtol = 0.05
            @test samples.c == ones(halton.n)
        end
    end

    @testset "LatticeRuleSampling" begin
        lattice = LatticeRuleSampling(1000, :none)

        @test isa(lattice, LatticeRuleSampling)

        @testset "sample" begin
            inputs = [RandomVariable.(Uniform(), [:a, :b]); Parameter(1, :c)]

            samples = sample(inputs, lattice)
            @test mean(samples.a) ≈ 0.5 rtol = 0.05
            @test mean(samples.b) ≈ 0.5 rtol = 0.05
            @test samples.c == ones(lattice.n)
        end
    end

    @testset "LatinHypercubeSampling" begin
        lhs = LatinHypercubeSampling(1000)

        inputs = RandomVariable.([Uniform(-1, 1), Uniform(0, 3)], [:a, :b])

        samples = sample(inputs, lhs)

        h = fit(Histogram, samples.a, range(-1, 1; length=1001))
        @test all(h.weights .== 1)

        h = fit(Histogram, samples.b, range(0, 3; length=1001))
        @test all(h.weights .== 1)
    end

    @testset "RQMC" begin
        inputs = RandomVariable.([Uniform(-1, 1), Uniform(0, 3)], [:a, :b])

        sobol = SobolSampling(4, :matousek)
        @test sample(inputs, sobol) != sample(inputs, sobol)

        faure = FaureSampling(4, :owen)
        @test sample(inputs, faure) != sample(inputs, faure)

        lattice = LatticeRuleSampling(4, :shift)
        @test sample(inputs, lattice) != sample(inputs, lattice)
    end
end

@testitem "JointInterval sampling" begin
    x = rand(3, 10)
    ji = JointInterval(x, [:x1, :x2, :x3])
    rv = RandomVariable(Normal(), :y)
    i = IntervalVariable(-1, 5, :z)

    inputs = [i, ji, rv]

    samples = sample(inputs, SobolSampling(256, :matousek); intervals=false)

    @test size(samples)[1] == 256

    @test all(insupport.(rv, samples.y))
    @test all(in.(samples.z, i))
    @test all(in.(samples.x1, ji.intervals[1]))
    @test all(in.(samples.x2, ji.intervals[2]))
    @test all(in.(samples.x3, ji.intervals[3]))
    @test all(in.(eachrow(Matrix(samples[:, [:x1, :x2, :x3]])), ji))

    @test_throws ErrorException(
        "QMC sampling must be randomized to be applied to joint intervals"
    ) sample(inputs, SobolSampling(256, :none); intervals=false)
end
