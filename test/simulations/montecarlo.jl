@testitem "MonteCarlo" begin
    mc = MonteCarlo(1000)

    @test isa(mc, MonteCarlo)
    @test mc.n == 1000
end

@testitem "QuasiMonteCarlo" setup = [TestSetup] begin

    # Make sure that all QMC Types implement qmc_samples
    for qmc in subtypes(AbstractQuasiMonteCarlo)
        @test hasmethod(qmc_samples, (qmc, Integer))
    end
end

@testitem "QuasiMonteCarlo: SobolSampling" begin
    sobol = SobolSampling(1000, :none)

    @test isa(sobol, SobolSampling)
    @test sobol.n == 1024

    inputs = [RandomVariable.(Uniform(), [:a, :b]); Parameter(1, :c)]

    samples = sample(inputs, SobolSampling(4, :none))

    @test isapprox(samples.a, [0.375, 0.875, 0.625, 0.125])
    @test isapprox(samples.b, [0.375, 0.875, 0.125, 0.625])
    @test samples.c == [1.0, 1.0, 1.0, 1.0]
end

@testitem "QuasiMonteCarlo: FaureSampling" begin
    faure = FaureSampling(1000, :none)

    @test isa(faure, FaureSampling)

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

@testitem "QuasiMonteCarlo: HaltonSampling" begin
    halton = HaltonSampling(1000, :none)

    @test isa(halton, HaltonSampling)

    inputs = [
        RandomVariable.([Uniform(-1, 0), Uniform()], [:a, :b])
        Parameter(1, :c)
    ]

    samples = sample(inputs, halton)

    @test mean(samples.a) ≈ -0.5 rtol = 0.05
    @test mean(samples.b) ≈ 0.5 rtol = 0.05
    @test samples.c == ones(halton.n)
end

@testitem "QuasiMonteCarlo: LatticeRuleSampling" begin
    lattice = LatticeRuleSampling(1000, :none)

    @test isa(lattice, LatticeRuleSampling)

    inputs = [RandomVariable.(Uniform(), [:a, :b]); Parameter(1, :c)]

    samples = sample(inputs, lattice)
    @test mean(samples.a) ≈ 0.5 rtol = 0.05
    @test mean(samples.b) ≈ 0.5 rtol = 0.05
    @test samples.c == ones(lattice.n)
end

@testitem "QuasiMonteCarlo: LatinHypercubeSampling" setup = [TestSetup] begin
    lhs = LatinHypercubeSampling(1000)

    inputs = RandomVariable.([Uniform(-1, 1), Uniform(0, 3)], [:a, :b])

    samples = sample(inputs, lhs)

    h = fit(Histogram, samples.a, range(-1, 1; length=1001))
    @test all(h.weights .== 1)

    h = fit(Histogram, samples.b, range(0, 3; length=1001))
    @test all(h.weights .== 1)
end

@testitem "QuasiMonteCarlo: RQMC" begin
    inputs = RandomVariable.([Uniform(-1, 1), Uniform(0, 3)], [:a, :b])

    sobol = SobolSampling(4, :matousek)
    @test sample(inputs, sobol) != sample(inputs, sobol)

    faure = FaureSampling(4, :owen)
    @test sample(inputs, faure) != sample(inputs, faure)

    lattice = LatticeRuleSampling(4, :shift)
    @test sample(inputs, lattice) != sample(inputs, lattice)
end
