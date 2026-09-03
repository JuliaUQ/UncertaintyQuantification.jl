@testitem "MonteCarlo" begin
    mc = MonteCarlo(1000)

    @test isa(mc, MonteCarlo)
    @test mc.n == 1000
    @test double_samples(mc).n == 2000
end

@testitem "QuasiMonteCarlo" begin
    using QuasiMonteCarlo

    sobol = QuasiMonteCarloSampling(4, SobolSample())
    @test isa(sobol, QuasiMonteCarloSampling)

    @test double_samples(sobol).n == 8

    inputs = [RandomVariable.(Uniform(), [:a, :b]); Parameter(1, :c)]
    samples = UncertaintyQuantification.sample(inputs, sobol)

    input = RandomVariable(Uniform(), :x)
    sample = UncertaintyQuantification.sample(input, sobol)

    @test isapprox(samples.a, [0.375, 0.875, 0.625, 0.125])
    @test isapprox(samples.b, [0.375, 0.875, 0.125, 0.625])
    @test samples.c == [1.0, 1.0, 1.0, 1.0]
    @test isapprox(sample.x, [0.375, 0.875, 0.625, 0.125])


    sobol = QuasiMonteCarloSampling(
        64, SobolSample(R = OwenScramble(base = 2, pad = 32))
    )
    @test UncertaintyQuantification.sample(inputs, sobol) !=
        UncertaintyQuantification.sample(inputs, sobol)

end
