@testsnippet sobolindices begin
    x = RandomVariable.(Uniform(-π, π), [:x1, :x2, :x3])

    ishigami1 = Model(
        df -> sin.(df.x1) + 7 .* sin.(df.x2) .^ 2 + 0.1 .* df.x3 .^ 4 .* sin.(df.x1), :f1
    )
    ishigami2 = Model(
        df -> sin.(df.x1) + 7 .* sin.(df.x2) .^ 2 + 0.05 .* df.x3 .^ 4 .* sin.(df.x1), :f2
    )

    n_mc = 10000
    n_qmc = 10000

    firstorder_analytical1 = [0.3138, 0.4424, 0.00]
    totaleffect_analytical1 = [0.5574, 0.4424, 0.2436]

    firstorder_analytical2 = [0.219, 0.687, 0.00]
    totaleffect_analytical2 = [0.3136, 0.687, 0.0946]
end

@testitem "Sobol Indices: Monte Carlo" setup = [sobolindices] begin
    si = sobolindices(ishigami1, x, :f1, MonteCarlo(n_mc))

    @test si.FirstOrder[1:2] ≈ firstorder_analytical1[1:2] rtol = 0.1
    @test si.FirstOrder[3] ≈ firstorder_analytical1[3] atol = 0.1
    @test si.TotalEffect ≈ totaleffect_analytical1 rtol = 0.1
end

@testitem "Sobol Indices: Sobol" setup = [sobolindices] begin
    si = sobolindices(ishigami1, x, :f1, SobolSampling(n_qmc))

    @test si.FirstOrder[1:2] ≈ firstorder_analytical1[1:2] rtol = 0.1
    @test si.FirstOrder[3] ≈ firstorder_analytical1[3] atol = 0.1
    @test si.TotalEffect ≈ totaleffect_analytical1 rtol = 0.1
end

@testitem "Sobol Indices: Halton" setup = [sobolindices] begin
    si = sobolindices(ishigami1, x, :f1, HaltonSampling(n_qmc))

    @test si.FirstOrder[1:2] ≈ firstorder_analytical1[1:2] rtol = 0.1
    @test si.FirstOrder[3] ≈ firstorder_analytical1[3] atol = 0.1
    @test si.TotalEffect ≈ totaleffect_analytical1 rtol = 0.1
end

# @testitem "Latin Hypercube" begin
#     si = sobolindices(ishigami1, x, :f1, LatinHypercubeSampling(n_qmc))

#     @test si.FirstOrder[1:2] ≈ firstorder_analytical1[1:2] rtol = 0.1
#     @test si.FirstOrder[3] ≈ firstorder_analytical1[3] atol = 0.1
#     @test si.TotalEffect ≈ totaleffect_analytical1 rtol = 0.1
# end

@testitem "Sobol Indices: Multiple Outputs" setup = [sobolindices] begin
    si = sobolindices([ishigami1, ishigami2], x, [:f1, :f2], SobolSampling(n_qmc))

    @test si[:f1].FirstOrder[1:2] ≈ firstorder_analytical1[1:2] rtol = 0.1
    @test si[:f1].FirstOrder[3] ≈ firstorder_analytical1[3] atol = 0.1
    @test si[:f1].TotalEffect ≈ totaleffect_analytical1 rtol = 0.1

    @test si[:f2].FirstOrder[1:2] ≈ firstorder_analytical2[1:2] rtol = 0.1
    @test si[:f2].FirstOrder[3] ≈ firstorder_analytical2[3] atol = 0.1
    @test si[:f2].TotalEffect ≈ totaleffect_analytical2 rtol = 0.1
end

@testitem "Sobol Indices: Convenience Functions" setup = [sobolindices, TestSetup] begin
    x1 = RandomVariable(Uniform(-π, +π), :x1)
    x2 = RandomVariable(Uniform(0, +π), :x2)
    t1 = Model(df -> sin.(df.x1), :t1)
    t2 = Model(df -> cos.(df.x1), :t2)

    @test isa(sobolindices([t1; t2], x1, :t1, SobolSampling(2)), DataFrame)
    @test isa(sobolindices(t1, [x1; x2], :t1, SobolSampling(2)), DataFrame)
    @test isa(sobolindices([t1; t2], [x1; x2], :t1, SobolSampling(2)), DataFrame)
    @test isa(
        sobolindices([t1; t2], x1, [:t1, :t2], SobolSampling(2)), Dict{Symbol,DataFrame}
    )
    @test isa(sobolindices(t1, x1, :t1, SobolSampling(2)), DataFrame)
end

@testitem "Sobol Indices: Polynomial Chaos Expansion" setup = [sobolindices] begin
    p = 6
    basis = LegendreBasis()
    Ψ = PolynomialChaosBasis([basis, basis, basis], p)

    gq = GaussQuadrature()
    pce, _ = polynomialchaos(x, ishigami1, Ψ, :f1, gq)

    si = sobolindices(pce)

    @test si.FirstOrder[1:2] ≈ firstorder_analytical1[1:2] rtol = 0.1
    @test si.FirstOrder[3] ≈ firstorder_analytical1[3] atol = 0.1
    @test si.TotalEffect ≈ totaleffect_analytical1 rtol = 0.1
end
