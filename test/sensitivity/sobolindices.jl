@testsnippet sobolindices begin
    using QuasiMonteCarlo
    using DataFrames

    x = RandomVariable.(Uniform(-π, π), [:x1, :x2, :x3])

    ishigami_fun = (x1, x2, x3, a, b) -> sin(x1) + a * sin(x2)^2 + b * x3^4 * sin(x1)
    a1 = 7; b1 = 0.1
    ishigami1 = Model(
        df -> ishigami_fun.(df.x1, df.x2, df.x3, a1, b1), [:x1, :x2, :x3], :f1
    )
    a2 = 7; b2 = 0.05
    ishigami2 = Model(
        df -> ishigami_fun.(df.x1, df.x2, df.x3, a2, b2), [:x1, :x2, :x3], :f2
    )

    n_mc = 10000
    n_qmc = 10000

    # Analytical formula for the ishigami partial variances. The missing variances are all 0
    variance_analytical = (a, b) -> (0.5 + a^2 / 8 + b * pi^4 / 5 + b^2 * pi^8 / 18)
    V1_analytic = (a, b) -> (0.5 + b * pi^4 / 5 + b^2 * pi^8 / 50)
    V2_analytic = (a, b) -> (a^2 / 8)
    V13_analytic = (a, b) -> (0.5 + a^2 / 8 + b * pi^4 / 5 + b^2 * pi^8 / 18 - a^2 / 8 - 0.5 - b * pi^4 / 5 - b^2 * pi^8 / 50)

    variance_analytical1 = variance_analytical(a1, b1)
    firstorder_analytical1 = [
        V1_analytic(a1, b1) / variance_analytical1,
        V2_analytic(a1, b1) / variance_analytical1, 0.0,
    ]
    totaleffect_analytical1 = [
        (V1_analytic(a1, b1) + V13_analytic(a1, b1)) / variance_analytical1,
        V2_analytic(a1, b1) / variance_analytical1,
        V13_analytic(a1, b1) / variance_analytical1,
    ]

    variance_analytical2 = variance_analytical(a2, b2)
    firstorder_analytical2 = [
        V1_analytic(a2, b2) / variance_analytical2,
        V2_analytic(a2, b2) / variance_analytical2,
        0.0,
    ]
    totaleffect_analytical2 = [
        (V1_analytic(a2, b2) + V13_analytic(a2, b2)) / variance_analytical2,
        V2_analytic(a2, b2) / variance_analytical2,
        V13_analytic(a2, b2) / variance_analytical2,
    ]
end

@testitem "Sobol Indices: Monte Carlo" setup = [sobolindices] begin
    si = sobolindices(ishigami1, x, :f1, MonteCarlo(n_mc))

    @test first(si.OutputVariance) ≈ variance_analytical1 rtol = 0.05
    @test si.FirstOrder[1:2] ≈ firstorder_analytical1[1:2] rtol = 0.1
    @test si.FirstOrder[3] ≈ firstorder_analytical1[3] atol = 0.1
    @test si.TotalEffect ≈ totaleffect_analytical1 rtol = 0.1
end

@testitem "Sobol Indices: Evaluations" setup = [sobolindices] begin
    si, evals = sobolindices(ishigami1, x, :f1, MonteCarlo(n_mc), return_samples = true)

    @test first(si.OutputVariance) ≈ variance_analytical1 rtol = 0.05
    @test si.FirstOrder[1:2] ≈ firstorder_analytical1[1:2] rtol = 0.1
    @test si.FirstOrder[3] ≈ firstorder_analytical1[3] atol = 0.1
    @test si.TotalEffect ≈ totaleffect_analytical1 rtol = 0.1

    inputs = [:x1, :x2, :x3]

    @test nrow(evals.A) == n_mc
    @test nrow(evals.B) == n_mc
    @test sort(collect(keys(evals.AB))) == inputs

    # Compared as vectors so near-zero function values are not an issue
    @test evals.A.f1 ≈ ishigami_fun.(evals.A.x1, evals.A.x2, evals.A.x3, a1, b1)
    @test evals.B.f1 ≈ ishigami_fun.(evals.B.x1, evals.B.x2, evals.B.x3, a1, b1)

    for input in inputs
        ab = evals.AB[input]
        @test nrow(ab) == n_mc
        # Column `input` comes from B, all other inputs from A
        @test ab[:, input] == evals.B[:, input]
        for other in setdiff(inputs, [input])
            @test ab[:, other] == evals.A[:, other]
        end
        @test ab.f1 ≈ ishigami_fun.(ab.x1, ab.x2, ab.x3, a1, b1)
    end
end

@testitem "Sobol Indices: Evaluations multiple outputs" setup = [sobolindices] begin
    si, evals = sobolindices(
        [ishigami1, ishigami2], x, [:f1, :f2], MonteCarlo(n_mc), return_samples = true
    )

    @test first(si[:f1].OutputVariance) ≈ variance_analytical1 rtol = 0.05
    @test first(si[:f2].OutputVariance) ≈ variance_analytical2 rtol = 0.05

    for df in (evals.A, evals.B, values(evals.AB)...)
        @test df.f1 ≈ ishigami_fun.(df.x1, df.x2, df.x3, a1, b1)
        @test df.f2 ≈ ishigami_fun.(df.x1, df.x2, df.x3, a2, b2)
    end
end

@testitem "Sobol Indices: Sobol" setup = [sobolindices] begin
    si = sobolindices(ishigami1, x, :f1, QuasiMonteCarloSampling(n_qmc, QuasiMonteCarlo.SobolSample()))

    @test first(si.OutputVariance) ≈ variance_analytical1 rtol = 0.05
    @test si.FirstOrder[1:2] ≈ firstorder_analytical1[1:2] rtol = 0.1
    @test si.FirstOrder[3] ≈ firstorder_analytical1[3] atol = 0.1
    @test si.TotalEffect ≈ totaleffect_analytical1 rtol = 0.1
end

@testitem "Sobol Indices: Halton" setup = [sobolindices] begin
    si = sobolindices(ishigami1, x, :f1, QuasiMonteCarloSampling(n_qmc, QuasiMonteCarlo.HaltonSample()))

    @test first(si.OutputVariance) ≈ variance_analytical1 rtol = 0.05
    @test si.FirstOrder[1:2] ≈ firstorder_analytical1[1:2] rtol = 0.1
    @test si.FirstOrder[3] ≈ firstorder_analytical1[3] atol = 0.1
    @test si.TotalEffect ≈ totaleffect_analytical1 rtol = 0.1
end

@testitem "Latin Hypercube" setup = [sobolindices] begin
    si = sobolindices(ishigami1, x, :f1, QuasiMonteCarloSampling(n_qmc, LatinHypercubeSample()))

    @test first(si.OutputVariance) ≈ variance_analytical1 rtol = 0.05
    @test si.FirstOrder[1:2] ≈ firstorder_analytical1[1:2] rtol = 0.1
    @test si.FirstOrder[3] ≈ firstorder_analytical1[3] atol = 0.1
    @test si.TotalEffect ≈ totaleffect_analytical1 rtol = 0.1
end

@testitem "Sobol Indices: Multiple Outputs" setup = [sobolindices] begin
    si = sobolindices(
        [ishigami1, ishigami2], x, [:f1, :f2], QuasiMonteCarloSampling(n_qmc, QuasiMonteCarlo.SobolSample())
    )

    @test first(si[:f1].OutputVariance) ≈ variance_analytical1 rtol = 0.05
    @test si[:f1].FirstOrder[1:2] ≈ firstorder_analytical1[1:2] rtol = 0.1
    @test si[:f1].FirstOrder[3] ≈ firstorder_analytical1[3] atol = 0.1
    @test si[:f1].TotalEffect ≈ totaleffect_analytical1 rtol = 0.1

    @test first(si[:f2].OutputVariance) ≈ variance_analytical2 rtol = 0.05
    @test si[:f2].FirstOrder[1:2] ≈ firstorder_analytical2[1:2] rtol = 0.1
    @test si[:f2].FirstOrder[3] ≈ firstorder_analytical2[3] atol = 0.1
    @test si[:f2].TotalEffect ≈ totaleffect_analytical2 rtol = 0.1
end

@testitem "Sobol Indices: Convenience Functions" setup = [sobolindices] begin
    x1 = RandomVariable(Uniform(-π, +π), :x1)
    x2 = RandomVariable(Uniform(0, +π), :x2)
    t1 = Model(df -> sin.(df.x1), [:x1], :t1)
    t2 = Model(df -> cos.(df.x1), [:x1], :t2)

    @test isa(
        sobolindices([t1; t2], x1, :t1, QuasiMonteCarloSampling(2, QuasiMonteCarlo.SobolSample())), DataFrame
    )
    @test isa(
        sobolindices(t1, [x1; x2], :t1, QuasiMonteCarloSampling(2, QuasiMonteCarlo.SobolSample())), DataFrame
    )
    @test isa(
        sobolindices([t1; t2], [x1; x2], :t1, QuasiMonteCarloSampling(2, QuasiMonteCarlo.SobolSample())),
        DataFrame,
    )
    @test isa(
        sobolindices([t1; t2], x1, [:t1, :t2], QuasiMonteCarloSampling(2, QuasiMonteCarlo.SobolSample())),
        Dict{Symbol, DataFrame},
    )
    @test isa(sobolindices(t1, x1, :t1, QuasiMonteCarloSampling(2, QuasiMonteCarlo.SobolSample())), DataFrame)
end

@testitem "Sobol Indices: Polynomial Chaos Expansion" setup = [sobolindices] begin
    p = 6
    basis = LegendreBasis()
    Ψ = PolynomialChaosBasis([basis, basis, basis], p)

    gq = GaussQuadrature()
    pce, _ = polynomialchaos(x, ishigami1, Ψ, :f1, gq)

    si = sobolindices(pce)

    @test first(si.OutputVariance) ≈ variance_analytical1 rtol = 0.1
    @test si.FirstOrder[1:2] ≈ firstorder_analytical1[1:2] rtol = 0.1
    @test si.FirstOrder[3] ≈ firstorder_analytical1[3] atol = 0.1
    @test si.TotalEffect ≈ totaleffect_analytical1 rtol = 0.1
end
