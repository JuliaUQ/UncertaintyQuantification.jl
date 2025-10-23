@testset "Spectral Representation" begin
    N = 128
    ω_u = 4π
    Δω = ω_u / N
    T0 = 2π / Δω
    Δt = 2π / (2 * ω_u)

    t = collect(Δt:Δt:(T0))
    ω = collect(0:Δω:(ω_u - Δω))

    sd = ShinozukaDeodatis(ω, 1, 1)
    Sval = evaluate(sd)

    srm_obj = SpectralRepresentation(sd, t, :ShnzkSR)
    ϕ = sample(srm_obj)
    ϕ_temp = copy(ϕ)
    x = evaluate(srm_obj, collect(ϕ[1, srm_obj.ϕnames]))

    Ŝ = Δt^2 / T0 * abs.(UncertaintyQuantification.dft(x)) .^ 2 / (2π)
    S̃ = periodogram(x, t)
    S̄ = periodogram(x, t, false)

    x_2 = srm_obj(collect(ϕ[1, srm_obj.ϕnames]))
    x_3 = srm_obj(collect(ϕ[1, srm_obj.ϕnames]))

    to_standard_normal_space!(srm_obj, ϕ)
    to_physical_space!(srm_obj, ϕ)

    @test Sval ≈ Ŝ[1:N]
    @test Sval ≈ S̃[1:N]
    @test Ŝ ≈ S̃
    @test S̄ ≈ 2 .* Ŝ[1:N]
    @test S̄ ≈ 2 .* S̃[1:N]

    @test x ≈ x_2
    @test x ≈ x_3
    @test x_2 ≈ x_3

    @test ϕ ≈ ϕ_temp

    t = collect(0:1:10)
    @test_logs (:warn, r"The frequency of the signal") SpectralRepresentation(sd, t, :ShnzkNySR)

    @testset "Reliability" begin
        
        ω = collect(range(0, 50, 100))

        cp = CloughPenzien(ω, 0.1, 0.8π, 0.6, 8π, 0.6)

        gm = SpectralRepresentation(cp, collect(range(0, 10, 200)), :gm)
        gm_model = StochasticProcessModel(gm)

        capacity = Parameter(25, :cap)

        function limitstate(df)
            return df.cap - map(sum, df.gm)
        end

        models = [gm_model]
        inputs = [gm, capacity]

        mc = MonteCarlo(10^4)

        pf_mc, _, _ = probability_of_failure(models, limitstate, inputs, mc)

        # Reference solution obtained with 10^6 samples: 0.003529
        @test 0.001 < pf_mc < 0.0064 # 99 percentiles obtained from 5000 independent runs with 10^4 samples

        # We use subset to confirm that the mappings to sns are done correctly

        ss = SubSetInfinityAdaptive(2000, 0.1, 20, 10, 0.6, 1)

        pf_ss, _, _ = probability_of_failure(models, limitstate, inputs, ss)

        @test 0.001 < pf_ss < 0.0064
    end
end
