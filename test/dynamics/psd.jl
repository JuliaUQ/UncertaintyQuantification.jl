@testitem "PSD generation" begin
    ω = collect(0:0.6:150)
    cp = CloughPenzien(ω, 0.1, 0.8π, 0.6, 8π, 0.6)
    kt = KanaiTajimi(ω, 0.25, 5, 0.75)
    sd = ShinozukaDeodatis(ω, 1, 1)

    cp_val = evaluate(cp)
    kt_val = evaluate(kt)
    sd_val = evaluate(sd)

    ep_cp = EmpiricalPSD(ω, cp_val)
    ep_kt = EmpiricalPSD(ω, kt_val)
    ep_sd = EmpiricalPSD(ω, sd_val)

    @test isa(cp, CloughPenzien)
    @test isa(kt, KanaiTajimi)
    @test isa(sd, ShinozukaDeodatis)
    @test cp_val ≈ evaluate(ep_cp)
    @test kt_val ≈ evaluate(ep_kt)
    @test sd_val ≈ evaluate(ep_sd)
end