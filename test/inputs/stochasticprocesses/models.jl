@testitem "Stochastic Process Input Model" begin
    ω = collect(0:0.01:25)
    t = collect(0:0.02:10)
    cp = CloughPenzien(ω, 1, π, 0.2, 4π, 0.3)
    gm = SpectralRepresentation(cp, t, :gm)
    gm_model = StochasticProcessModel(gm)

    @test isa(gm_model, StochasticProcessModel)
    @test gm_model.name == :gm
    @test gm_model.proc == gm

    ϕ = sample(gm)

    evaluate!(gm_model, ϕ)
    x̃ = evaluate(gm, collect(ϕ[1, gm.ϕnames]))

    @test all(ϕ.gm[1] .== x̃)
end
