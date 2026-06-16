@testitem "FORM" setup = [TestSetup] begin
    form = FORM()

    x1 = RandomVariable(Normal(200, 20), :x1)
    x2 = RandomVariable(Normal(150, 10), :x2)

    inputs = [x1, x2]

    model = Model(df -> df.x1 .- df.x2, :y)

    pf, β, dp, α = probability_of_failure(df -> df.x1 .- df.x2, inputs, form)

    @test round(pf; digits=4) ≈ 0.0127
    @test round(β; digits=4) ≈ 2.2361

    dp = DataFrame([dp])
    to_standard_normal_space!(inputs, dp)
    u = -β * [values(α)...]

    @test all(u .≈ [dp[1, :]...])

    pf, β, dp, α = probability_of_failure(model, df -> df.y, inputs, form)

    @test round(pf; digits=4) ≈ 0.0127
    @test round(β; digits=4) ≈ 2.2361

    dp = DataFrame([dp])
    to_standard_normal_space!(inputs, dp)
    u = -β * [values(α)...]

    @test all(u .≈ [dp[1, :]...])
end
