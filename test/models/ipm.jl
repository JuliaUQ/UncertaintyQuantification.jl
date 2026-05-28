@testset "IntervalPredictorModel" begin
    N = 150

    data = DataFrame(:x => rand(Uniform(-5.5, 5.5), N))

    verify = copy(data)

    m = Model(
        df ->
            df.x .^ 2 .* cos.(df.x) .- sin.(3 * df.x) .* exp.(-df.x .^ 2) .- df.x .-
            cos.(df.x .^ 2) .+ df.x .* randn(size(df, 1)),
        :y,
    )

    evaluate!(m, data)

    ipm = IntervalPredictorModel(data, :y, MonomialBasis(1, 6))

    evaluate!(ipm, verify)

    @test all(in.(data.y, verify.y))
end