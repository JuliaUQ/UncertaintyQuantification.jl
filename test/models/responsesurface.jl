@testset "ResponseSurface" begin
    x = RandomVariable.(Uniform(-5, 5), [:x1, :x2])

    himmelblau = Model(
        df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2, :y
    )

    data = sample(x, FullFactorial([5, 5]))
    evaluate!(himmelblau, data)

    rs = ResponseSurface(data, :y, 4)

    test_data = sample(x, SobolSampling(1024))
    validate_data = copy(test_data)

    evaluate!(himmelblau, test_data)
    evaluate!(rs, validate_data)

    mse = mean((test_data.y .- validate_data.y) .^ 2)

    @test mse < 1e-25

    @test_throws ErrorException("Degree(p) of ResponseSurface must be non-negative.") ResponseSurface(
        data, :y, -3
    )
end
