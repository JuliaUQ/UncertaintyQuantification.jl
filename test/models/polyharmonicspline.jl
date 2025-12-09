@testset "PolyharmonicSpline" begin
    x = RandomVariable.(Uniform(-π, π), [:x1, :x2, :x3])
    a = Parameter(7, :a)
    b = Parameter(0.1, :b)

    inputs = [x; a; b]

    ishigami = Model(
        df ->
            sin.(df.x1) .+ df.a .* sin.(df.x2) .^ 2 .+ df.b .* (df.x3 .^ 4) .* sin.(df.x1),
        :y,
    )

    Random.seed!(8128)

    data = sample(inputs, 100)
    evaluate!(ishigami, data)

    Random.seed!()
    spline = PolyharmonicSpline(data, 2, :y)

    splinedata = select(data, Not(:y))
    evaluate!(spline, splinedata)

    mse = mean((data.y .- splinedata.y) .^ 2)
    @test isapprox(mse, 0; atol=eps(Float64))
end
