@testset "IntervalPredictorModel" begin
    N = 150

    data = sample(RandomVariable(Uniform(-5.5, 5.5), :x), HaltonSampling(N))

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

    @assert all(getproperty.(verify.y, :lb) .<= data.y .+ abs.(data.y) * 1e-8)

    @assert all(getproperty.(verify.y, :ub) .>= data.y .- abs.(data.y) * 1e-8)

    # lower bound only
    verify = copy(data)
    evaluate!(ipm, verify, :lb)
    @assert all(verify.y .<= data.y .+ abs.(data.y) * 1e-8)

    # upper bound only
    verify = copy(data)
    evaluate!(ipm, verify, :ub)
    @assert all(verify.y .>= data.y .- abs.(data.y) * 1e-8)

    @test ipm.N == 150
    @test reliability(ipm, 0.1548) ≈ 0.01 atol=0.001
end

@testset "Interval propagation IPM" begin
    X1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 2), :σ => 1)), :X1)
    X2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-2, 1), :σ => 2)), :X2)
    X3 = RandomVariable(Normal(0, 1), :X3)
    X4 = Parameter(5, :X4)

    inputs = [X1, X2, X3, X4]
    models = Model(df -> df.X1 .^ 2 .+ df.X2 .+ df.X3 .+ df.X4, :g)

    data_ipm = sample(
        [
            RandomVariable(Normal(1.5, 1), :X1),
            RandomVariable(Uniform(-1.5, 2), :X2),
            X3,
            X4,
        ],
        HaltonSampling(150),
    )

    evaluate!(models, data_ipm)

    b = MonomialBasis(2, 3)
    ipm = IntervalPredictorModel(data_ipm, :g, b, [:X1, :X2, :X3])

    df = sample(inputs, 500)

    propagate_intervals!(ipm, df)

    @test eltype(df.g) == Interval
end
