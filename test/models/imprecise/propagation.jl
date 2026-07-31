@testset "Interval propagation" begin
    X1 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 2), :σ => 1)), :X1)
    X2 = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-2, 1), :σ => 2)), :X2)
    X3 = RandomVariable(Normal(0, 1), :X3)
    X4 = Parameter(5, :X4)

    inputs = [X1, X2, X3, X4]
    models = Model(df -> df.X1 .^ 2 .+ df.X2 .+ df.X3 .+ df.X4, :g)

    df = sample(inputs, 500)

    @test_throws "Invalid bound: hi" propagate_intervals!(models, df, :hi)
    propagate_intervals!(models, df)

    @test eltype(df.g) == Interval

    inputs = [RandomVariable.(Normal(), [:X1, :X2, :X3])..., X4]
    df = sample(inputs, 500)

    @test_throws "No intervals to propagate." propagate_intervals!(models, df)
end
