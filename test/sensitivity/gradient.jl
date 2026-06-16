@testitem "Gradient in physical space" setup = [TestSetup] begin
    p = Parameter(2.0, :p)
    x = RandomVariable(Uniform(-2.0, 2.0), :x)
    y = RandomVariable(Uniform(-2.0, 2.0), :y)

    model = Model(df -> df.p .* df.x .^ 2 - df.y .^ 2, :f)

    g = gradient([model], [p, x, y], DataFrame(; p=2.0, x=2.0, y=1.0), :f)

    @test isapprox(g.x, 8.0; rtol=0.001)
    @test isapprox(g.y, -2.0; rtol=0.001)

    g = gradient(
        [model],
        [p, x, y],
        DataFrame(; p=2.0, x=2.0, y=1.0),
        :f;
        fdm=ForwardFiniteDifferences(3, 1),
    )

    @test isapprox(g.x, 8.0; rtol=0.001)
    @test isapprox(g.y, -2.0; rtol=0.001)

    g = gradient(
        [model],
        [p, x, y],
        DataFrame(; p=2.0, x=2.0, y=1.0),
        :f;
        fdm=BackwardFiniteDifferences(3, 1),
    )

    @test isapprox(g.x, 8.0; rtol=0.001)
    @test isapprox(g.y, -2.0; rtol=0.001)
end
