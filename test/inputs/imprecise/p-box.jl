@testset "P-box" begin
    params = Dict(
        :a => Interval(0.14, 0.16), :b => Interval(0.21, 0.23), :c => Interval(0, 1)
    )
    @test_throws ErrorException(
        "Parameter mismatch for ProbabilityBox [:a, :b, :c] != [:a, :b]."
    ) ProbabilityBox{Uniform}(params)

    params = Dict(:a => Interval(0.14, 0.16), :b => Interval(0.21, 0.23))

    p_box = ProbabilityBox{Uniform}(params)
    @test p_box.parameters == params
    @test p_box.lb == 0.14
    @test p_box.ub == 0.23
    @test mean(p_box) == Interval(0.175, 0.195)

    @test UncertaintyQuantification.bounds(p_box) == ([0.14, 0.21], [0.16, 0.23])

    par = [0.13, 0.20]
    @test_throws ErrorException("Values outside of parameter intervals for ProbabilityBox") UncertaintyQuantification.map_to_precise(
        par, p_box
    )

    par = [0.17, 0.25]
    @test_throws ErrorException("Values outside of parameter intervals for ProbabilityBox") UncertaintyQuantification.map_to_precise(
        par, p_box
    )

    par = [0.15, 0.22]
    @test UncertaintyQuantification.map_to_precise(par, p_box) == Uniform(par...)

    p_box = ProbabilityBox{Exponential}(Interval(0.5, 0.75))
    @test p_box.parameters == Dict{Symbol,Interval}(:θ => Interval(0.5, 0.75))
    @test p_box.lb == 0.0
    @test p_box.ub == Inf
    @test mean(p_box) == Interval(0.5, 0.75)
    @test var(p_box) == Interval(0.25, 0.5625)

    pbox = ProbabilityBox{Normal}([:μ => Interval(-1, 1), :σ => 1])
    @test pbox.parameters == Dict(:μ => Interval(-1, 1), :σ => 1)
    @test pbox.lb == -Inf
    @test pbox.ub == Inf

    pbox = ProbabilityBox{Normal}([:μ => Interval(-1, 1), :σ => 1], 0, Inf)
    @test pbox.parameters == Dict(:μ => Interval(-1, 1), :σ => 1)
    @test pbox.lb == 0
    @test pbox.ub == Inf

    @test_logs (
        :warn,
        "ProbabilityBox() returns a UnivariateDistribution if no intervals are passed",
    ) ProbabilityBox{Normal}(Dict(:μ => 0.0, :σ => 1.0))

    @test ProbabilityBox{Normal}(Dict(:μ => 0.0, :σ => 1.0)) == Normal()

    @testset "Invalid distributions" begin
        @test_throws ArgumentError(
            "Invalid Normal distribution for parameter combination (0, -1)"
        ) ProbabilityBox{Normal}(Dict(:μ => 0, :σ => Interval(-1, 1)))

        @test_throws ArgumentError(
            "Invalid Uniform distribution for parameter combination (0.16, 0.16)"
        ) ProbabilityBox{Uniform}(Dict(:a => Interval(0.14, 0.16), :b => 0.16))
    end

    @testset "Quantile and sampling" begin
        p_box = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => Interval(0.1, 1)))
        a = UncertaintyQuantification.quantile(p_box, 0.25)
        @test a.lb == quantile(Normal(0, 1), 0.25)
        @test a.ub == quantile(Normal(1, 0.1), 0.25)

        a = UncertaintyQuantification.quantile(p_box, 0.5)
        @test a.lb == quantile(Normal(0, 0.1), 0.5)
        @test a.lb == quantile(Normal(0, 1), 0.5)
        @test a.ub == quantile(Normal(1, 0.1), 0.5)
        @test a.ub == quantile(Normal(1, 1), 0.5)

        a = UncertaintyQuantification.quantile(p_box, 0.75)
        @test a.lb == quantile(Normal(0, 0.1), 0.75)
        @test a.ub == quantile(Normal(1, 1), 0.75)

        p_box = ProbabilityBox{Normal}(Dict(:μ => 0, :σ => Interval(0.1, 1)))
        @test UncertaintyQuantification.bounds(p_box) == ([0.1], [1])

        a = UncertaintyQuantification.quantile(p_box, 0.25)
        @test a.lb == quantile(Normal(0, 1), 0.25)
        @test a.ub == quantile(Normal(0, 0.1), 0.25)

        a = UncertaintyQuantification.quantile(p_box, 0.5)
        @test a == Interval(0.0, 0.0)

        a = UncertaintyQuantification.quantile(p_box, 0.75)
        @test a.lb == quantile(Normal(0, 0.1), 0.75)
        @test a.ub == quantile(Normal(0, 1), 0.75)
    end

    @testset "Inverse quantile" begin
        p_box = ProbabilityBox{Uniform}(
            Dict(:a => Interval(0, 0.2), :b => Interval(0.5, 1))
        )

        Nsamples = 1000

        u = rand(Nsamples)
        x = quantile.(Ref(p_box), u)

        u_back = UncertaintyQuantification.reverse_quantile.(Ref(p_box), x)

        @test all(abs.(u_back .- u) .<= 10^-10)
    end
end
