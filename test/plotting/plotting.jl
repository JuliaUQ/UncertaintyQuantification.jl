@testset "Plotting Recipes" begin

    @testset "RandomVariable PDF Plot" begin
        rv = RandomVariable(Normal(0, 1), :X)
        plt = plot(rv; cdf_on=false)
        @test typeof(plt) <: Plots.Plot
        @test plt.series_list[1][:seriestype] == :path
        @test plt.series_list[2][:seriestype] == :path
    end

    @testset "RandomVariable CDF Plot" begin
        rv = RandomVariable(Normal(0, 1), :X)
        plt = plot(rv; cdf_on=true)
        @test typeof(plt) <: Plots.Plot
    end

    @testset "IntervalVariable Plot" begin
        iv = IntervalVariable(1.0, 2.0, :Y)
        plt = plot(iv)
        @test typeof(plt) <: Plots.Plot
        @test length(plt.series_list) == 3
    end

    @testset "ProbabilityBox Plot" begin
        pb = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 2), :σ => 2)), :X2)
        plt = plot(pb)
        @test typeof(plt) <: Plots.Plot
        @test length(plt.series_list) == 3
    end

    @testset "Vector of UQInputs Plot" begin
        inputs = [RandomVariable(Normal(0,1), :A), IntervalVariable(1.0, 2.0, :B)]
        plt = plot(inputs)
        @test typeof(plt) <: Plots.Plot
        @test plt.layout isa Plots.GridLayout
    end

    @testset "2D IntervalBox Plot" begin
        x = Interval(1.0, 2.0)
        y = Interval(3.0, 4.0)
        plt = plot(x, y)
        @test typeof(plt) <: Plots.Plot
        @test plt.series_list[1][:seriestype] == :shape
    end

    @testset "Vector of IntervalBoxes Plot" begin
        xs = [Interval(1.0, 2.0), Interval(2.0, 3.0)]
        ys = [Interval(3.0, 4.0), Interval(4.0, 5.0)]
        plt = plot(xs, ys)
        @test typeof(plt) <: Plots.Plot
        @test plt.series_list[1][:seriestype] == :shape
    end

    @testset "Vector of Intervals Plot" begin
        intervals = [Interval(1.0, 2.0), Interval(1.5, 2.5), Interval(2.0, 3.0)]
        plt = plot(intervals)
        @test typeof(plt) <: Plots.Plot
    end
end
