@testset "Plotting Recipes" begin

    @testset "RandomVariable PDF Plot" begin
        rv = RandomVariable(Normal(0, 1), :X)
        plt = plot(rv; cdf_on = false)
        @test typeof(plt) <: Plots.Plot
        @test plt.series_list[1][:seriestype] == :path
        @test plt.series_list[2][:seriestype] == :path
    end

    @testset "RandomVariable CDF Plot" begin
        rv = RandomVariable(Normal(0, 1), :X)
        plt = plot(rv; cdf_on = true)
        @test typeof(plt) <: Plots.Plot
    end

    @testset "IntervalVariable Plot" begin
        iv = IntervalVariable(1.0, 2.0, :Y)
        plt = plot(iv)
        @test typeof(plt) <: Plots.Plot
    end

    @testset "ProbabilityBox Plot" begin
        pb = RandomVariable(ProbabilityBox{Normal}(Dict(:μ => Interval(-1, 2), :σ => 2)), :X2)
        plt = plot(pb)
        @test typeof(plt) <: Plots.Plot
        @test plt.series_list[1][:label] == "X2"
    end

    @testset "Direct ProbabilityBox Plot" begin
        pboxes = [
            ProbabilityBox(; mean = 0, var = 1),
            ProbabilityBox(; mean = 2, lb = 0),
            ProbabilityBox(; mean = Interval(1, 2), var = Interval(0.2, 1), range = Interval(0, 5)),
            ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1)),
        ]
        for pbox in pboxes
            plt = plot(pbox)
            @test plt isa Plots.Plot
            @test length(plt.series_list) == 3
            xs = plt.series_list[1][:x]
            @test all(isfinite, xs)
            cdfs = cdf.(Ref(pbox), xs)
            @test plt.series_list[1][:y] == getproperty.(cdfs, :ub)
            @test plt.series_list[2][:y] == getproperty.(cdfs, :lb)
            @test plt.series_list[3][:fillrange] == plt.series_list[1][:y]
        end
        pbox = first(pboxes)
        plt = plot(pbox; shade = false, label = "a")
        @test length(plt.series_list) == 2
        @test plt.series_list[1][:label] == "a"
        named = plot(RandomVariable(pbox, :x); shade = false)
        @test length(named.series_list) == 2
        @test named.series_list[1][:label] == "x"
        @test named.series_list[1][:x] == plt.series_list[1][:x]
        @test named.series_list[1][:y] == plt.series_list[1][:y]
        plot!(plt, pbox; shade = false, label = "b")
        @test length(plt.series_list) == 4
    end

    @testset "Vector of UQInputs Plot" begin
        inputs = [RandomVariable(Normal(0, 1), :A), IntervalVariable(1.0, 2.0, :B)]
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
        plt = plot(xs, ys; label = "boxes")
        @test typeof(plt) <: Plots.Plot
        @test plt.series_list[1][:seriestype] == :shape
        @test plt.series_list[1][:label] == "boxes"
    end

    @testset "Vector of Intervals Plot" begin
        intervals = [Interval(1.0, 2.0), Interval(1.5, 2.5), Interval(2.0, 3.0)]
        plt = plot(intervals)
        @test length(plt.series_list) == 2
        @test plt.series_list[1][:linewidth] == 2
        @test plt.series_list[2][:linewidth] == 2

        shaded = plot(intervals; shade = true)
        @test length(shaded.series_list) == 3
        @test shaded.series_list[3][:fillrange] !== nothing
    end
end
