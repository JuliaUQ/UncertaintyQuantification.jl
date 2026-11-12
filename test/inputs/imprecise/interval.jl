@testitem "Interval" begin
    using UncertaintyQuantification: lo, hi
    name = :l
    lb = 0.14
    ub = 0.16
    @test_throws ErrorException(
        "Lower bound parameter must be smaller than upper bound parameter for Intervals."
    ) Interval(ub, lb)
    interval = Interval(lb, ub)
    @test interval.lb == lb
    @test interval.ub == ub

    @test !(0.13 ∈ interval)
    @test 0.14 ∈ interval
    @test 0.15 ∈ interval
    @test 0.16 ∈ interval
    @test !(0.17 ∈ interval)

    @test sprint(show, interval) == "[0.14, 0.16]"

    @test hi(interval) == interval.ub
    @test lo(interval) == interval.lb

    @test hi(2.0) == 2.0
    @test lo(2.0) == 2.0
end

@testitem "Interval intersections" begin
    using UncertaintyQuantification: bounds

    a, b = Interval(0, 2), Interval(1, 3)
    @test bounds(a ∩ b) == (1, 2)
    @test bounds(b ∩ a) == (1, 2)
    @test bounds(a ∩ a) == (0, 2)
    @test bounds(a ∩ Interval(2, 4)) == (2, 2)
    @test bounds(a ∩ Interval(0.5, 1.5)) == (0.5, 1.5)
    @test !isempty(a)
    @test !isempty(a ∩ b)
    @test !isempty(Interval(2, 2))
    @test isempty(a ∩ Interval(3, 4))
    @test isempty(Interval(3, 4) ∩ a)
    @test isempty(Interval())
    @test isempty(Interval() ∩ a)
    @test isempty(a ∩ Interval())
    @test !(0 ∈ Interval())
    @test sprint(show, Interval()) == "∅"
    @test bounds(a ∩ 1) == (1, 1)
    @test bounds(1 ∩ a) == (1, 1)
    @test isempty(a ∩ 3)
    @test isempty(3 ∩ a)
    @test_throws ArgumentError a ∩ Interval(NaN, 1)
end

@testitem "IntervalVariable" setup = [TestSetup, QMC] begin
    name = :l
    lb = 0.14
    ub = 0.16
    @test_throws ErrorException(
        "Lower bound parameter must be smaller than upper bound parameter for Interval $name.",
    ) IntervalVariable(ub, lb, name)
    interval = IntervalVariable(lb, ub, name)
    @test interval.lb == lb
    @test interval.ub == ub
    @test interval.name == name
    @test !(0.13 ∈ interval)
    @test 0.14 ∈ interval
    @test 0.15 ∈ interval
    @test 0.16 ∈ interval
    @test !(0.17 ∈ interval)

    @test sprint(show, interval) == "l ∈ [0.14, 0.16]"

    par = 0.13
    @test_throws ErrorException("0.13 not in [0.14, 0.16] for Interval l.") UncertaintyQuantification.map_to_precise(
        par, interval
    )

    par = 0.17
    @test_throws ErrorException("0.17 not in [0.14, 0.16] for Interval l.") UncertaintyQuantification.map_to_precise(
        par, interval
    )

    @test UncertaintyQuantification.map_to_precise(0.15, interval) ==
        Parameter(0.15, interval.name)

    interval = IntervalVariable(0, 1, :x)

    @test mean(interval) == Interval(0, 1)
    @test var(interval) == Interval(0.0, 0.25)

    @test UncertaintyQuantification.sample(interval) == DataFrame(; x = Interval(interval))
end
