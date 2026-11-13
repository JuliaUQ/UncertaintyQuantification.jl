@testitem "CABO: generic input mapping" begin
    using DataFrames
    using Random
    UQ = UncertaintyQuantification
    inputs = [
        Parameter(3, :offset),
        RandomVariable(ProbabilityBox{Normal}(Dict(:σ => Interval(1, 2), :μ => Interval(-2, 1))), :load),
        IntervalVariable(-1, 2, :shift), RandomVariable(Uniform(-2, 2), :x),
        IntervalVariable(4, 4, :constant),
    ]
    layout = UQ._cabo_input_map(inputs)
    @test layout.coordinates == [(:load, :μ), (:load, :σ), (:shift, :value), (:constant, :value)]
    augmented = DataFrame(
        _cabo_a1 = [0.5, cdf(Normal(), 1.0)], _cabo_a2 = [1.0, -1.0],
        _cabo_e1 = [-2.0, 1.0], _cabo_e2 = [2.0, 1.0], _cabo_e3 = [-1.0, 2.0], _cabo_e4 = [4.0, 4.0]
    )
    saved = copy(augmented)
    physical = UQ._cabo_to_physical(layout, augmented)
    @test physical.load ≈ [-2.0, 2.0]
    @test physical.shift == [-1.0, 2.0]
    @test physical.x == [1.0, -1.0]
    @test physical.offset == [3.0, 3.0]
    @test physical.constant == [4.0, 4.0]
    @test augmented == saved
    candidates = [-2.0 1.0 -1.0 4.0; 1.0 2.0 2.0 4.0; -2.0 1.0 -1.0 4.0]
    @test size(UQ._cabo_candidates(layout, CABO(epistemic_candidates = candidates))) == (2, 4)

    # Automatic grids include both endpoints and retain constant coordinates.
    grid = UQ._cabo_candidates(layout, CABO(grid_points = 3))
    @test size(grid) == (27, 4)
    @test grid[1, :] == [-2.0, 1.0, -1.0, 4.0]
    @test grid[end, :] == [1.0, 2.0, 2.0, 4.0]
    @test all(==(4.0), grid[:, 4])

    # A second family verifies that the mapping does not hard-code Normal.
    other = RandomVariable(ProbabilityBox{Exponential}(Dict(:θ => Interval(1, 3))), :life)
    other_layout = UQ._cabo_input_map(other)
    other_points = DataFrame(_cabo_a1 = [0.5], _cabo_e1 = [2.0])
    @test UQ._cabo_to_physical(other_layout, other_points).life ≈ [2log(2)]
end

@testitem "CABO: budgets, model chain, reuse and reproducibility" begin
    using DataFrames
    using Random
    using Statistics
    calls = Ref(0)
    inputs = [
        RandomVariable(Normal(), :x), IntervalVariable(-1, 1, :shift),
        Parameter(2, :scale), IntervalVariable(0, 0, :offset),
    ]
    models = [
        Model(
            df -> begin
                calls[] += nrow(df)
                df.x .+ df.shift
            end, :y
        ), Model(df -> df.scale .* df.y .+ df.offset, :g),
    ]
    original_models = copy(models)
    method = CABO(
        initial_points = 8, lower_bound_iterations = 3, upper_bound_iterations = 2,
        aleatory_samples = 128, posterior_draws = 32, grid_points = 3
    )
    global_rng = copy(Random.default_rng())
    pf, xmin, xmax, info = probability_of_failure(
        models, df -> df.g, inputs, method;
        rng = MersenneTwister(19), return_diagnostics = true
    )
    @test rand() == rand(global_rng) # explicit RNG must not consume the global stream
    @test calls[] == info.calls == nrow(info.samples) == nrow(info.gp.data) == 13
    @test models == original_models
    @test info.history.phase == [:minimum, :minimum, :minimum, :maximum, :maximum]
    @test info.history.call == collect(9:13)
    @test nrow(unique(info.training_data[:, Not(:_cabo_g)])) == 13
    @test info.training_data._cabo_g == info.samples.g
    @test info.samples.g ≈ 2 .* (info.samples.x .+ info.samples.shift)
    @test all(info.samples.offset .== 0)
    @test info.epistemic_coordinates == [(:shift, :value), (:offset, :value)]
    @test !(:_cabo_e2 in info.gp.inputs) # constants do not create zero-variance GP dimensions
    @test xmin == [1.0, 0.0]
    @test xmax == [-1.0, 0.0]
    @test pf.lb ≈ cdf(Normal(), -1) atol = 0.08
    @test pf.ub ≈ cdf(Normal(), 1) atol = 0.08
    @test 0 <= pf.lb <= pf.ub <= 1
    @test method.jitter <= info.sampling_jitter <= 0.01

    # Initial observations must survive every refit and the minimum/maximum transition.
    initial = UncertaintyQuantification._cabo_design(
        MersenneTwister(19),
        UncertaintyQuantification._cabo_input_map(inputs), 8
    )
    @test info.training_data[1:8, Not(:_cabo_g)] == initial
    repeated = probability_of_failure(
        models, df -> df.g, inputs, method;
        rng = MersenneTwister(19), return_diagnostics = true
    )
    @test repeated[4].training_data == info.training_data
    @test repeated[4].history == info.history
    @test repeated[4].probability_draws == info.probability_draws
    short_result = probability_of_failure(
        models, df -> df.g, inputs,
        CABO(
            initial_points = 4, lower_bound_iterations = 0, upper_bound_iterations = 0,
            aleatory_samples = 16, posterior_draws = 8, grid_points = 2, learn_hyperparameters = false
        );
        rng = MersenneTwister(2)
    )
    @test length(short_result) == 3
    @test short_result[1] isa Interval
end

@testitem "CABO: supplied designs and finite candidate pool" begin
    using DataFrames
    using Random
    inputs = [RandomVariable(Uniform(-1, 1), :x), IntervalVariable(-1, 1, :shift)]
    initial = [0.25 0.25; 0.75 0.75]
    aleatory = reshape([0.1, 0.9], 2, 1)
    saved_initial, saved_aleatory = copy(initial), copy(aleatory)
    method = CABO(
        initial_points = 2, lower_bound_iterations = 1, upper_bound_iterations = 1,
        aleatory_samples = 2, posterior_draws = 8, learn_hyperparameters = false,
        epistemic_candidates = reshape([0.0], 1, 1)
    )
    calls = Ref(0)
    model = Model(df -> (calls[] += nrow(df); df.x .+ df.shift), :g)
    _, xmin, xmax, info = probability_of_failure(
        model, df -> df.g, inputs, method;
        rng = MersenneTwister(7), initial_design = initial, aleatory_design = aleatory,
        return_diagnostics = true
    )
    @test calls[] == info.calls == 4
    @test info.samples.x[1:2] == [-0.5, 0.5]
    @test info.samples.shift[1:2] == [-0.5, 0.5]
    @test sort(info.samples.x[3:4]) ≈ [-0.8, 0.8]
    @test info.samples.shift[3:4] == [0.0, 0.0]
    @test xmin == xmax == [0.0]
    @test info.aleatory_samples._cabo_a1 ≈ [-0.8, 0.8]
    @test initial == saved_initial && aleatory == saved_aleatory
    @test all(==(1), info.history.candidate)

    calls[] = 0
    too_many = CABO(
        initial_points = 2, lower_bound_iterations = 3, upper_bound_iterations = 0,
        aleatory_samples = 2, posterior_draws = 8, epistemic_candidates = reshape([0.0], 1, 1)
    )
    @test_throws ArgumentError probability_of_failure(
        model, df -> df.g, inputs, too_many;
        initial_design = initial, aleatory_design = aleatory
    )
    overlaps = CABO(
        initial_points = 2, lower_bound_iterations = 3, upper_bound_iterations = 0,
        aleatory_samples = 2, posterior_draws = 8, epistemic_candidates = reshape([-1.0, 1.0], 2, 1)
    )
    @test_throws ArgumentError probability_of_failure(
        model, df -> df.g, inputs, overlaps;
        initial_design = [0.1 0.0; 0.9 1.0], aleatory_design = aleatory
    )
    for bad in (zeros(3, 2), zeros(2, 2), fill(0.5, 2, 2), [0.2 0.5; 0.8 0.5])
        @test_throws ArgumentError probability_of_failure(
            model, df -> df.g, inputs, method;
            initial_design = bad, aleatory_design = aleatory
        )
    end
    @test_throws ArgumentError probability_of_failure(
        model, df -> df.g, inputs, method;
        initial_design = initial, aleatory_design = fill(0.5, 2, 1)
    )
    @test calls[] == 0
end

@testitem "CABO: probability EI and strict failure event" begin
    using DataFrames
    using Random
    UQ = UncertaintyQuantification
    probabilities = [0.0 0.4; 0.3 0.3; 0.2 0.8]
    @test UQ._cabo_expected_improvement(probabilities, true) ≈ [0.1, 0.0, 0.0]
    @test UQ._cabo_expected_improvement(probabilities, false) ≈ [0.0, 0.0, 0.15]
    inputs = [RandomVariable(Normal(), :x), IntervalVariable(-1, 1, :shift)]
    layout = UQ._cabo_input_map(inputs)
    method = CABO(
        initial_points = 4, lower_bound_iterations = 0, upper_bound_iterations = 0,
        aleatory_samples = 16, posterior_draws = 8, learn_hyperparameters = false
    )
    for value in (-10.0, 10.0)
        pf, _, _ = probability_of_failure(
            UQModel[], df -> fill(value, nrow(df)), inputs, method;
            rng = MersenneTwister(7)
        )
        @test pf.lb == pf.ub == (value < 0 ? 1.0 : 0.0)
    end
end

@testitem "CABO: reject unsupported or invalid configurations before simulation" begin
    using DataFrames
    using Random
    @test_throws ArgumentError CABO(initial_points = 1)
    @test_throws ArgumentError CABO(lower_bound_iterations = -1)
    @test_throws ArgumentError CABO(posterior_draws = 1)
    @test_throws ArgumentError CABO(aleatory_samples = 0)
    @test_throws ArgumentError CABO(σ² = NaN)
    @test_throws ArgumentError CABO(jitter = 0)
    @test_throws ArgumentError CABO(epistemic_candidates = [1, 2])
    calls = Ref(0)
    model = Model(df -> (calls[] += nrow(df); zeros(nrow(df))), :g)
    inputs = [RandomVariable(Normal(), :x), IntervalVariable(-1, 1, :shift)]
    for candidates in (zeros(0, 1), zeros(3, 2), reshape([NaN], 1, 1), reshape([2.0], 1, 1))
        method = CABO(epistemic_candidates = candidates)
        @test_throws ArgumentError probability_of_failure(model, df -> df.g, inputs, method)
    end
    @test_throws ArgumentError probability_of_failure(model, df -> df.g, RandomVariable(Normal(), :x), CABO())
    @test_throws ArgumentError probability_of_failure(model, df -> df.g, IntervalVariable(-1, 1, :x), CABO())
    @test_throws ArgumentError probability_of_failure(
        model, df -> df.g,
        [RandomVariable(Bernoulli(0.5), :x), IntervalVariable(-1, 1, :shift)], CABO()
    )
    @test_throws ArgumentError probability_of_failure(
        model, df -> df.g,
        [inputs; IntervalVariable(-1, 1, :x)], CABO()
    )
    @test_throws ArgumentError probability_of_failure(
        model, df -> df.g,
        [RandomVariable(Normal(), :x), IntervalVariable(-Inf, 1, :shift)], CABO()
    )
    @test_throws ArgumentError probability_of_failure(
        model, df -> df.g,
        [inputs; IntervalVariable(-1, 1, :b); IntervalVariable(-1, 1, :c)], CABO()
    )
    @test calls[] == 0
    tiny = CABO(
        initial_points = 2, lower_bound_iterations = 0, upper_bound_iterations = 0,
        aleatory_samples = 4, posterior_draws = 2
    )
    @test_throws ArgumentError probability_of_failure(UQModel[], df -> 1.0, inputs, tiny)
    @test_throws ArgumentError probability_of_failure(UQModel[], df -> fill(NaN, nrow(df)), inputs, tiny)
end

@testitem "Hong CABO example: independent quadrature reference" begin
    using QuadGK
    # Validate the reference directly without executing the interactive demo.
    reference_probability(x3, mu) = quadgk(
        z -> pdf(Normal(), z) *
            cdf(Normal(mu, 2), (z + x3)^2 - 7), -Inf, Inf; rtol = 1.0e-10
    )[1]
    # A second integral conditions on X2 instead of X1.
    x3, mu = 0.7, -0.4
    other, _ = quadgk(-7.0, Inf; rtol = 1.0e-9) do x2
        r = sqrt(max(7 + x2, 0))
        pdf(Normal(mu, 2), x2) * (cdf(Normal(), -r - x3) + ccdf(Normal(), r - x3))
    end
    other += cdf(Normal(mu, 2), -7)
    @test reference_probability(x3, mu) ≈ other rtol = 1.0e-8
    @test reference_probability(-x3, mu) ≈ other rtol = 1.0e-8
    @test reference_probability(0.0, 1.0) ≈ 0.008989396434 rtol = 1.0e-8
    @test reference_probability(2.0, -2.0) ≈ 0.430793328967 rtol = 1.0e-8
end
