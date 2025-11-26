@testset "SubSetSimulation" begin
    proposal = Normal()
    subset = SubSetSimulation(2000, 0.2, 10, proposal)

    @test isa(subset, SubSetSimulation)
    @test subset.n == 2000
    @test subset.target == 0.2
    @test subset.levels == 10
    @test subset.proposal == proposal

    @test_throws ErrorException("proposal must be a symmetric distribution") SubSetSimulation(
        2000, 0.2, 10, Exponential()
    )
    @test_throws ErrorException("proposal must be centered in 0") SubSetSimulation(
        2000, 0.2, 10, Uniform()
    )
    @test_logs (:warn, "A proposal pdf with large variance (≥ 2) can be inefficient.") SubSetSimulation(
        2000, 0.2, 10, Uniform(-4, 4)
    )
    @testset "nextlevelsamples - 1 sample per chain" begin
        x = RandomVariable(Normal(), :x)
        function dummy_model(x)
            if x > 3
                return x
            else
                return 1
            end
        end
        inputs = [x]
        models = Model(df -> dummy_model.(df.x), :y)
        sim = SubSetSimulation(10, 0.1, 10, Uniform(-0.2, 0.2))
        performancefunction = df -> df.y
        threshold = 1
        samples = sample(x, 1)
        evaluate!(models, samples)
        performance = samples[:, :y]

        res = UncertaintyQuantification.nextlevelsamples(
            samples, performance, 1, models, performancefunction, [x], sim
        )
        @test !isnothing(res)
    end
end

@testset "SubSetInfinity" begin
    subset = SubSetInfinity(2000, 0.2, 10, 0.5)

    @test isa(subset, SubSetInfinity)
    @test subset.n == 2000
    @test subset.target == 0.2
    @test subset.levels == 10
    @test subset.s == 0.5

    @test_throws ErrorException("standard deviation must be between 0.0 and 1.0") SubSetInfinity(
        2000, 0.2, 10, 2.0
    )
    @test_throws ErrorException("standard deviation must be between 0.0 and 1.0") SubSetInfinity(
        2000, 0.2, 10, -1.0
    )
end

@testset "SubSetInfinityAdaptive" begin
    subset = SubSetInfinityAdaptive(2000, 0.2, 10, 4, 1, 0.5)

    @test isa(subset, SubSetInfinityAdaptive)
    @test subset.n == 2000
    @test subset.target == 0.2
    @test subset.levels == 10
    @test subset.Na == 4
    @test subset.λ == 1
    @test subset.s == 0.5

    subset = SubSetInfinityAdaptive(2000, 0.2, 10, 4, 1)

    @test isa(subset, SubSetInfinityAdaptive)
    @test subset.n == 2000
    @test subset.target == 0.2
    @test subset.levels == 10
    @test subset.Na == 4
    @test subset.λ == 1
    @test subset.s == 1

    subset = SubSetInfinityAdaptive(2000, 0.2, 10, 4)

    @test isa(subset, SubSetInfinityAdaptive)
    @test subset.n == 2000
    @test subset.target == 0.2
    @test subset.levels == 10
    @test subset.Na == 4
    @test subset.λ == 1
    @test subset.s == 1

    @test_throws ErrorException("standard deviation must be between 0.0 and 1.0") SubSetInfinityAdaptive(
        2000, 0.1, 10, 2, 1, 3
    )

    @test_throws ErrorException(
        "Scaling parameter must be between 0.0 and 1.0. A good initial choice is 1.0"
    ) SubSetInfinityAdaptive(2000, 0.1, 10, 4, 2.0)

    @test_throws ErrorException(
        "Scaling parameter must be between 0.0 and 1.0. A good initial choice is 1.0"
    ) SubSetInfinityAdaptive(2000, 0.1, 10, 4, -2.0)

    @test_throws ErrorException(
        "Number of partitions Na must be a multiple of `n` * `target`"
    ) SubSetInfinityAdaptive(2000, 0.1, 10, 9, 1)

    @test_throws ErrorException("Number of partitions Na must be less than `n` * `target`") SubSetInfinityAdaptive(
        2000, 0.1, 10, 400, 1
    )
end