@testitem "Model" setup = [Import] begin
    input = DataFrame(; a=1, b=2)

    model = Model(df -> df.a + 2 * df.b, :c)
    @test isa(model, Model)

    evaluate!(model, input)
    @test input.c == [5]
    @test model(input) == [5]
end

@testitem "ParallelModel" setup = [Import] begin
    procs = addprocs(2)

    @everywhere using UncertaintyQuantification

    input = DataFrame(; a=rand(6), b=rand(6))

    model = ParallelModel(df -> df.a + 2 * df.b, :c)

    @test isa(model, ParallelModel)

    try
        evaluate!(model, input)
        @test input.c == input.a .+ 2 .* input.b
        @test model(input) == input.a .+ 2 .* input.b
    finally
        rmprocs(procs)
    end
end
