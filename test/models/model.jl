@testitem "Model" setup = [TestSetup] begin
    input = DataFrame(; a = 1, b = 2)

    model = Model(df -> df.a + 2 * df.b, :c)
    @test isa(model, Model)

    evaluate!(model, input)
    @test input.c == [5]
end

#TODO: replace with a testitem if possible
@testset "Model distributed=true" begin
    procs = addprocs(2)

    @everywhere using UncertaintyQuantification

    input = DataFrame(; a = rand(6), b = rand(6))

    model = Model(df -> df.a + 2 * df.b, :c; distributed = true)

    try
        evaluate!(model, input)
        @test input.c == input.a .+ 2 .* input.b
    finally
        rmprocs(procs)
    end
end
