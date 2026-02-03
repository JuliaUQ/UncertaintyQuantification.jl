include("../../test_utilities/read_write_utils.jl")

@testset "ExternalModel" begin
    sourcedir = tempdir()
    sourcefile = ["radius.jl"]

    # create source file for radius solver
    open(joinpath(sourcedir, "radius.jl"), "w") do input
        println(input, "x = {{{:x}}}")
        println(input, "y = {{{:y}}}")
        println(input, "z = sqrt(x^2+y^2)")
        println(input, "write(\"out.txt\", string(z))")
    end

    # create source file for squared solver
    open(joinpath(sourcedir, "squared.jl"), "w") do input
        println(input, "x = parse(Float64, readline(\"out.txt\"))")
        println(input, "y = x^2")
        println(input, "write(\"out-squared.txt\", string(y))")
    end

    numberformats = Dict(:x => ".8e", :* => ".8e")

    radius = Extractor(
        base -> begin
            return parse(Float64, readline(joinpath(base, "out.txt")))
        end, :r
    )

    binary = joinpath(Sys.BINDIR, "julia")

    solver = Solver(binary, "radius.jl")
    solver2 = Solver(binary, "squared.jl")

    open(joinpath(sourcedir, "extra.txt"), "w") do input
        println(input, "This is an extra file")
    end

    x = RandomVariable(Uniform(0, 1), :x)
    y = RandomVariable(Uniform(0, 1), :y)

    df = sample([x, y], 5)

    @testset "Directory management" begin
        dirname = tempname()
        ext = ExternalModel(
            sourcedir,
            sourcefile,
            radius,
            solver;
            workdir=tempname(),
            formats=numberformats,
            extras="extra.txt",
        )

        UncertaintyQuantification.makedirectory(ext, df[1, :], dirname)

        formatted_inputs = UncertaintyQuantification.formatinputs(df[1, :], numberformats)

        @test isdir(dirname)
        @test isfile(joinpath(dirname, "radius.jl"))
        @test isfile(joinpath(dirname, "extra.txt"))
        @test !isnotanywhere(joinpath(dirname, "radius.jl"), formatted_inputs[1])
        @test !isnotanywhere(joinpath(dirname, "radius.jl"), formatted_inputs[2])

        run(ext.solver, dirname)

        result = UncertaintyQuantification.getresult(ext, dirname)

        @test isapprox(result[1], sqrt.(df.x[1] .^ 2 + df.y[1] .^ 2))

        ext = ExternalModel(
            sourcedir,
            sourcefile,
            radius,
            solver;
            workdir=tempname(),
            formats=numberformats,
            extras="",
        )
        UncertaintyQuantification.makedirectory(ext, df[1, :], dirname)
        @test isdir(dirname)
        @test isfile(joinpath(dirname, "radius.jl"))
        @test !isnotanywhere(joinpath(dirname, "radius.jl"), formatted_inputs[1])
        @test !isnotanywhere(joinpath(dirname, "radius.jl"), formatted_inputs[2])
    end

    @testset "No Cleanup" begin
        ext = ExternalModel(
            sourcedir,
            sourcefile,
            radius,
            solver;
            workdir=tempname(),
            formats=numberformats,
            extras="extra.txt",
        )

        evaluate!(ext, df)

        @test length(readdir(readdir(ext.workdir; join=true)[1])) == 5
        @test "extra.txt" in
            readdir(readdir(readdir(ext.workdir; join=true)[1]; join=true)[1])
        @test df.r ≈ sqrt.(df.x .^ 2 .+ df.y .^ 2)
    end

    @testset "Cleanup" begin
        ext = ExternalModel(
            sourcedir,
            sourcefile,
            radius,
            solver;
            formats=numberformats,
            workdir=tempname(),
            cleanup=true,
        )
        evaluate!(ext, df)
        @test length(readdir(readdir(ext.workdir; join=true)[1])) == 0
        @test isapprox(df.r, sqrt.(df.x .^ 2 + df.y .^ 2))
    end

    @testset "Reuse model output" begin
        workdir = tempname()

        ext1 = ExternalModel(
            sourcedir, sourcefile, radius, solver; formats=numberformats, workdir=workdir
        )

        squared = Extractor(
            base -> begin
                return parse(Float64, readline(joinpath(base, "out-squared.txt")))
            end,
            :r2,
        )

        ext2 = ExternalModel(
            sourcedir, ["squared.jl"], squared, solver2; workdir=workdir, cleanup=true
        )

        evaluate!([ext1, ext2], df)
        @test df.r2 ≈ df.r .^ 2
        @test length(readdir(readdir(ext1.workdir; join=true)[1])) == 0
        @test length(readdir(readdir(ext2.workdir; join=true)[1])) == 0
    end
end
