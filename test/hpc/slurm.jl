include("../test_utilities/read_write_utils.jl")

@testset "Slurm" begin
    sourcedir = tempdir()
    sourcefile = ["radius.jl"]

    # create source file for radius solver
    open(joinpath(sourcedir, "radius.jl"), "w") do input
        println(input, "x = {{{:x}}}")
        println(input, "y = {{{:y}}}")
        println(input, "z = sqrt(x^2+y^2)")
        println(input, "write(\"out.txt\", string(z))")
    end

    numberformats = Dict(:x => ".8e", :* => ".8e")

    radius = Extractor(
        base -> begin
            return parse(Float64, readline(joinpath(base, "out.txt")))
        end, :r
    )

    binary = joinpath(Sys.BINDIR, "julia")
    solver = Solver(binary, "radius.jl")

    open(joinpath(sourcedir, "extra.txt"), "w") do input
        println(input, "This is an extra file")
    end

    options = Dict(
        "job-name" => "my_test_job",
        "partition" => HPC_partition,
        "time" => "10:00:00",
        "mem-per-cpu" => "100",
        "nodes" => "2",
        "ntasks" => "32",
    )

    @test_logs (
        :warn,
        "Omitting account or partition may cause your simulation to fail if your scheduler does not assign defaults.",
    ) SlurmInterface(options)

    options = Dict(
        "job-name" => "my_test_job",
        "account" => HPC_account,
        "time" => "10:00:00",
        "mem-per-cpu" => "100",
        "nodes" => "2",
        "ntasks" => "32",
    )

    @test_logs (
        :warn,
        "Omitting account or partition may cause your simulation to fail if your scheduler does not assign defaults.",
    ) SlurmInterface(options)

    options = Dict(
        "job-name" => "my_test_job",
        "account" => HPC_account,
        "partition" => HPC_partition,
        "array" => "[1-199]%2",
        "time" => "10:00:00",
        "mem-per-cpu" => "100",
        "nodes" => "2",
        "ntasks" => "32",
    )

    @test_throws ErrorException("Do not pass array as an option.") SlurmInterface(options)

    options = Dict(
        "job-name" => "my_test_job",
        "account" => HPC_account,
        "partition" => HPC_partition,
        "time" => "10:00:00",
        "mem-per-cpu" => "100",
        "nodes" => "2",
        "ntasks" => "32",
    )

    @testset "Setup jobs" begin
        workdir = tempname()
        mkdir(workdir)

        @testset "unbatched" begin
            slurm = SlurmInterface(options)

            ext = ExternalModel(
                sourcedir,
                sourcefile,
                radius,
                solver;
                workdir=tempname(),
                formats=numberformats,
                extras="extra.txt",
                scheduler=slurm,
            )

            UncertaintyQuantification.setup_hpc_jobs(slurm, ext, 100, workdir)

            generated_file = joinpath(workdir, "slurm_array.sh")

            @test isfile(generated_file)
            @test isline(generated_file, "#SBATCH --account=$HPC_account")
            @test isline(generated_file, "#SBATCH --partition=$HPC_partition")
            @test isline(generated_file, "#SBATCH --nodes=2")
            @test isline(generated_file, "#SBATCH --ntasks=32")
            @test isline(generated_file, "#SBATCH --array=[1-100]")

            @test isline(generated_file, "$(solver.path) $(solver.source)")
        end

        @testset "batched" begin
            slurm = SlurmInterface(
                options;
                throttle=2,
                batchsize=10,
                extras=["load something", "load something else"],
            )

            ext = ExternalModel(
                sourcedir,
                sourcefile,
                radius,
                solver;
                workdir=tempname(),
                formats=numberformats,
                extras="extra.txt",
                scheduler=slurm,
            )

            UncertaintyQuantification.setup_hpc_jobs(slurm, ext, 100, workdir)

            for batch in 1:10
                generated_file = joinpath(workdir, "slurm_array-$batch.sh")

                a = (batch - 1) * 10 + 1
                b = min(batch * 10, 100)

                @test isfile(generated_file)
                @test isline(generated_file, "#SBATCH --job-name=my_test_job")
                @test isline(generated_file, "#SBATCH --array=[$a-$b]%2")
                @test isline(generated_file, "#SBATCH --nodes=2")
                @test isline(generated_file, "#SBATCH --ntasks=32")
                @test isline(generated_file, "#SBATCH --time=10:00:00")
                @test isline(generated_file, "#SBATCH --mem-per-cpu=100")
                @test isline(generated_file, "load something")
                @test isline(generated_file, "load something else")
            end
        end
    end

    options = Dict(
        "job-name" => "my_test_job",
        "account" => HPC_account,
        "partition" => HPC_partition,
        "time" => "00:01:00",
        "mem-per-cpu" => "100",
        "nodes" => "1",
        "ntasks" => "1",
    )

    @testset "Run jobs" begin
        @testset "unbatched" begin
            slurm = SlurmInterface(options)

            ext = ExternalModel(
                sourcedir,
                sourcefile,
                radius,
                solver;
                workdir="hpc/test_dir",
                formats=numberformats,
                extras="extra.txt",
                scheduler=slurm,
            )

            x = RandomVariable(Uniform(0, 1), :x)
            y = RandomVariable(Uniform(0, 1), :y)

            df = sample([x, y], 10)

            evaluate!(ext, df)

            @test isapprox(df.r, sqrt.(df.x .^ 2 + df.y .^ 2))
        end

        @testset "batched" begin
            slurm = SlurmInterface(options; batchsize=4)

            ext = ExternalModel(
                sourcedir,
                sourcefile,
                radius,
                solver;
                workdir="hpc/test_dir",
                formats=numberformats,
                extras="extra.txt",
                scheduler=slurm,
            )

            x = RandomVariable(Uniform(0, 1), :x)
            y = RandomVariable(Uniform(0, 1), :y)

            df = sample([x, y], 10)

            evaluate!(ext, df)

            @test isapprox(df.r, sqrt.(df.x .^ 2 + df.y .^ 2))
        end
    end
end

#############################################

@testset "Slurm" begin
    sourcedir = tempdir()
    sourcefile = ["radius.jl"]

    # create source file for radius solver
    open(joinpath(sourcedir, "radius.jl"), "w") do input
        println(input, "x = {{{:x}}}")
        println(input, "y = {{{:y}}}")
        println(input, "z = sqrt(x^2+y^2)")
        println(input, "write(\"out.txt\", string(z))")
    end

    numberformats = Dict(:x => ".8e", :* => ".8e")

    radius = Extractor(
        base -> begin
            return parse(Float64, readline(joinpath(base, "out.txt")))
        end, :r
    )

    binary = joinpath(Sys.BINDIR, "julia")
    solver = Solver(binary, "radius.jl")

    open(joinpath(sourcedir, "extra.txt"), "w") do input
        println(input, "This is an extra file")
    end

    options = Dict(
        "job-name" => "my_test_job",
        "partition" => HPC_partition,
        "time" => "10:00:00",
        "mem-per-cpu" => "100",
        "nodes" => "2",
        "ntasks" => "32",
    )

    @test_logs (
        :warn,
        "Omitting account or partition may cause your simulation to fail if your scheduler does not assign defaults.",
    ) SlurmInterface(options)

    options = Dict(
        "job-name" => "my_test_job",
        "account" => HPC_account,
        "time" => "10:00:00",
        "mem-per-cpu" => "100",
        "nodes" => "2",
        "ntasks" => "32",
    )

    @test_logs (
        :warn,
        "Omitting account or partition may cause your simulation to fail if your scheduler does not assign defaults.",
    ) SlurmInterface(options)

    options = Dict(
        "job-name" => "my_test_job",
        "account" => HPC_account,
        "partition" => HPC_partition,
        "array" => "[1-199]%2",
        "time" => "10:00:00",
        "mem-per-cpu" => "100",
        "nodes" => "2",
        "ntasks" => "32",
    )

    @test_throws ErrorException("Do not pass array as an option.") SlurmInterface(options)

    options = Dict(
        "job-name" => "my_test_job",
        "account" => HPC_account,
        "partition" => HPC_partition,
        "time" => "10:00:00",
        "mem-per-cpu" => "100",
        "nodes" => "2",
        "ntasks" => "32",
    )

    @testset "Setup jobs" begin
        workdir = tempname()
        mkdir(workdir)

        @testset "unbatched" begin
            slurm = SlurmInterface(options)

            ext = ExternalModel(
                sourcedir,
                sourcefile,
                radius,
                solver;
                workdir=tempname(),
                formats=numberformats,
                extras="extra.txt",
                scheduler=slurm,
            )

            UncertaintyQuantification.setup_hpc_jobs(slurm, ext, 100, workdir)

            generated_file = joinpath(workdir, "slurm_array.sh")

            @test isfile(generated_file)
            @test isline(generated_file, "#SBATCH --account=$HPC_account")
            @test isline(generated_file, "#SBATCH --partition=$HPC_partition")
            @test isline(generated_file, "#SBATCH --nodes=2")
            @test isline(generated_file, "#SBATCH --ntasks=32")
            @test isline(generated_file, "#SBATCH --array=[1-100]")

            @test isline(generated_file, "$(solver.path) $(solver.source)")
        end

        @testset "batched" begin
            slurm = SlurmInterface(
                options;
                throttle=2,
                batchsize=10,
                extras=["load something", "load something else"],
            )

            ext = ExternalModel(
                sourcedir,
                sourcefile,
                radius,
                solver;
                workdir=tempname(),
                formats=numberformats,
                extras="extra.txt",
                scheduler=slurm,
            )

            UncertaintyQuantification.setup_hpc_jobs(slurm, ext, 100, workdir)

            for batch in 1:10
                generated_file = joinpath(workdir, "slurm_array-$batch.sh")

                a = (batch - 1) * 10 + 1
                b = min(batch * 10, 100)

                @test isfile(generated_file)
                @test isline(generated_file, "#SBATCH --job-name=my_test_job")
                @test isline(generated_file, "#SBATCH --array=[$a-$b]%2")
                @test isline(generated_file, "#SBATCH --nodes=2")
                @test isline(generated_file, "#SBATCH --ntasks=32")
                @test isline(generated_file, "#SBATCH --time=10:00:00")
                @test isline(generated_file, "#SBATCH --mem-per-cpu=100")
                @test isline(generated_file, "load something")
                @test isline(generated_file, "load something else")
            end
        end
    end

    options = Dict(
        "job-name" => "my_test_job",
        "account" => HPC_account,
        "partition" => HPC_partition,
        "time" => "00:01:00",
        "mem-per-cpu" => "100",
        "nodes" => "1",
        "ntasks" => "1",
    )

    @testset "Run jobs" begin
        @testset "unbatched" begin
            slurm = SlurmInterface(options)

            ext = ExternalModel(
                sourcedir,
                sourcefile,
                radius,
                solver;
                workdir="hpc/test_dir",
                formats=numberformats,
                extras="extra.txt",
                scheduler=slurm,
            )

            x = RandomVariable(Uniform(0, 1), :x)
            y = RandomVariable(Uniform(0, 1), :y)

            df = sample([x, y], 10)

            evaluate!(ext, df)

            @test isapprox(df.r, sqrt.(df.x .^ 2 + df.y .^ 2))
        end

        @testset "batched" begin
            slurm = SlurmInterface(options; batchsize=4)

            ext = ExternalModel(
                sourcedir,
                sourcefile,
                radius,
                solver;
                workdir="hpc/test_dir",
                formats=numberformats,
                extras="extra.txt",
                scheduler=slurm,
            )

            x = RandomVariable(Uniform(0, 1), :x)
            y = RandomVariable(Uniform(0, 1), :y)

            df = sample([x, y], 10)

            evaluate!(ext, df)

            @test isapprox(df.r, sqrt.(df.x .^ 2 + df.y .^ 2))
        end
    end
end
