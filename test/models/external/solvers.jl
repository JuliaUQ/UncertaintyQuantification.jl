@testitem "Solvers" begin
    binary = joinpath(Sys.BINDIR, "julia")

    solver = Solver(binary, "radius.jl"; args="--project")

    tmp = tempdir()

    open(joinpath(tmp, "radius.jl"), "w") do input
        println(input, "x = 0.5")
        println(input, "y = 0.5")
        println(input, "z = sqrt(x^2+y^2)")
        println(input, "write(\"out.txt\", string(z))")
    end

    run(solver, tmp)

    radius = parse(Float64, readline(joinpath(tmp, "out.txt")))

    @test radius == sqrt(0.5^2 + 0.5^2)

    current_dir = pwd()

    @test_logs (
        :warn,
        "Solver path is not absolute. Make sure this_path_is_not_absolute is on your PATH.",
    ) Solver("this_path_is_not_absolute", "in.txt")

    @test_throws Base.IOError run(
        Solver(joinpath(pwd(), "this/solver/does/not/exist"), "in.txt"), tmp
    )
    @test pwd() == current_dir
end
