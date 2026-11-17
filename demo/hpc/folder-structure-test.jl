using UncertaintyQuantification
using Test

function isline(file, string_check)
    for (i, line) in enumerate(eachline(file))
        if (line == string_check)
            return true
        end
    end

    return false
end

# Checks the pattern doesn't exist anywhere
function isnotanywhere(file, string_check)
    for (i, line) in enumerate(eachline(file))
        if (m = match(Regex(string_check), line); m !== nothing)
            return false
        end
    end

    return true
end


sourcedir = tempdir()
sourcefile = ["radius.jl"]
print(sourcedir)
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

dirname = tempname()
ext = ExternalModel(
    sourcedir,
    sourcefile,
    radius,
    solver;
    workdir = tempname(),
    formats = numberformats,
    extras = "extra.txt",
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
    workdir = tempname(),
    formats = numberformats,
    extras = "",
)
UncertaintyQuantification.makedirectory(ext, df[1, :], dirname)
@test isdir(dirname)
@test isfile(joinpath(dirname, "radius.jl"))
@test !isnotanywhere(joinpath(dirname, "radius.jl"), formatted_inputs[1])
@test !isnotanywhere(joinpath(dirname, "radius.jl"), formatted_inputs[2])


E = RandomVariable(Uniform(0.3, 0.7), :E)
R1 = RandomVariable(Uniform(8, 14), :R1)

function limitstate(df)
    return reduce(vcat, df.TBR) .- 1
end

subset = SubSetInfinity(800, 0.1, 10, 0.5)
@time pf, std, samples = probability_of_failure(ext, limitstate, [E, R1], subset)
println("Probability of failure: $pf")
