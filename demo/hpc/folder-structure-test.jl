using UncertaintyQuantification

sourcedir = joinpath(pwd(), "demo/models")
mkpath(sourcedir)

# Solver-Skript erstellen (wie in folder-structure-test.jl)
sourcefile = "radius.jl"
open(joinpath(sourcedir, sourcefile), "w") do input
    println(input, "x = {{{:x}}}")
    println(input, "y = {{{:y}}}")
    println(input, "z = sqrt(x^2+y^2)")
    println(input, "write(\"out.txt\", string(z))")
end

numberformats = Dict(:x => ".8e", :y => ".8e")

radius = Extractor(
    base -> begin
        return parse(Float64, readline(joinpath(base, "out.txt")))
    end, :r
)

binary = joinpath(Sys.BINDIR, "julia")
print(typeof(binary))
solver = Solver(binary, sourcefile)

x = RandomVariable(Uniform(0, 1), :x)
y = RandomVariable(Uniform(0, 1), :y)

workdir = joinpath(pwd(), "EXT-radius")

ext = ExternalModel(
    sourcedir,
    sourcefile,
    radius,
    solver;
    workdir = workdir,
    formats = numberformats,
)

function limitstate(df)
    return reduce(vcat, df.r) .- 0.1
end


# --- Simulation auswählen (jeweils eine Zeile aktivieren) ---

# MonteCarlo
# sim = MonteCarlo(100)

# LineSampling
# sim = LineSampling(10)

# AdvancedLineSampling
# sim = AdvancedLineSampling(10)

# FORM
sim = FORM()

# ImportanceSampling
# sim = ImportanceSampling(100)

# RadialBasedImportanceSampling
# sim = RadialBasedImportanceSampling(100)




#sim = SubSetSimulation(30, 0.1, 10, Normal())
# sim = SubSetInfinity(30, 0.1, 10, 0.5)
# sim = SubSetInfinityAdaptive(30, 0.1, 10, 1)

@time pf, σ, samples = probability_of_failure(ext, limitstate, [x, y], sim)
println("Probability of failure: $pf")
