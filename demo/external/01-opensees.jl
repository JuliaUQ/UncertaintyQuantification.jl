using UncertaintyQuantification
using DelimitedFiles # required to extract the simulation output

E = RandomVariable(Normal(1000, 5), :E) # Young's modulus

sourcedir = joinpath(pwd(), "demo/models")

sourcefile = "supported-beam.tcl"

numberformats = Dict(:E => ".8e")

workdir = joinpath(pwd(), "supported-beam")

disp = Extractor(base -> begin
    file = joinpath(base, "displacement.out")
    data = readdlm(file, ' ')

    return maximum(abs.(data[:, 2]))
end, :disp)

opensees = Solver(
    "OpenSees", # path to OpenSees binary, here we expect it to be available on the system PATH
    "supported-beam.tcl";
    args="", # (optional) extra arguments passed to the solver
)

ext = ExternalModel(
    sourcedir, sourcefile, disp, opensees; workdir=workdir, formats=numberformats
)

pf, σ_pf, samples = probability_of_failure(ext, df -> 0.35 .- df.disp, E, MonteCarlo(1000))

println("Probability of failure: $pf")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
