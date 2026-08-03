using Distributed

addprocs(6)

@everywhere begin
    using UncertaintyQuantification, DelimitedFiles

    E = RandomVariable(Normal(1000, 5), :E)

    sourcedir = joinpath(pwd(), "demo/models")

    sourcefile = "supported-beam.tcl"

    numberformats = Dict(:E => ".8e")

    workdir = joinpath(pwd(), "supported-beam")

    disp = Extractor(base -> begin
        file = joinpath(base, "displacement.out")
        data = readdlm(file, ' ')

        return maximum(abs.(data[:, 2]))
    end, :disp)

    opensees = Solver("OpenSees", "supported-beam.tcl"; args="")

    ext = ExternalModel(
        sourcedir, sourcefile, disp, opensees; workdir=workdir, formats=numberformats
    )
end

pf, σ_pf, samples  = probability_of_failure(ext, df -> 0.35 .- df.disp, E, MonteCarlo(1000))

println("Probability of failure: $pf")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
