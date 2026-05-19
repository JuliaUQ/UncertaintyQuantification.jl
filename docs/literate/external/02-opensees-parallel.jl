
#===

## OpenSees supported beam parallel
===#

# load the Distributed module
using Distributed

# add six local workers
addprocs(6)

# The setup of the example is enclosed in the @everywhere macro. This ensures, that the main process and all worker processes have loaded the necessary modules and received all defined variables and functions.
# Apart from this, the setup is identical to the serial example.

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

# The analysis is then executed on the main Julia process. When executing the model, the calls to the external solver will automatically be distributed to all available workers.

pf, σ_pf, samples  = probability_of_failure(ext, df -> 0.35 .- df.disp, E, MonteCarlo(1000))

println("Probability of failure: $pf")

#===
!!! note
    Local parallelisation will quickly reach it's limits for large and complex models. Making use of  cluster thrugh the [`SlurmInterface`](@ref) is strongly recommended.
===#
