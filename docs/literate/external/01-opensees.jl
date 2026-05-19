#===
# External Solvers

## OpenSees supported beam

In this example we will perform the reliability analysis of a supported beam using the open-source finite element software [OpenSees](https://opensees.berkeley.edu/).
The example definition can be found [here](https://opensees.berkeley.edu/wiki/index.php/Simply_supported_beam_modeled_with_two_dimensional_solid_elements)
We will be modeling the Young's modulus of the `ElasticIsotropic` material in the OpenSees model as a [`RandomVariable`](@ref).
===#

using UncertaintyQuantification
using DelimitedFiles # required to extract the simulation output

E = RandomVariable(Normal(1000, 5), :E) # Young's modulus

# Source/Extra files are expected to be in this folder
sourcedir = joinpath(pwd(), "demo/models")

# These files will be rendere through Mustache.jl and have values injected
sourcefile = "supported-beam.tcl"

# Dictionary to map format Strings (Format.jl) to variables
numberformats = Dict(:E => ".8e")

# UQ will create subfolders in here to run the solver and store the results
workdir = joinpath(pwd(), "supported-beam")

# Read output file and compute maximum (absolute) displacement.
# The input `base` of the function used to construct the [`Extractor`](@ref) is the working directory for the current sample.
disp = Extractor(base -> begin
    file = joinpath(base, "displacement.out")
    data = readdlm(file, ' ')

    return maximum(abs.(data[:, 2]))
end, :disp)

# Define the solver
opensees = Solver(
    "OpenSees", # path to OpenSees binary, here we expect it to be available on the system PATH
    "supported-beam.tcl";
    args="", # (optional) extra arguments passed to the solver
)

# Put everything together to construct the external model
ext = ExternalModel(
    sourcedir, sourcefile, disp, opensees; workdir=workdir, formats=numberformats
)

# Run the probability of failure analysis with 1000 samples
pf, σ_pf, samples = probability_of_failure(ext, df -> 0.35 .- df.disp, E, MonteCarlo(1000))

println("Probability of failure: $pf")

# Running large simulations with an external model in series can be numerically demanding. The next example shows how to run the same example but execute the model in parallel.
