#===
# Stochastic Dynamics Example

## OpenSees earthquake signal on a cantilever column

In this example we will perform the reliability analysis of a multi element cantilever column subjected to artificial stochastic ground motions
using the open-source finite element software [OpenSees](https://opensees.berkeley.edu/).
The example definition can be found [here](https://opensees.berkeley.edu/wiki/index.php?title=Time_History_Analysis_of_a_2D_Elastic_Cantilever_Column)
A stochastic signal is generated using the Clough-Penzien Power Spectral Density and the Spectral Representation Method.
The signal is applied as uniform excitation as "ground motion" to the base of the column structure.

For parallel execution, see the example in [OpenSees supported beam parallel](@ref)
===#

#md using UncertaintyQuantification # hide
#jl using UncertaintyQuantification

# Time discretization for the signal
Δt = Parameter(0.02, :dt)
t = collect(0:Δt.value:10)
timeSteps = Parameter(length(t), :timeSteps)

# Frequency discretization for the Power Spectral Density Function (PSD)
ω = collect(0:0.6:150)
# Definition of Clough Penzien PSD with prescribed parameters
cp = CloughPenzien(ω, 0.1, 0.8π, 0.6, 8π, 0.6)

# Ground motion formulation using the Spectral Representation Method
gm = SpectralRepresentation(cp, t, :gm)
gm_model = StochasticProcessModel(gm)

# Source/Extra files are expected to be in this folder, here the injection file ground-motion.dat is located
sourcedir = joinpath(pwd(), "demo/models/opensees-dynamic-cantilever-column")

# These files will be rendere through Mustach.jl and have values injected, for this example only ground-motion.dat will have time serieses injected
sourcefile = ["cantilever-column.tcl", "ground-motion.dat"]

# Dictionary to map format Strings (Formatting.jl) to variables
numberformats = Dict(:dt => ".8e", :gm => ".8e")

# UQ will create subfolders in here to run the solver and store the results
workdir = joinpath(pwd(), "workdir-cantilever-column")

# Read output file and compute maximum (absolute) displacement
# An extractor is based the working directory for the current sample
max_abs_disp = Extractor(
    base -> begin
        file = joinpath(base, "displacement.out")
        data = DelimitedFiles.readdlm(file, ' ')

        return maximum(abs.(data[:, 2]))
    end, :max_abs_disp
)

# Extractor for the full time series of the displacement at the top node
disp = Extractor(
    base -> begin
        file = joinpath(base, "displacement.out")
        data = DelimitedFiles.readdlm(file, ' ')

        return data[:, 2]
    end, :disp
)

# Extractor for the simulation time
sim_time = Extractor(
    base -> begin
        file = joinpath(base, "displacement.out")
        data = DelimitedFiles.readdlm(file, ' ')

        return data[:, 1]
    end, :sim_time
)


opensees = Solver(
    "OpenSees", # path to OpenSees binary
    "cantilever-column.tcl";
    args = "", # (optional) extra arguments passed to the solver
)

# Define the external model with all needed parameters and attributes
ext = ExternalModel(
    sourcedir, sourcefile, [max_abs_disp, disp, sim_time], opensees; workdir = workdir, formats = numberformats
)

# Define the UQ.jl models used in the analysis
models = [gm_model, ext]

# Simple Monte Carlo simulation with 1000 samples to estimate a failure probability (should be roughly around 10^-2)

#md # ```julia
#md # pf, mc_std, samples = probability_of_failure(models, df -> 200 .- df.max_abs_disp, [Δt, timeSteps, gm], MonteCarlo(100))
#md # ```

#jl pf, mc_std, samples = probability_of_failure(models, df -> 200 .- df.max_abs_disp, [Δt, timeSteps, gm], MonteCarlo(100))
#jl println("Probability of failure: $pf")

# Plotting of single time history

#md # ```julia
#md # plot(t, samples.gm[1]./(maximum(abs.(samples.gm[1]))); label="Stochastic ground motion acceleration", xlabel="time in s", ylabel="Normalized acceleration and displacement")
#md # plot!(samples.sim_time[1], samples.disp[1]./(maximum(abs.(samples.disp[1]))); label="Displacement at top node", linewidth=2)
#md # ```

#jl plot(t, samples.gm[1]./(maximum(abs.(samples.gm[1]))); label="Stochastic ground motion acceleration", xlabel="time in s", ylabel="Normalized acceleration and displacement")
#jl plot!(samples.sim_time[1], samples.disp[1]./(maximum(abs.(samples.disp[1]))); label="Displacement at top node", linewidth=2)

#md # ![Resulting time history](../assets/time-history.svg)
#md # A plot to visualize the stochastic input ground motion acceleration singal and the resulting displacement time series at the top node of the cantilever column.
