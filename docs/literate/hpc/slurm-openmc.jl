#===
# High Performance Computing

## OpenMC TBR uncertainty

In this example, we will run [OpenMC](https://openmc.org/), to compute the tritium breeding ratio (TBR) uncertainty, by varying material and geometric properties. This example was taken from the [Fusion Neutronics Workshop](https://github.com/fusion-energy/neutronics-workshop).

At first we need to create an array of random variables, that will be used when evaluating the points that our design produces.It will also define the range of the function we want the design to fit. This is also a good time to declare the function that we are working with.

Here we will vary the model's Li6 enrichment, and the radius of the tungsten layer.
===#

using UncertaintyQuantification

E = RandomVariable(Uniform(0.3, 0.7), :E)
R1 = RandomVariable(Uniform(8, 14), :R1)

# Source/Extra files are expected to be in this folder.
sourcedir = joinpath(pwd(), "demo/models")

# These files will be rendered through Mustache.jl and have values injected.
sourcefile = "openmc_TBR.py"

# Dictionary to map format Strings (Format.jl) to variables.
numberformats = Dict(:E => ".8e")

# UQ will create subfolders in here to run the solver and store the results.
workdir = joinpath(pwd(), "openmc_TBR")

#md # !!! note
#md #     If Slurm is to run the jobs on multiple nodes, all the above folders must be shared by the computing nodes.

# Read output file and compute the Tritium breeding ratio. An extractor is based the working directory for the current sample.
TBR = Extractor(
    base -> begin
        file = joinpath(base, "openmc.out")
        line = readline(file)
        tbr = parse(Float64, split(line, " ")[1])

        return tbr
    end, :TBR
)

# In this example, an OpenMC model is built and run using the Python API. We therefore specify the `python3`` command, and the python file to run.

openmc = Solver(
    "python3", # path to python3 binary
    "openmc_TBR.py";
    args = "", # (optional) extra arguments passed to the solver
)

# slurm  sbatch options
options = Dict(
    "job-name" => "UQ_slurm",
    "account" => "EXAMPLE-0001-CPU",
    "partition" => "cpu_partition",
    "nodes" => "1",
    "ntasks" => "1",
    "time" => "00:05:00",
)
# Slurm interface, passing required machine information. Note in `extras`, we specify the commands we require to run the model (for example, loading modulefiles or data).

slurm = SlurmInterface(
    options;
    throttle = 50,
    extras = ["module load openmc", "source ~/.virtualenvs/openmc/bin/activate"],
)

# With the `SlurmInterface` defined we can assemble the `ExternalModel`.
ext = ExternalModel(
    sourcedir,
    sourcefile,
    TBR,
    openmc;
    workdir = workdir,
    formats = numberformats,
    scheduler = slurm,
)

# Specify a limitstate function, negative value consititutes failure. Here we are interested in P(TBR <= 1).
function limitstate(df)
    return reduce(vcat, df.TBR) .- 1
end

# Finally, we can run a Monte Carlo simulation to obtain the probability of failure.
@time pf, σ, samples = probability_of_failure(ext, limitstate, [E, R1], MonteCarlo(5000))
#jl println("Probability of failure: $pf")

#jl using StatsBase

#jl TBR = samples.TBR
#jl TBR_mean = mean(TBR)
#jl TBR_std = std(TBR)
#jl lower_quantile = quantile(TBR, 0.025)
#jl upper_quantile = quantile(TBR, 0.975)
#jl println("TBR mean: $TBR_mean, TBR std: $TBR_std, TBR 95%: [$lower_quantile, $upper_quantile]")

# The simulation results in `pf = 0.0064` and `σ = 0.001127` for `5000` samples. Also `TBR mean = 1.2404114576444423`, `TBR std = 0.10000460056126671`, and `TBR 95%: [1.0379178446904211, 1.4130216792418262]`.

# We can also obtain the probability of failure with Subset simulation.

subset = SubSetInfinity(800, 0.1, 10, 0.5)

@time pf, std, samples = probability_of_failure(ext, limitstate, [E, R1], subset)
println("Probability of failure: $pf")

# This results in `pf = 0.0053` and `σ = 0.001133` for `2400` samples.
