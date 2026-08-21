using UncertaintyQuantification

E = RandomVariable(Uniform(0.3, 0.7), :E)
R1 = RandomVariable(Uniform(8, 14), :R1)

sourcedir = joinpath(pwd(), "demo/models")

sourcefile = "openmc_TBR.py"

numberformats = Dict(:E => ".8e")

workdir = joinpath(pwd(), "openmc_TBR")

TBR = Extractor(
    base -> begin
        file = joinpath(base, "openmc.out")
        line = readline(file)
        tbr = parse(Float64, split(line, " ")[1])

        return tbr
    end, :TBR
)

openmc = Solver(
    "python3", # path to python3 binary
    "openmc_TBR.py";
    args = "", # (optional) extra arguments passed to the solver
)

options = Dict(
    "job-name" => "UQ_slurm",
    "account" => "EXAMPLE-0001-CPU",
    "partition" => "cpu_partition",
    "nodes" => "1",
    "ntasks" => "1",
    "time" => "00:05:00",
)

slurm = SlurmInterface(
    options;
    throttle = 50,
    extras = ["module load openmc", "source ~/.virtualenvs/openmc/bin/activate"],
)

ext = ExternalModel(
    sourcedir,
    sourcefile,
    TBR,
    openmc;
    workdir = workdir,
    formats = numberformats,
    scheduler = slurm,
)

function limitstate(df)
    return reduce(vcat, df.TBR) .- 1
end

@time pf, σ, samples = probability_of_failure(ext, limitstate, [E, R1], MonteCarlo(5000))
println("Probability of failure: $pf")

using StatsBase

TBR = samples.TBR
TBR_mean = mean(TBR)
TBR_std = std(TBR)
lower_quantile = quantile(TBR, 0.025)
upper_quantile = quantile(TBR, 0.975)
println("TBR mean: $TBR_mean, TBR std: $TBR_std, TBR 95%: [$lower_quantile, $upper_quantile]")

subset = SubSetInfinity(800, 0.1, 10, 0.5)

@time pf, std, samples = probability_of_failure(ext, limitstate, [E, R1], subset)
println("Probability of failure: $pf")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
