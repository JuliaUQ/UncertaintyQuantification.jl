using UncertaintyQuantification

Δt = Parameter(0.02, :dt)
t = collect(0:Δt.value:10)
timeSteps = Parameter(length(t), :timeSteps)

ω = collect(0:0.6:150)

cp = CloughPenzien(ω, 0.1, 0.8π, 0.6, 8π, 0.6)

gm = SpectralRepresentation(cp, t, :gm)
gm_model = StochasticProcessModel(gm)

sourcedir = joinpath(pwd(), "demo/models/opensees-dynamic-cantilever-column")

sourcefile = ["cantilever-column.tcl", "ground-motion.dat"]

numberformats = Dict(:dt => ".8e", :gm => ".8e")

workdir = joinpath(pwd(), "workdir-cantilever-column")

max_abs_disp = Extractor(
    base -> begin
        file = joinpath(base, "displacement.out")
        data = DelimitedFiles.readdlm(file, ' ')

        return maximum(abs.(data[:, 2]))
    end, :max_abs_disp
)

disp = Extractor(
    base -> begin
        file = joinpath(base, "displacement.out")
        data = DelimitedFiles.readdlm(file, ' ')

        return data[:, 2]
    end, :disp
)

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

ext = ExternalModel(
    sourcedir, sourcefile, [max_abs_disp, disp, sim_time], opensees; workdir = workdir, formats = numberformats
)

models = [gm_model, ext]

pf, mc_std, samples = probability_of_failure(models, df -> 200 .- df.max_abs_disp, [Δt, timeSteps, gm], MonteCarlo(100))
println("Probability of failure: $pf")

plot(t, samples.gm[1] ./ (maximum(abs.(samples.gm[1]))); label = "Stochastic ground motion acceleration", xlabel = "time in s", ylabel = "Normalized acceleration and displacement")
plot!(samples.sim_time[1], samples.disp[1] ./ (maximum(abs.(samples.disp[1]))); label = "Displacement at top node", linewidth = 2)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
