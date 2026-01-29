using DataFrames
using Distributed
using HCubature
using HypothesisTests
using InteractiveUtils
using QuasiMonteCarlo
using Random
using StatsBase: fit, Histogram, corkendall
using LinearAlgebra: diag
using Test
using UncertaintyQuantification

include("inputs/empiricaldistribution.jl")
include("dynamics/psd.jl")
include("inputs/parameter.jl")
include("inputs/gaussianmixtures.jl")
include("inputs/jointdistribution.jl")
include("inputs/imprecise/interval.jl")
include("inputs/imprecise/p-box.jl")
include("inputs/randomvariables/randomvariable.jl")
include("inputs/randomvariables/distributionparameters.jl")
include("inputs/jointdistribution.jl");
include("inputs/inputs.jl")
include("inputs/copulas/gaussian.jl")
include("inputs/stochasticprocesses/spectralrepresentation.jl")
include("inputs/stochasticprocesses/models.jl")

include("models/external/solvers.jl")
include("models/external/externalmodel.jl")
include("models/model.jl")
include("models/polyharmonicspline.jl")
include("models/pce/pcebases.jl")
include("models/pce/polynomialchaosexpansion.jl")
include("models/responsesurface.jl")
include("models/imprecise/propagation.jl")

include("modelupdating/bayesianupdating.jl")
include("modelupdating/bayesianMAP.jl")

include("reliability/form.jl")
include("reliability/probabilityoffailure.jl")
include("reliability/probabilityoffailure_imprecise.jl")

include("sensitivity/gradient.jl")
include("sensitivity/sobolindices.jl")
include("sensitivity/kucherenkoindices.jl")

include("simulations/doe.jl")
include("simulations/montecarlo.jl")
include("simulations/subset.jl")

include("util/fourier-transform.jl")

if Sys.islinux()
    HPC = false
    HPC_account = "HPC_account_1"
    HPC_partition = "CPU_partition"
    if "HPC" in ARGS
        HPC = true
        HPC_account = ARGS[2]
        HPC_partition = ARGS[3]
        @warn "Running a slurm test with HPC=ON, using account $HPC_account and partition $HPC_partition. Several (20) small 1-task calculations will be submitted to slurm for testing in different job array configuations."
    end

    if HPC == false && !occursin("test/test_utilities", ENV["PATH"])
        @warn "For slurm test to pass on Linux, test_utilities/sbatch must be added to PATH"
        @warn "sbatch command line tool may use the fake test_utilities/sbatch"
    end
    include("hpc/slurm.jl")
end
