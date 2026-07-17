using Copulas
using DataFrames
using Distributed
using HCubature
using HypothesisTests
using InteractiveUtils
using QuasiMonteCarlo
using Random
using StatsBase: fit, Histogram, corkendall
using Test
using TestItemRunner
using UncertaintyQuantification

@testsnippet TestSetup begin
    using Copulas
    using DataFrames
    using Distributed
    using HCubature
    using HypothesisTests
    using InteractiveUtils
    using QuasiMonteCarlo
    using Random
    using StatsBase: fit, Histogram, corkendall
end

@testsnippet ReadWriteUtil begin
    # Function to check if (exact) line exits in file
    function isline(file, string_check)
        for (i, line) in enumerate(eachline(file))
            if (line == string_check)
                return true
            end
        end

        return false
    end

    # Checks the pattern doesn't exist anywhere
    function isnotanywhere(file, string_check)
        for (i, line) in enumerate(eachline(file))
            if (m = match(Regex(string_check), line); m !== nothing)
                return false
            end
        end

        return true
    end

end
include("models/model.jl")
@run_package_tests


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
        @warn "Adding test utilities to PATH variable"
        path = ENV["PATH"]
        ENV["PATH"] = "$(pwd())/test_utilities:$path"
    end

    include("hpc/slurm.jl")
end
