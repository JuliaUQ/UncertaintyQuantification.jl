using Copulas
using DataFrames
using Distributed
using HCubature
using InteractiveUtils
using LinearAlgebra: norm
using ParameterHandling
using Random
using StatsBase: fit, Histogram, corkendall
using Test
using Plots
using TestItemRunner
using UncertaintyQuantification
import UncertaintyQuantification: sample

@testsnippet TestSetup begin
    using ADTypes
    using Copulas
    using DataFrames
    using Distributed
    using HCubature
    using InteractiveUtils
    using Random
    using StatsBase: fit, Histogram, corkendall
end

@testsnippet QMC begin
    using QuasiMonteCarlo
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
include("models/gp/parameterization.jl")
include("inputs/jointdistribution.jl")
include("inputs/imprecise/p-box.jl")
include("plotting/plotting.jl")

@run_package_tests

if Sys.islinux()
    if !occursin("test/test_utilities", ENV["PATH"])
        @warn "Adding test utilities to PATH variable"
        path = ENV["PATH"]
        ENV["PATH"] = "$(pwd())/test_utilities:$path"
    end

    include("hpc/slurm.jl")
end
