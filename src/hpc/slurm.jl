"""
	SlurmInterface(options::Dict{String,String}, throttle::Integer, btachsize::Integer, extras::Vector{String})

When `SlurmInterface` is passed to an `ExternalModel`, model evaluations are executed using slurm job arrays. This allows for heavier simulations or workflows to be sampled, without relying on Julia's native parallelism. `SlurmInterface` automatically generates a slurm job array script, and Julia waits for this job to finish before extracting results.

When using `SlurmInterface`, you no longer need to load workers into Julia with `addprocs(N)`, and the requested nodes / tasks those required by individual model evaluations. Use `extras` to specify anything that must be preloaded for your models to be executed (for example loading modules).

The `throttle` specifies the number of simulations in the job array which are run concurrently. I.e., if you perform
`MonteCarlo` simulation with `N=1000` samples, with `throttle=200`, it will run 1000 simulations in total, but only 200 at the same time. Your HPC scheduler (and admin) may be unhappy if you request too many concurrent jobs. If left empty, you scheduler's default throttle will be used. In the case that your HPC machine limits the size of submitted job arrays, you can split the submissions into smaller "batches". Specify "batchsize" to the maximum size of a job array. This does not change the total number of runs.

# parameters

    options   : A dictionary of SBATCH options to add to the slurm script
    throttle  : the number of jobs to be run at the same time
    batchsize : maximum size of the slurm array, use when HPC limits the number of jobs in arrays
    extras    : instructions to be executed before the model is run, e.g. activating a python environment or loading modules

# Examples
```jldoctest
julia> slurm = SlurmInterface(Dict("account" => "HPC_account_1", "partition" => "CPU_partition"), extras = ["load python3"])
SlurmInterface(Dict("account" => "HPC_account_1", "partition" => "CPU_partition"), 0, 0, ["load python3"])
```

"""
Base.@kwdef struct SlurmInterface <: AbstractHPCScheduler
    options::Dict{String, String}
    throttle::Integer = 0
    batchsize::Integer = 0
    extras::Vector{String} = String[]

    function SlurmInterface(
            options::Dict{String, String};
            throttle::Integer = 0,
            batchsize::Integer = 0,
            extras::Vector{String} = String[],
        )
        if haskey(options, "array")
            error("Do not pass array as an option.")
        end

        if !haskey(options, "account") || !haskey(options, "partition")
            @warn "Omitting account or partition may cause your simulation to fail if your scheduler does not assign defaults."
        end

        return new(options, throttle, batchsize, extras)
    end
end

function setup_hpc_jobs(si::SlurmInterface, m::ExternalModel, n::Integer, datetime::String)
    return if si.batchsize == 0
        setup_slurm_array(si, m::ExternalModel, n::Integer, datetime::String)
    else
        for batch in 1:ceil(Integer, n / si.batchsize)
            setup_slurm_array(si, m, n, datetime, batch)
        end
    end
end

function run_hpc_jobs(si::SlurmInterface, m::ExternalModel, n::Integer, datetime::String)
    return if si.batchsize == 0
        run_slurm_array(si, m::ExternalModel, datetime::String)
    else
        for batch in 1:ceil(Integer, n / si.batchsize)
            run_slurm_array(si, m, datetime, batch)
        end
    end
end

function setup_slurm_array(
        si::SlurmInterface, m::ExternalModel, n::Integer, path::String, batch::Integer = 0
    )
    binary = m.solver.path
    source = m.solver.source
    args = m.solver.args

    extras = si.extras

    run_command = !isempty(args) ? "$binary $args $source" : "$binary $source"

    a, b = if batch > 0
        (batch - 1) * si.batchsize + 1, min(batch * si.batchsize, n)
    else
        1, n
    end

    array_command = if iszero(si.throttle)
        "#SBATCH --array=[$a-$b]\n"
    else
        "#SBATCH --array=[$a-$b]%$(si.throttle)\n"
    end

    digits = ndigits(n)

    fname = if iszero(batch)
        "slurm_array.sh"
    else
        "slurm_array-$batch.sh"
    end

    dirpath = joinpath(m.workdir, path)
    fpath = joinpath(dirpath, fname)

    open(fpath, "w") do file
        write(file, "#!/bin/bash -l\n")

        for (k, v) in si.options
            write(file, "#SBATCH --$k=$v\n")
        end

        write(
            file, "#SBATCH --output=sample-%$(digits)a/UncertaintyQuantification-%a.out\n"
        )
        write(file, "#SBATCH --error=sample-%$(digits)a/UncertaintyQuantification-%a.err\n")
        write(file, array_command)
        write(file, "\n\n\n")
        write(file, "#### EXTRAS ####\n")

        for extra in extras
            write(file, "$extra\n")
        end

        write(file, "\n\n\n")

        write(file, "echo ========================================================\n")
        write(file, "echo SLURM job: submitted date = `date`\n")
        write(file, "date_start=`date +%s`\n")
        write(file, "echo =========================================================\n")
        write(file, "echo Job output begins\n")
        write(file, "echo -----------------\n")
        write(file, "echo\n")
        write(file, "hostname\n")
        write(file, "echo Running with \$SLURM_NTASKS cores\n")

        write(file, "\n\n\n")

        write(file, "#### RUN COMMAND ####\n")
        write(file, "cd sample-\$(printf %0$(digits)d \$SLURM_ARRAY_TASK_ID)\n")
        write(file, "$run_command\n")

        write(file, "\n\n\n")

        write(file, "echo\n")
        write(file, "echo ---------------\n")
        write(file, "echo Job output ends\n")
        write(file, "date_end=`date +%s`\n")
        write(file, "seconds=\$((date_end-date_start))\n")
        write(file, "minutes=\$((seconds/60))\n")
        write(file, "seconds=\$((seconds-60*minutes))\n")
        write(file, "hours=\$((minutes/60))\n")
        write(file, "minutes=\$((minutes-60*hours))\n")
        write(file, "echo =========================================================\n")
        write(file, "echo SLURM job: finished date = `date`\n")
        write(
            file,
            "echo Total run time : \$hours Hours \$minutes Minutes \$seconds Seconds\n",
        )
        write(file, "echo =========================================================\n")
    end

    return nothing
end

# Slurm interface is passed to dispatch function for slurm. Perhaps there is a more elegant solution using parametric typing.
function run_slurm_array(
        si::SlurmInterface, m::ExternalModel, path::String, batch::Integer = 0
    )
    dirpath = joinpath(m.workdir, path)

    p = if iszero(batch)
        pipeline(`sbatch --wait slurm_array.sh`)
    else
        pipeline(`sbatch --wait slurm_array-$batch.sh`)
    end

    cd(() -> run(p), dirpath)

    return nothing
end
