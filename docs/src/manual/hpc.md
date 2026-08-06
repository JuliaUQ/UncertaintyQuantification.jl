# High performance computing

## Slurm job arrays

When sampling large simulation models, or complicated workflows, Julia's inbuilt parallelism is sometimes insufficient. Job arrays are a useful feature of the slurm scheduler which allow you to run many similar jobs, which differ by an index (for example a sample number). This allows `UncertaintyQuantification.jl` to run heavier simulations (for example, simulations requiring multiple nodes), by offloading model sampling to an HPC machine using slurm. This way, `UncertaintyQuantification.jl` can be started on a single worker, and the HPC machine handles the rest.

For more information on job arrays, see: [job arrays](https://slurm.schedmd.com/job_array.html).

## Slurm Interface

When a `SlurmInterface` is passed to an `ExternalModel`, a slurm job array script is automatically generated and executed. Julia waits for this job to finish before extracting results and proceeding.

```@example hpc
using UncertaintyQuantification #hide
options = Dict(
    "account"=>"HPC_account_1",
    "partition"=>"CPU_partition",
    "job-name"=>"UQ_array",
    "nodes"=>"1",
    "ntasks" =>"32",
    "time"=>"01:00:00"
)

slurm = SlurmInterface(
    options;
    throttle=50,
    batchsize=200,
    extras=["load python3"]
)
return nothing #hide
```

Here `account` is your account (provided by your HPC admin/PI), and `partition` specifies the queue that jobs will be submitted to (ask admin if unsure). `nodes` and `ntasks` are the number of nodes and CPUs that your individual simulations requires. Depending on your HPC machine, each node has a specific number of CPUs. If your application requires more CPUs than are available per node, you can use multiple nodes. Through `options` the `SlurmInterface` supports all options of `SBATCH` except for `array` since the job array is constructed dynamically.

The parameter `time` specifies the maximum time that each simulation will be run for, before being killed.

!!! warning "Individual model runs VS overall batch"
    `nodes`, `ntasks`, and `time` are parameters required for each _individual_ model evaluation, not the entire batch. For example, if you are running a large FEM simulation that requires 100 CPUs to evaluate one sample, and your HPC machine has 50 CPUs per node, you would specify `nodes = 2` and `ntasks = 100`.

!!! note "Compiling with MPI"
    If your model requires multiple `nodes`, it may be best to compile your application with MPI, if your model allows for it. Please check your application's documentation for compiling with MPI.

Any commands in `extras` will be executed before you model is run, for example loading any module files or data your model requires. Multiple commands can be passed: `extras = ["load python", "python3 get_data.py"]`.

!!! note
    If your `extras` command requires `""` or `$` symbols, they must be properly escaped as `\"\"` and `\$`.

The job array task throttle, which is the number of samples that will be run concurrently at any given time, is specified by `throttle`. For example, when running a `MonteCarlo` simulation with 2000 samples, and `throttle = 50`, 2000 model evaluations will be run in total, but only 50 at the same time. If left empty, your scheduler's default throttle will be used. Sometimes the scheduler limits the maximum size of a single job array. In these cases, the maximum size can be set through the `batchsize` parameter. This will separate the jobs into multiple smaller arrays.

## Testing your HPC configuration

Slurm is tested _only_ on linux systems, not Mac or Windows. When testing `UncertaintyQuantification.jl` locally, we use a dummy function `test/test_utilities/sbatch` to mimic an HPC scheduler.

!!! warning "Testing locally on Linux"
    Certain Slurm tests may fail unless `test/test_utilities/` is added to PATH. To do this: `export PATH=UncertaintyQuantification.jl/test/test_utilities/:$PATH`. Additionally, _actual_ slurm submissions may fail if `test/test_utilities/sbatch` is called in place of your system installation. To find out which sbatch you're using, call `which sbatch`.

If you'd like to **actually** test the Slurm interface your HPC machine:

```julia
using Pkg
Pkg.test("UncertaintyQuantification"; test_args=["HPC", "YOUR_ACCOUNT", "YOUR_PARTITION"])
```

or if you have a local clone, from the top directory:

```julia
julia --project
using Pkg
Pkg.test(; test_args=["HPC", "YOUR_ACCOUNT", "YOUR_PARTITION"])
```

`YOUR_ACCOUNT` and `YOUR_PARTITION` should be replaced with your account and partition you wish to use for testing. This test will submit 4 slurm job arrays, of a lightweight calculation (> 1 minute per job) requiring 1 core/task each.

### Usage

See [High Performance Computing](@ref) for a detailed example.
