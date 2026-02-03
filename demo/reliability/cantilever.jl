using UncertaintyQuantification

l = Parameter(1.8, :l) # length
b = Parameter(0.12, :b) # width

h = RandomVariable(Normal(0.24, 0.01), :h) # height

μ, σ = distribution_parameters(10e9, 1.6e9, LogNormal)
E = RandomVariable(LogNormal(μ, σ), :E) # young's modulus

μ, σ = distribution_parameters(5000, 400, LogNormal)
P = RandomVariable(LogNormal(μ, σ), :P) # tip load

μ, σ = distribution_parameters(600, 140, LogNormal)
ρ = RandomVariable(LogNormal(μ, σ), :ρ) # density

c = GaussianCopula([1 0.8; 0.8 1])
jd = JointDistribution(c, [E, ρ])

inputs = [l, b, h, P, jd]

inertia = Model(df -> df.b .* df.h .^ 3 / 12, :I)

displacement = Model(
    df ->
        (df.ρ .* 9.81 .* df.b .* df.h .* df.l .^ 4) ./ (8 .* df.E .* df.I) .+
        (df.P .* df.l .^ 3) ./ (3 .* df.E .* df.I),
    :w,
)

max_displacement = 0.01

# Compute probability of failure using standard Monte Carlo
mc = MonteCarlo(10^6)

mc_pf, mc_std, mc_samples = probability_of_failure(
    [inertia, displacement], df -> max_displacement .- df.w, inputs, mc
)

println(
    "Monte Carlo probability of failure $mc_pf ($(size(mc_samples, 1)) model evaluations)"
)

# Compute probability of failure using Importance Sampling
pf, β, dp, α = probability_of_failure(
    [inertia, displacement], df -> max_displacement .- df.w, inputs, FORM()
)
is = ImportanceSampling(10^4, β, dp, α)

is_pf, is_std, is_samples = probability_of_failure(
    [inertia, displacement], df -> max_displacement .- df.w, inputs, is
)

println(
    "Importance Sampling probability of failure: $is_pf ($(size(is_samples, 1)) model evaluations)",
)

rbis = RadialBasedImportanceSampling(10^4, β)

rbis_pf, rbis_cov, rbis_samples = probability_of_failure(
    [inertia, displacement], df -> max_displacement .- df.w, inputs, rbis
)

println(
    "Radial Based Importance Sampling probability of failure $rbis_pf ($(size(rbis_samples, 1)) model evaluations)",
)

# Compute probability of failure using Line Sampling
ls = LineSampling(200)

ls_pf, ls_std, ls_samples = probability_of_failure(
    [inertia, displacement], df -> max_displacement .- df.w, inputs, ls
)

println(
    "Line Sampling probability of failure: $ls_pf ($(size(ls_samples, 1)) model evaluations)",
)

# Compute probability of failure using Subset Sampling
subset = UncertaintyQuantification.SubSetSimulation(2000, 0.1, 10, Uniform(-0.5, 0.5))

subset_pf, subset_std, subset_samples = probability_of_failure(
    [inertia, displacement], df -> max_displacement .- df.w, inputs, subset
)

println(
    "Subset Simulation probability of failure: $subset_pf ($(size(subset_samples, 1)) model evaluations)",
)

# Compute probability of failure using conditional Subset Sampling
subset_inf = UncertaintyQuantification.SubSetInfinity(2000, 0.1, 10, 0.5)

subset_pf_inf, subset_std_inf, subset_samples_inf = probability_of_failure(
    [inertia, displacement], df -> max_displacement .- df.w, inputs, subset_inf
)

println(
    "Subset infinity probability of failure: $subset_pf_inf ($(size(subset_samples_inf, 1)) model evaluations)",
)

# Compute probability of failure using adaptive Subset Sampling
subset_adap = UncertaintyQuantification.SubSetInfinityAdaptive(2000, 0.1, 10, 10, 0.6, 1.0)

subset_pf_adap, subset_std_adap, subset_samples_adap = probability_of_failure(
    [inertia, displacement], df -> max_displacement .- df.w, inputs, subset_adap
)

println(
    "Subset infinity adaptive probability of failure: $subset_pf_adap ($(size(subset_samples_adap, 1)) model evaluations)",
)
