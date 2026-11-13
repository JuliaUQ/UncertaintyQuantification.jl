using UncertaintyQuantification
using Random
Random.seed!(42)

# Example modified:
# Papaioannou, I., & Straub, D. (2021). Combination line sampling for structural reliability
# analysis. Structural Safety, 88, 102025. https://doi.org/10.1016/j.strusafe.2020.102025

#   Input:
#       𝑿 = [𝑋₁, 𝑋₂] with 𝑋ᵢ ∼ 𝒩(0,1)
#   Performance function:
#       𝑔(𝑿) = 0.1(𝑋₁ - 𝑋₂)² - 1/√2 (𝑋₁ + 𝑋₂) + 4

# Input
x = RandomVariable.(Normal(), [:x1, :x2])

# Model
y = Model(df -> 0.1 * (df.x1 - df.x2) .^ 2 - 1 / sqrt(2) * (df.x1 + df.x2), :y)

# Performance function
g(df) = df.y .+ 4

# Perform FORM
pf_form, _ = probability_of_failure([y], g, x, FORM())

# Perform Monte Calo
MC = MonteCarlo(10_000_000)
pf_mc, σ_pf_mc, samples_mc = probability_of_failure([y], g, x, MC)
cov_mc = σ_pf_mc / pf_mc

# Perform Importance Sampling
IS = ImportanceSampling(1_000)
pf_is, σ_pf_is, samples_is = probability_of_failure([y], g, x, IS)
cov_is = σ_pf_is / pf_is

# Perform Line Sampling
LS = LineSampling(100, collect(0.5:0.5:10))
pf_ls, σ_pf_ls, samples_ls = probability_of_failure([y], g, x, LS)
cov_ls = σ_pf_ls / pf_ls

# Perform Advanced Line Sampling
ALS = AdvancedLineSampling(100, collect(0.5:0.5:10))
pf_als, σ_pf_als, samples_als = probability_of_failure([y], g, x, ALS)
cov_als = σ_pf_als / pf_als

# Subset
SuS = SubSetSimulation(1000, 0.1, 10, Normal())
pf_sus, σ_sus, samples_sus = probability_of_failure([y], g, x, SuS)
cov_sus = σ_sus / pf_sus

# Subset Infinity
SuSInf = SubSetInfinity(1000, 0.1, 10, 0.5)
pf_sus_inf, σ_sus_inf, samples_sus_inf = probability_of_failure([y], g, x, SuSInf)
cov_sus_inf = σ_sus_inf / pf_sus_inf

println("FORM:")
println("Pf = $pf_form\n")

println("Monte Carlo:")
println("Pf = $pf_mc")
println("CoV = $cov_mc")
println("Model Calls = $(length(samples_mc.x1))\n")

println("Importance Sampling:")
println("Pf = $pf_is")
println("CoV = $cov_is")
println("Model Calls = $(length(samples_is.x1))\n")

println("Line Sampling:")
println("Pf = $pf_ls")
println("CoV = $cov_ls")
println("Model Calls = $(length(samples_ls.x1))\n")

println("Advanced Line Sampling:")
println("Pf = $pf_als")
println("CoV = $cov_als")
println("Model Calls = $(length(samples_als.x1))\n")

println("Subset Simulation:")
println("Pf = $pf_sus")
println("CoV = $cov_sus")
println("Model Calls = $(length(samples_sus.x1))\n")

println("Subset Infinity:")
println("Pf = $pf_sus_inf")
println("CoV = $cov_sus_inf")
println("Model Calls = $(length(samples_sus_inf.x1))\n")
