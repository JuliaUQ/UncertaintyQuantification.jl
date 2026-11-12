using UncertaintyQuantification

# Example from:
# Dang, C., Valdebenito, M. A., Song, J., Wei, P., & Beer, M. (2023).
# Estimation of small failure probabilities by partially Bayesian active learning
# line sampling: Theory and algorithm. Computer Methods in Applied Mechanics
# and Engineering, 412, 116068. https://doi.org/10.1016/j.cma.2023.116068

#   Input:
#       𝑿 = [𝑋₁, 𝑋₂] with 𝑋ᵢ ∼ 𝒩(0,1)
#   Limit-state function:
#       𝑔(𝑿) = 5 - 𝑋₂ + 0.01 𝑋₁³ + sin(𝑋₁)

# Input
x = RandomVariable.(Normal(), [:x1, :x2])

# Model
y = Model(df -> df.x2 .+ 0.01 * df.x1 .^ 3 .+ sin.(df.x1), [:x1, :x2], :y)

# Limit-state function
g(df) = 3 .- df.y

# Simulation
numberlines = 200

# Perform Line Sampling
LS = LineSampling(numberlines, collect(0.5:0.5:8))
pf_ls, σ_pf_ls, samples_ls = probability_of_failure([y], g, x, LS)
cov_ls = σ_pf_ls / pf_ls

# Perform Advanced Line Sampling
ALS = AdvancedLineSampling(numberlines, collect(0.5:0.5:8))
pf_als, σ_pf_als, samples_als = probability_of_failure([y], g, x, ALS)
cov_als = σ_pf_als / pf_als

# Perform Monte Calo for Reference
pf_MC, σ_pf_MC, samples_MC = probability_of_failure([y], g, x, MonteCarlo(10_000_000))
cov_MC = σ_pf_MC / pf_MC

println("Line Sampling:")
println("Pf = $pf_ls")
println("CoV = $cov_ls")
println("Model Calls = $(length(samples_ls.x1))\n")

println("Advanced Line Sampling:")
println("Pf = $pf_als")
println("CoV = $cov_als")
println("Model Calls = $(length(samples_als.x1))\n")

println("Monte Carlo:")
println("Pf = $pf_MC")
println("CoV = $cov_MC")
println("Model Calls = $(length(samples_MC.x1))")

# Imprecise Analysis
println("Performing Imprecise Reliability Analysis")

μ = Interval(-0.2, 0.2)
σ = Interval(0.8, 1.2)

x = RandomVariable.(ProbabilityBox{Normal}(Dict(:μ => μ, :σ => σ)), [:x1, :x2])

# Perform Line Sampling
@time pf_ls_DL, x_lb_ls, x_ub_ls = probability_of_failure(
    [y], g, x, DoubleLoop(LineSampling(numberlines, collect(0.5:0.5:8)))
)
@time pf_ls_RS, σ_pf_ls_RS, samples_ls_RS = probability_of_failure(
    [y], g, x, RandomSlicing(LineSampling(numberlines, collect(0.5:0.5:8)))
)

# Perform Advanced Line Sampling
@time pf_als_DL, x_lb_als, x_ub_als = probability_of_failure(
    [y], g, x, DoubleLoop(AdvancedLineSampling(numberlines, collect(0.5:0.5:8)))
)
@time pf_als_RS, σ_pf_als_RS, samples_als_RS = probability_of_failure(
    [y], g, x, RandomSlicing(AdvancedLineSampling(numberlines, collect(0.5:0.5:8)))
)

println("Double Loop:")
println("Line Sampling pf = [$(pf_ls_DL.lb), $(pf_ls_DL.ub)]")
println("Advanced Line Sampling pf = [$(pf_als_DL.lb), $(pf_als_DL.ub)] \n")

println("Random Slicing:")
println("Line Sample pf = $pf_ls_RS")
println("Advanced Line Sample pf = $pf_als_RS")
