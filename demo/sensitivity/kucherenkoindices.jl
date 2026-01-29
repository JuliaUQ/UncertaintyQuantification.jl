using UncertaintyQuantification
using LinearAlgebra

# Kucherenko et al. (2012) (DOI: 10.1016/j.cpc.2011.12.020)  --- Test Case 2: Portfolio model with analytical indices ---

μ = [0.0, 0.0, 250.0, 400.0]

σ1, σ2, σ3, σ4 = sqrt(16.0), sqrt(4.0), sqrt(4e4), sqrt(9e4)
σ12, σ21 = 2.4, 2.4
σ34, σ43 = -1.8e4, -1.8e4

Σ = [
    16.0    2.4     0.0     0.0;
    2.4     4.0     0.0     0.0;
    0.0     0.0     4e4     -1.8e4;
    0.0     0.0     -1.8e4  9e4
]

# Marginals
marginals = RandomVariable[
    RandomVariable(Normal(μ[1], σ1), :x1),
    RandomVariable(Normal(μ[2], σ2), :x2),
    RandomVariable(Normal(μ[3], σ3), :x3),
    RandomVariable(Normal(μ[4], σ4), :x4)
]

# Correlation matrix for Gaussian copula
Dvec = sqrt.(diag(Σ))
R = Σ ./ (Dvec * Dvec')

inputs = [
    JointDistribution(GaussianCopula(R), marginals)
]

model = Model(df -> df.x1 .* df.x3 .+ df.x2 .* df.x4, :y)
sim = MonteCarlo(200000)


# Analytical values
μ3, μ4 = μ[3], μ[4]
σ2_1, σ2_2, σ2_3, σ2_4 = 16.0, 4.0, 4e4, 9e4
σ12, σ34 = 2.4, -1.8e4
ρ12 = σ12 / (σ1 * σ2)
ρ34 = σ34 / (σ3 * σ4)

D = σ2_1 * (σ2_3 + μ3^2) + σ2_2 * (σ2_4 + μ4^2) + 2 * σ12 * (σ34 + μ3 * μ4)

S1_analytical = σ2_1 * (μ3 + μ4 * ρ12 * σ2 / σ1)^2 / D
ST1_analytical = σ2_1 * (1 - ρ12^2) * (σ2_3 + μ3^2) / D

S2_analytical = σ2_2 * (μ4 + μ3 * ρ12 * σ1 / σ2)^2 / D
ST2_analytical = σ2_2 * (1 - ρ12^2) * (σ2_4 + μ4^2) / D

S3_analytical = 0.0
ST3_analytical = σ2_1 * σ2_3 * (1 - ρ34^2) / D

S4_analytical = 0.0
ST4_analytical = σ2_2 * σ2_4 * (1 - ρ34^2) / D



try
    indices, bin_samples = kucherenkoindices_bin([model], inputs, [:y], sim; min_bin_sample_multi_dims=5)
    println("Sample-based Kucherenko Indices calculation using bins:")
    println(indices)

    tol = 0.01
    @assert abs(indices.FirstOrder[1] - S1_analytical) < tol "S1 error too large"
    @assert abs(indices.FirstOrder[2] - S2_analytical) < tol "S2 error too large"
    @assert abs(indices.FirstOrder[3] - S3_analytical) < tol "S3 error too large"
    @assert abs(indices.FirstOrder[4] - S4_analytical) < tol "S4 error too large"
    @assert abs(indices.TotalEffect[1] - ST1_analytical) < tol "ST1 error too large"
    @assert abs(indices.TotalEffect[2] - ST2_analytical) < tol "ST2 error too large"
    @assert abs(indices.TotalEffect[3] - ST3_analytical) < tol "ST3 error too large"
    @assert abs(indices.TotalEffect[4] - ST4_analytical) < tol "ST4 error too large"

    println("Portfolio model Kucherenko indices validation passed - all values within tolerance.")

catch e
    println("Error computing Portfolio model Kucherenko indices: $e")
    rethrow(e)
end

try 
    indices = indices = kucherenkoindices([model], inputs, [:y], sim)
    println("Standard Kucherenko Indices calculation:")
    println(indices)

    tol = 0.01
    @assert abs(indices.FirstOrder[1] - S1_analytical) < tol "S1 error too large"
    @assert abs(indices.FirstOrder[2] - S2_analytical) < tol "S2 error too large"
    @assert abs(indices.FirstOrder[3] - S3_analytical) < tol "S3 error too large"
    @assert abs(indices.FirstOrder[4] - S4_analytical) < tol "S4 error too large"
    @assert abs(indices.TotalEffect[1] - ST1_analytical) < tol "ST1 error too large"
    @assert abs(indices.TotalEffect[2] - ST2_analytical) < tol "ST2 error too large"
    @assert abs(indices.TotalEffect[3] - ST3_analytical) < tol "ST3 error too large"
    @assert abs(indices.TotalEffect[4] - ST4_analytical) < tol "ST4 error too large"

    println("Portfolio model Kucherenko indices validation passed - all values within tolerance.")
catch e
    println("Error computing Portfolio model Kucherenko indices: $e")
    rethrow(e)
end
