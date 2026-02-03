using UncertaintyQuantification, Plots

x = RandomVariable.(Uniform(-1, 1), [:x1, :x2])

model = Model(df -> begin
    π .* (df.x1 .- 1) .* sin.(π .* df.x1) .* (1 .- df.x2 .^ 2)
end, :y)

p = 8
Ψ = PolynomialChaosBasis([LegendreBasis(), LegendreBasis()], p)

# Estimation by least squares
ls_n = 1000
ls = LeastSquares(SobolSampling(ls_n))
pceLS, samples, mse = polynomialchaos(x, model, Ψ, :y, ls)

println("LS Mean: $(mean(pceLS))")
println("LS Variance: $(var(pceLS))")
println("LS Mean Squared Error: $mse")
println("--------------------------")

# Estimation by WAFP
wafp = WeightedApproximateFetekePoints(SobolSampling(ls_n))
pceWAFP, samples, mse = polynomialchaos(x, model, Ψ, :y, wafp)

println("WAFP Mean: $(mean(pceWAFP))")
println("WAFP Variance: $(var(pceWAFP))")
println("WAFP Mean Squared Error: $mse")
println("--------------------------")

# Estimation by full quadrature
gq = GaussQuadrature()
pceGQ, samples = polynomialchaos(x, model, Ψ, :y, gq)

println("GQ Mean: $(mean(pceGQ))")
println("GQ Variance: $(var(pceGQ))")
println("--------------------------")

# Estimation by MC
samplesMC = sample(x, 10^5)
evaluate!(model, samplesMC)

println("MC Mean: $(mean(samplesMC.y))")
println("MC Variance: $(var(samplesMC.y))")

# Plots 4 histograms
samplesLS = sample(pceLS, 10^5)
samplesWAFP = sample(pceWAFP, 10^5)
samplesGQ = sample(pceGQ, 10^5)

histogram(samplesLS.y; alpha=0.5, label="Least squares", normalize=:probability, bins=100)
histogram!(samplesWAFP.y; alpha=0.5, label="WAFP", normalize=:probability, bins=100)
histogram!(samplesGQ.y; alpha=0.5, label="Quadrature", normalize=:probability, bins=100)
histogram!(samplesMC.y; alpha=0.5, label="Monte Carlo", normalize=:probability, bins=100)
