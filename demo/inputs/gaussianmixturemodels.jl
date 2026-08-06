using DataFrames
using Plots
using UncertaintyQuantification

n1, n2 = 200, 500
N₁ = MvNormal([2.0, 2.0], [0.5 0.0; 0.0 0.5])
N₂ = MvNormal([5.0, 3.0], [1.0 0.8; 0.8 1.5])
X = permutedims([rand(N₁, n1) rand(N₂, n2)])

df = DataFrame(X, [:x1, :x2])

gmm = GaussianMixtureModel(df, 2)

x_range = range(-2, 10, length=100)
y_range = range(-2, 10, length=100)

scatter(df.x1, df.x2, alpha=0.3, label="Data")
contour!(x_range, y_range, (x,y) -> pdf(gmm, [x, y]), levels=10, linewidth=2, c=2, label="GMM")

samples = sample(gmm, 500)
scatter!(samples.x1, samples.x2, alpha=0.3, c=2, label="Samples")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
