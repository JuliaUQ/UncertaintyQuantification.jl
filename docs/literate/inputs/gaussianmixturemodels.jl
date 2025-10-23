#===

# Gaussian Mixture Model Example

In *UncertaintyQuantification.jl*, we can construct a GMM from available data using the EM
algorithm, as described in [Gaussian Mixture Models](@ref).

In this example, we will fit a GMM to synthetic data generated from two bivariate Gaussian distributions.
We first load the necessary packages to fit the GMM and visualize the results.
===#

using DataFrames
using Plots
using UncertaintyQuantification

# Then, we generate some data from two bivariate Gaussian distributions that we use to fit a GMM.
n1, n2 = 200, 500
N₁ = MvNormal([2.0, 2.0], [0.5 0.0; 0.0 0.5])
N₂ = MvNormal([5.0, 3.0], [1.0 0.8; 0.8 1.5])
X = permutedims([rand(N₁, n1) rand(N₂, n2)])
#md nothing # hide

# To store and process the data, we use a `DataFrame`:
df = DataFrame(X, [:x1, :x2])
#md nothing # hide

# Then, we fit a `GaussianMixtureModel` with two dimensions ($x_1$ and $x_2$) and $K=2$ components to the data stored in `df`:
gmm = GaussianMixtureModel(df, 2)

# To visually validate the fit, we can plot the data and the fitted GMM. We create a grid of points to evaluate the GMM's PDF and plot the contours.
x_range = range(-2, 10, length=100)
y_range = range(-2, 10, length=100)

scatter(df.x1, df.x2, alpha=0.3, label="Data")
contour!(x_range, y_range, (x,y) -> pdf(gmm, [x, y]), levels=10, linewidth=2, c=2, label="GMM")
#md savefig("data-gmm.svg"); nothing # hide

# ![Data and fitted GMM](data-gmm.svg)

# From the fitted GMM, we can also draw samples and compare them to the original data.
# We generate 500 samples from the GMM and plot them.
samples = sample(gmm, 500)
scatter!(samples.x1, samples.x2, alpha=0.3, c=2, label="Samples")
#md savefig("samples-gmm.svg"); nothing # hide

# ![Data and fitted GMM, and samples from the GMM](samples-gmm.svg)
