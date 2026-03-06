# [Transport Maps](@id transport_map_manual)

**Transport maps** (TMs) construct a deterministic transformation between a simple reference distribution (typically standard normal) and a complex target distribution [marzoukSamplingMeasureTransport2016](@cite), [ramgraberTriangularTransport2025](@cite).
Once such a map is constructed, generating samples from the target becomes trivial: simply draw samples from the reference and apply the transformation. Moreover, transport maps enable efficient computation of conditional distributions, marginals, and other probabilistic quantities through the geometric structure they impose 

## Mathematical Formulation

Mathematically, a transport map $T: \boldsymbol{Z} \mapsto \boldsymbol{\Theta}$ is defined as a deterministic coupling between a reference space $\boldsymbol{Z} \sim \rho(\boldsymbol{z})$ and a target space $\boldsymbol{\Theta} \sim \pi(\boldsymbol{\theta})$ .
Hence, the inverse map $T^{-1}: \boldsymbol{\Theta} \mapsto \boldsymbol{Z}$ maps from the target space back to the reference space.

The target distribution is approximated by the so-called *pull-back* density:
```math
\pi(\theta) \approx T_{\#}\rho = \rho\left(T^{-1}(\bm{a},\theta)\right) \ |\det \nabla T^{-1}(\bm{a},\theta)|.
```

### Triangular Structure

The construction and inversion of the transport map can be greatly simplified by using a triangular structure following the Knothe-Rosenblatt rearrangement [knotheContributions1957](@cite), [rosenblattRemarks1952](@cite).
This triangular structure guarantees invertibility and makes the Jacobian determinant straightforward to compute.
A triangular map in $n$ dimensions has the form:

```math
T(\boldsymbol{z}) =
\left(\begin{array}{c}
T_1(z_1) \\
T_2(z_1, z_2) \\
T_3(z_1, z_2, z_3) \\
\vdots \\
T_n(z_1, z_2, \dots, z_n)
\end{array}
\right)
```

A key requirement for the transport map is that each component $T_k$ must be strictly monotonically increasing in its last argument $z_k$ (while possibly depending non-monotonically on the earlier arguments $z_{1:k-1}$).
This monotonicity constraint ensures invertibility and is enforced through a specialized parameterization.

### Integrated Rectifier Parameterization

This monotonicity requirement is commonly achieved using an integrated rectifier parameterization, where each component takes the form:

```math
T_k(z_1, \ldots, z_k; \boldsymbol{a}) = f(z_1, \ldots, z_{k-1}, 0; \boldsymbol{a}) + \int_0^{z_k} g\left(\partial_k f(z_1, \ldots, z_{k-1}, \xi; \boldsymbol{a})\right) d\xi.
```

Here, $f(z_1, \ldots, z_k; \boldsymbol{a})$ is a multivariate polynomial

```math
f(z_1, \ldots, z_k; \boldsymbol{a}) = \sum_{\alpha \in \mathcal{A}_k} a_\alpha \Psi_\alpha(z_1, \ldots, z_k)
```
where:
- $\mathcal{A}_k$ is a multi-index set defining which basis functions are included
- $a_\alpha$ are the optimization coefficients
- $\Psi_\alpha$ are multivariate basis functions
- $g: \mathbb{R} \to \mathbb{R}^+$ is a rectifier function that maps the derivative of $f$ to a strictly positive value, ensuring monotonicity

## Implementation

In *UncertaintyQuantification.jl*, transport maps are implemented using the [TransportMaps.jl](https://juliauq.github.io/TransportMaps.jl/stable/) package.
This package provides the backend implementation, including the construction of basis functions, rectifier functions, and the optimization procedures to determine the map coefficients.

There are two main approaches to constructing transport maps, depending on the available information about the target distribution:
1. **From target density**: When an analytical expression for the log-density is available
2. **From target samples**: When only samples from the target distribution are available

### Map Construction from Target Density

When an analytical expression for the target log-density is available, we determine the map coefficients $\boldsymbol{a}$ by solving an optimization problem that minimizes the Kullback-Leibler (KL) divergence between the target density and the transport map approximation:
```math
\min_{\bm{a}} \mathcal{D}_{\mathsf{KL}}(T_{\#}\rho||\pi)
```
The KL divergence is computed as an expected value with respect to the reference measure and approximated using numerical quadrature:
```math
\sum_{i=1}^{N} w_{q,i}\Big[-\log\pi\bigl(T(\boldsymbol{a},\boldsymbol{z}_{q,i})\bigr)-\log |\det\nabla T(\boldsymbol{a},\boldsymbol{z}_{q,i}) |\Big].
```
Here, $w_{q,i}$ are quadrature weights and $\boldsymbol{z}_{q,i}$ are quadrature points.
We use the quadrature schemes defined in [TransportMaps.jl: Quadrature methods](https://juliauq.github.io/TransportMaps.jl/stable/Manuals/quadrature_methods).
Currently available schemes are:
- Monte Carlo
- Latin Hypercube
- Gauss-Hermite, and 
- Sparse Smolyak

#### Usage

The [`TransportMap`](@ref) is constructed using the [`mapfromdensity`](@ref) function, which requires the following inputs:

- `map`: The map structure to be optimized (e.g., a [`TransportMaps.PolynomialMap`](https://juliauq.github.io/TransportMaps.jl/stable/api/maps#TransportMaps.PolynomialMap))
- `target`: The target density as a [`TransportMaps.MapTargetDensity`](https://juliauq.github.io/TransportMaps.jl/stable/api/densities#TransportMaps.MapTargetDensity) object
- `quadrature`: The quadrature scheme used to evaluate the KL divergence (see [list of quadrature schemes](https://juliauq.github.io/TransportMaps.jl/stable/api/quadrature))
- `names`: A `Vector{Symbol}` containing the names of the random variables


The following example demonstrates how to construct a transport map from a given log-density function:
```@example transportmap_banana
using UncertaintyQuantification # hide
using Plots # hide
using DataFrames #hide


# Define the log-density of the target (banana-shaped distribution)
logtarget(x) = logpdf.(Normal(), x[1]) + logpdf.(Normal(), x[2] .- x[1].^2)

# Create a 2D polynomial map with degree 2
# Defaults: reference=Normal(), rectifier=Softplus(), basis=LinearizedHermiteBasis()
pm = PolynomialMap(2, 2)

# Define 2D quadrature using a Gauss-Hermite tensor product with 3 points per dimension
quad = GaussHermiteWeights(3, 2) 

# Construct the target density and fit the map
target = MapTargetDensity(logtarget) # Requires a log-density function
tm_opt = mapfromdensity(pm, target, quad, [:x1, :x2])
```

Once the map is constructed, we can generate samples from the target distribution and evaluate its probability density function:

```@example transportmap_banana
# Generate 1000 samples from the target distribution
samples = sample(tm_opt, 1000)

# Evaluate the pdf on a grid for visualization
x1_range = -4:0.1:4
x2_range = -3:0.1:7
pdf_vals = [pdf(tm_opt, [x1, x2]) for x2 in x2_range, x1 in x1_range]

scatter(samples.x1, samples.x2; alpha=0.8, label="TM Samples")
contour!(x1_range, x2_range, pdf_vals)
savefig("tm-banana-1.svg"); nothing # hide
```

![Banana density](tm-banana-1.svg)

!!! note "Bayesian Updating with Transport Maps"
    The abaility to have an analytical expression for the density and the ability to generate samples make transport maps appealing for Bayesian inference applications. For the usage with [`bayesianupdating`](@ref) see [Variational Inference with Transport Maps](@ref).

### Map Construction from Target Samples

When only samples from the target distribution are available (without an analytical density), the KL-divergence is formulated in reverse, i.e., as an expected value with respect to the target measure rather than the reference measure.
In this case, the construction of the transport map allows for density estimation from samples and can provide an alternative to [Gaussian Mixture Models](@ref).

#### Usage

This approach is implemented as a [`TransportMapFromSamples`](@ref) constructed using the [`mapfromsamples`](@ref) function, which requires the following inputs:

- `transportmap`: The map structure to be optimized (e.g., a [`TransportMaps.PolynomialMap`](https://juliauq.github.io/TransportMaps.jl/stable/api/maps#TransportMaps.PolynomialMap))
- `samples`: A `DataFrame` containing the samples from the target distribution

We consider the same banana-shaped distribution as before. However, we now start with samples generated using a simple acceptance-rejection method:
```@example transportmap_banana
function generate_samples(n_samples::Int)
    x1_samples = Vector{Float64}(undef, n_samples)
    x2_samples = Vector{Float64}(undef, n_samples)

    count = 0
    while count < n_samples
        x1 = randn() * 2
        x2 = randn() * 3 + x1^2

        if rand() < exp(logtarget([x1, x2])) / 0.4
            count += 1
            x1_samples[count] = x1
            x2_samples[count] = x2
        end
    end

    return DataFrame(x1 = x1_samples, x2 = x2_samples)
end

target_samples = generate_samples(1000) # Returns a DataFrame with samples
nothing # hide
```

We define the transport map structure as a [`PolynomialMap`](https://juliauq.github.io/TransportMaps.jl/stable/api/maps#TransportMaps.PolynomialMap) and fit it using the available samples:

```@example transportmap_banana
pm_samples = PolynomialMap(2, 2)
tm = mapfromsamples(pm_samples, target_samples)
```
As in the density-based approach, the fitted map enables sampling and probability density evaluation through the mapping from the reference space.
```@example transportmap_banana
# Generate 1000 new samples from the fitted transport map
tm_samples = sample(tm, 1000)

# Evaluate the pdf on a grid
pdf_vals_tm = [pdf(tm, [x1, x2]) for x2 in x2_range, x1 in x1_range]

# Visualize both the original and generated samples with the fitted density
scatter(target_samples.x1, target_samples.x2; alpha=0.8, label="Original Samples")
scatter!(tm_samples.x1, tm_samples.x2; alpha=0.8, label="TM Samples")
contour!(x1_range, x2_range, pdf_vals_tm)
savefig("tm-banana-2.svg"); nothing # hide
```

![Banana density](tm-banana-2.svg)
