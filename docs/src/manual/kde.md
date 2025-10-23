# Kernel Density Estimation

Kernel density estimation (KDE) is a non-parametric method to estimate the probability density function of a random variable through *kernel smoothing* [silvermanDensityEstimationStatistics1986](@cite).

The kernel density estimate ``\hat{f}_h`` of a univariate density `f` based on a random sample ``X_1,\ldots,X_n`` is defined as

```math
\hat{f}_h(x) = n^{-1} \sum_{i=1}^n h^{-1} K \left\{\frac{x-X_i}{h}\right\},
```

where ``h`` is the so called *bandwidth* and ``K`` is the kernel function. The kernel function is assumed to be a symmetric probability density and is set to be a Gaussian density in *UncertaintyQuantification.jl*. The bandwidth ``h`` also called the *smoothing parameter* has a strong effect on the resulting density estimate. There are various different methods to select an optimal bandwidth. Here we have decided to apply the method developed by Sheather & Jones [sheatherReliableDataBasedBandwidth1991](@cite) for its excellent performance and straightforward implementation.

The kernel density estimation is exposed through the [`EmpiricalDistribution`](@ref). Since the bandwidth is automatically selected only a vector containing the data must be passed to the constructor.

```julia
 d = EmpiricalDistribution(x)
```

Internally, we perform the kernel density estimation to obtain the *PDF* of the distribution. From this PDF we estimate the support of the distribution through numerical root finding. The *CDF* and the quantile function (inverse CDF) are interpolated from the numerical integral of the PDF. The number of points used for this interpolation (defaults to ``10^4``) can be passed to the constructor as an optional second parameter. As a `ContinousUnivariateDistribution` the [`EmpiricalDistribution`](@ref) can be applied the same as any of the native distributions from *Distributions.jl*.

## Example

As an example we consider synthetic data generated from a bimodal distribution and fit the empirical distribution.

```@example kde
using UncertaintyQuantification # hide
x = [rand(Normal(5), 500)..., rand(Normal(10), 500)...]
ed = EmpiricalDistribution(x)
return nothing # hide
```

Next, we plot the normalized histogram of the data and the resulting PDF.

```@example kde
    using Plots # hide
    histogram(x, bins=200, label="Data", normalize=:pdf) # hide
    t = collect(range(ed.lb, ed.ub, 1000)) # hide
    den = pdf.(ed, t) # hide
    plot!(t, den, label="PDF", lw=2) # hide
    savefig("kernel-density.svg"); nothing # hide
```

![Kernel Density Plot](kernel-density.svg)

!!! note "Multivariate Density Estimation with Gaussian Mixture Models"
    Multivariate density estimation can be performed using Gaussian mixture models.
    For more information see [Gaussian Mixture Models](@ref).

## Large datasets

For large datasets the numerical demand required can be prohibitive. To still estimate a density we can use linear binning [jonesErrorsInvolvedComputing1983](@cite) by passing the `nbins` keyword. This will discretize the data on a linear grid. For each data point, weights are assigned proportionally to the distance to the two bordering grid points. Note, that grid points with a weight of zero are discarded.

```julia
d = EmpiricalDistribution(x; nbins=1000)
```
