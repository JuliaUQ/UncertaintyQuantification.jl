"""
Parameterized objects: a uniform interface for trainable models.

`Parameterized(obj)` wraps an object so it can be called with a parameter vector `θ`:
    ```julia
    model, θ = parameterize(obj)
    model(θ)  # returns a new object with parameters applied
    This works for mean functions, kernels, transformations, and Gaussian processes.

The system relies on two core functions:

    1.  extract_parameters(obj)

    Returns the free parameters of obj wrapped in ParameterHandling containers.
    Enforces constraints (e.g., positive or bounded) where applicable.
    For composite objects (like KernelSum or GP), returns a tuple or vector of parameter sets.
    Returns nothing for objects without trainable parameters.

    2.  apply_parameters(obj, θ)

    Returns a new object of the same type with parameters θ applied.
    For hierarchical objects, θ is expected to match the structure returned by extract_parameters.

This interface enables generic optimization routines to work across all supported types.
"""

struct Parameterized{T}
    object::T
end

function (p::Parameterized)(θ)
    return apply_parameters(p.object, ParameterHandling.value(θ))
end

parameterize(object) = Parameterized(object), extract_parameters(object)

# ---------------- Mean functions ----------------

extract_parameters(::ZeroMean) = nothing
apply_parameters(m::ZeroMean, θ) = m

extract_parameters(m::ConstMean) = m.c
apply_parameters(::ConstMean, θ) = ConstMean(θ)

# TODO: CustomMean uses a user defined function that could depend on parameters 
#       We should support custom definitions for applying and extracting parameters
extract_parameters(::CustomMean) = nothing
apply_parameters(m::CustomMean, θ) = m

# ---------------- Kernel functions ----------------

# Kernels and transforms without parameters
BaseKernelsWithoutParameters = Union{
    ZeroKernel, WhiteKernel, CosineKernel,
    SqExponentialKernel, ExponentialKernel,
    ExponentiatedKernel, Matern32Kernel,
    Matern52Kernel, NeuralNetworkKernel,
    PiecewisePolynomialKernel, WienerKernel
}

# TODO: GibbsKernel has a lengthscale function which could depend on trainable parameters
KernelsWithoutParameters = GibbsKernel 

# TODO: FunctionTransform has a transformation function which could depend on trainable parameters
TransformsWithoutParameters = Union{FunctionTransform, SelectTransform, KernelFunctions.IdentityTransform}

AllWithoutParameters = Union{
    BaseKernelsWithoutParameters, 
    KernelsWithoutParameters, 
    TransformsWithoutParameters
}

# TODO: Add support for multi-output models (MOKernel) and general neural networks as kernels (NeuralKernelNetwork)
UnsupportedKernels = Union{
    IndependentMOKernel, IntrinsicCoregionMOKernel, 
    LatentFactorMOKernel, LinearMixingModelKernel,
    KernelFunctions.NeuralKernelNetwork
}

# no parameters
extract_parameters(::T) where {T<:AllWithoutParameters} = nothing
apply_parameters(k::T, θ) where {T<:AllWithoutParameters} = k

# currently unsupported
extract_parameters(::T) where {T<:UnsupportedKernels} = throw(ArgumentError("`extract_parameters` is not supported for kernel type $(T)."))
apply_parameters(k::T, θ) where {T<:UnsupportedKernels} = throw(ArgumentError("`apply_parameters` is not supported for kernel type $(T)."))

# basekernels (see KernelFunctions.jl src/basekernels)
extract_parameters(k::ConstantKernel) = ParameterHandling.positive(k.c)
apply_parameters(::ConstantKernel, θ) = ConstantKernel(; c=only(θ))

extract_parameters(k::GammaExponentialKernel) = ParameterHandling.bounded(k.γ, 0.0, 2.0)
apply_parameters(::GammaExponentialKernel, θ) = GammaExponentialKernel(; γ=only(θ))

extract_parameters(k::FBMKernel) = ParameterHandling.bounded(k.h, 0.0, 1.0)
apply_parameters(::FBMKernel, θ) = FBMKernel(; h=only(θ))

extract_parameters(k::MaternKernel) = ParameterHandling.positive(k.ν)
apply_parameters(::MaternKernel, θ) = MaternKernel(; ν=only(θ))

extract_parameters(k::PeriodicKernel) = ParameterHandling.positive(k.r)
apply_parameters(::PeriodicKernel, θ) = PeriodicKernel(; r=θ)

extract_parameters(k::LinearKernel) = ParameterHandling.positive(k.c)
apply_parameters(::LinearKernel, θ) = LinearKernel(; c=only(θ))

extract_parameters(k::PolynomialKernel) = ParameterHandling.positive(k.c)
apply_parameters(::PolynomialKernel, θ) = PolynomialKernel(; c=only(θ))

extract_parameters(k::RationalKernel) = ParameterHandling.positive(k.α)
apply_parameters(::RationalKernel, θ) = RationalKernel(; α=only(θ))

extract_parameters(k::RationalQuadraticKernel) = ParameterHandling.positive(k.α)
apply_parameters(::RationalQuadraticKernel, θ) = RationalQuadraticKernel(; α=only(θ))

extract_parameters(k::GammaRationalKernel) = (
    ParameterHandling.positive(k.α), 
    ParameterHandling.bounded(k.γ, 0.0, 2.0)
)
apply_parameters(::GammaRationalKernel, θ) = GammaRationalKernel(; α=only(θ[1]), γ=only(θ[2]))

# kernels (see KernelFunctions.jl src/kernels)
extract_parameters(k::KernelProduct) = map(extract_parameters, k.kernels)
apply_parameters(k::KernelProduct, θ) = KernelProduct(map(apply_parameters, k.kernels, θ))

extract_parameters(k::KernelSum) = map(extract_parameters, k.kernels)
apply_parameters(k::KernelSum, θ) = KernelSum(map(apply_parameters, k.kernels, θ))

extract_parameters(k::KernelTensorProduct) = map(extract_parameters, k.kernels)
apply_parameters(k::KernelTensorProduct, θ) = KernelTensorProduct(map(apply_parameters, k.kernels, θ))

extract_parameters(k::NormalizedKernel) = extract_parameters(k.kernel)
apply_parameters(k::NormalizedKernel, θ) = NormalizedKernel(apply_parameters(k.kernel, θ))

extract_parameters(k::ScaledKernel) = (extract_parameters(k.kernel), ParameterHandling.positive(only(k.σ²)))
apply_parameters(k::ScaledKernel, θ) = ScaledKernel(apply_parameters(k.kernel, θ[1]), θ[2])

extract_parameters(k::TransformedKernel) = (extract_parameters(k.kernel), extract_parameters(k.transform))
apply_parameters(k::TransformedKernel, θ) = TransformedKernel(
    apply_parameters(k.kernel, θ[1]), apply_parameters(k.transform, θ[2])
    )

# transform (see KernelFunctions.jl src/transform)
extract_parameters(t::ARDTransform) = ParameterHandling.positive(t.v)
apply_parameters(::ARDTransform, θ) = ARDTransform(θ)

extract_parameters(t::ChainTransform) = map(extract_parameters, t.transforms)
apply_parameters(t::ChainTransform, θ) = ChainTransform(map(apply_parameters, t.transforms, θ))

extract_parameters(t::LinearTransform) = t.A
apply_parameters(::LinearTransform, θ) = LinearTransform(θ)

extract_parameters(t::PeriodicTransform) = ParameterHandling.positive(t.f)
apply_parameters(::PeriodicTransform, θ) = PeriodicTransform(θ)

extract_parameters(t::ScaleTransform) = ParameterHandling.positive(t.s)
apply_parameters(::ScaleTransform, θ) = ScaleTransform(θ)

# ---------------- Gaussian Processes ----------------

extract_parameters(f::GP) = (extract_parameters(f.mean), extract_parameters(f.kernel))
apply_parameters(f::GP, θ) = GP(
    apply_parameters(f.mean, θ[1]), 
    apply_parameters(f.kernel, θ[2])
)

"""
    NoisyGP(gp::GP, σ²::Real)

Wraps a Gaussian process `gp` and adds learnable Gaussian observation noise with zero mean and variance `σ²` to the diagonal of its finite-dimensional covariance matrix.
"""
struct NoisyGP{T<:GP,Tn<:Real}
    gp::T
    σ²::Tn
end

(gp::NoisyGP)(x) = gp.gp(x, gp.σ²)

"""
    with_gaussian_noise(gp::AbstractGPs.GP, σ²::Real)

Wraps a Gaussian process `gp` with additive Gaussian observation noise of variance `σ²`.

This creates a [`NoisyGP`](@ref) object, which adds `σ²` to the diagonal of the covariance
matrix when evaluating the finite-dimensional projection of `gp`.

# Examples
```jldoctest
julia> using AbstractGPs

julia> gp = GP(SqExponentialKernel());

julia> noisy_gp = with_gaussian_noise(gp, 0.1);
```
"""
with_gaussian_noise(gp::GP, σ²::Real) = NoisyGP(gp, σ²)

extract_parameters(f::NoisyGP) = (
    extract_parameters(f.gp), 
    ParameterHandling.positive(f.σ², exp, 1e-6)
)
apply_parameters(f::NoisyGP, θ) = NoisyGP(apply_parameters(f.gp, θ[1]), θ[2])