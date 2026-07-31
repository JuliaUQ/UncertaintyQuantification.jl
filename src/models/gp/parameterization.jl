"""
# Developer Note

`Parameterized(object)` wraps `object` so it can be called with parameters `θ`.

`parameterize(object)` returns a parameterized, callable version of the object and its parameters.
    ```julia
    model, θ = parameterize(obj)
    model(θ)  # returns a new object with parameters applied
    This works for mean functions, kernels, transformations, and Gaussian processes.

Based on two core functions, this system can extract model parameters for an optimization routine 
and apply potentially constrained parameters to the underlying model to compute the optimization objective. 
The two core functions are:

    1.  extract_parameters(obj)

    Returns the free parameters of obj wrapped in ParameterHandling containers.
    Enforces constraints (e.g., positive or bounded) where applicable.
    For composite objects (like a `AbstractGPs.GP`), returns a tuple of componentwise parameter sets.
    Returns nothing for objects without trainable parameters.

    2.  apply_parameters(obj, θ)

    Returns a new object of the same type with parameters θ applied.
    For hierarchical objects, θ is expected to match the structure returned by extract_parameters.
"""

struct Parameterized{T}
    object::T
end

function (p::Parameterized)(θ)
    return apply_parameters(p.object, ParameterHandling.value(θ))
end

parameterize(object) = Parameterized(object), extract_parameters(object)

extract_parameters(::ZeroMean) = nothing
apply_parameters(m::ZeroMean, θ) = m

extract_parameters(m::ConstMean) = m.c
apply_parameters(::ConstMean, θ) = ConstMean(θ)

extract_parameters(::CustomMean) = nothing
apply_parameters(m::CustomMean, θ) = m

extract_parameters(::ZeroKernel) = nothing
apply_parameters(k::ZeroKernel, _) = k

extract_parameters(::WhiteKernel) = nothing
apply_parameters(k::WhiteKernel, _) = k

extract_parameters(::CosineKernel) = nothing
apply_parameters(k::CosineKernel, _) = k

extract_parameters(::SqExponentialKernel) = nothing
apply_parameters(k::SqExponentialKernel, _) = k

extract_parameters(::ExponentialKernel) = nothing
apply_parameters(k::ExponentialKernel, _) = k

extract_parameters(::ExponentiatedKernel) = nothing
apply_parameters(k::ExponentiatedKernel, _) = k

extract_parameters(::Matern32Kernel) = nothing
apply_parameters(k::Matern32Kernel, _) = k

extract_parameters(::Matern52Kernel) = nothing
apply_parameters(k::Matern52Kernel, _) = k

extract_parameters(::Matern72Kernel) = nothing
apply_parameters(k::Matern72Kernel, _) = k

extract_parameters(::NeuralNetworkKernel) = nothing
apply_parameters(k::NeuralNetworkKernel, _) = k

extract_parameters(::ZeroKernel) = nothing
apply_parameters(k::ZeroKernel, _) = k

extract_parameters(::PiecewisePolynomialKernel) = nothing
apply_parameters(k::PiecewisePolynomialKernel, _) = k

extract_parameters(::WienerKernel) = nothing
apply_parameters(k::WienerKernel, _) = k

extract_parameters(::FunctionTransform) = nothing
apply_parameters(t::FunctionTransform, _) = t

extract_parameters(::SelectTransform) = nothing
apply_parameters(t::SelectTransform, _) = t

extract_parameters(::IdentityTransform) = nothing
apply_parameters(t::IdentityTransform, _) = t

function extract_parameters(::IndependentMOKernel)
    throw(
        ArgumentError("IndependentMOKernel not supported for hyper parameter optimization.")
    )
end
function extract_parameters(::IntrinsicCoregionMOKernel)
    throw(
        ArgumentError(
            "IntrinsicCoregionMOKernel not supported hyper parameter optimization."
        ),
    )
end
function extract_parameters(::LatentFactorMOKernel)
    throw(ArgumentError("LatentFactorMOKernel not supported hyper parameter optimization."))
end
function extract_parameters(::LinearMixingModelKernel)
    throw(
        ArgumentError("LinearMixingModelKernel not supported hyper parameter optimization.")
    )
end
function extract_parameters(::KernelFunctions.NeuralKernelNetwork)
    throw(ArgumentError("NeuralKernelNetwork not supported hyper parameter optimization."))
end
function extract_parameters(::GibbsKernel)
    throw(ArgumentError("GibbsKernel not supported hyper parameter optimization."))
end

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

function extract_parameters(k::GammaRationalKernel)
    (ParameterHandling.positive(k.α), ParameterHandling.bounded(k.γ, 0.0, 2.0))
end
function apply_parameters(::GammaRationalKernel, θ)
    GammaRationalKernel(; α=only(θ[1]), γ=only(θ[2]))
end

# kernels (see KernelFunctions.jl src/kernels)
extract_parameters(k::KernelProduct) = map(extract_parameters, k.kernels)
apply_parameters(k::KernelProduct, θ) = KernelProduct(map(apply_parameters, k.kernels, θ))

extract_parameters(k::KernelSum) = map(extract_parameters, k.kernels)
apply_parameters(k::KernelSum, θ) = KernelSum(map(apply_parameters, k.kernels, θ))

extract_parameters(k::KernelTensorProduct) = map(extract_parameters, k.kernels)
function apply_parameters(k::KernelTensorProduct, θ)
    KernelTensorProduct(map(apply_parameters, k.kernels, θ))
end

extract_parameters(k::NormalizedKernel) = extract_parameters(k.kernel)
apply_parameters(k::NormalizedKernel, θ) = NormalizedKernel(apply_parameters(k.kernel, θ))

function extract_parameters(k::ScaledKernel)
    (extract_parameters(k.kernel), ParameterHandling.positive(only(k.σ²)))
end
apply_parameters(k::ScaledKernel, θ) = ScaledKernel(apply_parameters(k.kernel, θ[1]), θ[2])

function extract_parameters(k::TransformedKernel)
    (extract_parameters(k.kernel), extract_parameters(k.transform))
end
function apply_parameters(k::TransformedKernel, θ)
    TransformedKernel(apply_parameters(k.kernel, θ[1]), apply_parameters(k.transform, θ[2]))
end

# transform (see KernelFunctions.jl src/transform)
extract_parameters(t::ARDTransform) = ParameterHandling.positive(t.v)
apply_parameters(::ARDTransform, θ) = ARDTransform(θ)

extract_parameters(t::ChainTransform) = map(extract_parameters, t.transforms)
function apply_parameters(t::ChainTransform, θ)
    ChainTransform(map(apply_parameters, t.transforms, θ))
end

extract_parameters(t::LinearTransform) = t.A
apply_parameters(::LinearTransform, θ) = LinearTransform(θ)

extract_parameters(t::PeriodicTransform) = ParameterHandling.positive(t.f)
apply_parameters(::PeriodicTransform, θ) = PeriodicTransform(θ)

extract_parameters(t::ScaleTransform) = ParameterHandling.positive(t.s)
apply_parameters(::ScaleTransform, θ) = ScaleTransform(θ)

# ---------------- Gaussian Processes ----------------
extract_parameters(f::GP) = (extract_parameters(f.mean), extract_parameters(f.kernel))
function apply_parameters(f::GP, θ)
    GP(apply_parameters(f.mean, θ[1]), apply_parameters(f.kernel, θ[2]))
end

"""
Internal struct for hyperparameter optimization. The struct saves the GP that is to be optimized and the noise.
"""
struct PriorGP{T<:GP,Tn<:Real}
    gp::T
    σ²::Tn
    learn_noise::Bool
end

(gp::PriorGP)(x) = gp.gp(x, gp.σ²)
"""
    with_gaussian_noise(gp::AbstractGPs.GP, σ²::Real)

Wraps a Gaussian process `gp` with additive Gaussian observation noise of variance `σ²`.

This creates a Gaussian process object, which adds `σ²` to the diagonal of the covariance
matrix when evaluating the finite-dimensional projection of `gp`.

# Examples
```jldoctest
julia> gp = GP(SqExponentialKernel());

julia> noisy_gp = with_gaussian_noise(gp, 0.1);
```
"""

extract_parameters(f::PriorGP) = begin 
   f.learn_noise ? (
    extract_parameters(f.gp), 
    ParameterHandling.positive(f.σ², exp, 1e-6)
    ) :
    (extract_parameters(f.gp))
end

apply_parameters(f::PriorGP, θ) = begin
    f.learn_noise ? 
    PriorGP(apply_parameters(f.gp, θ[1]), θ[2], f.learn_noise) :
    PriorGP(apply_parameters(f.gp, θ), f.σ², f.learn_noise)
end
