abstract type AbstractPowerSpectralDensity end

struct CloughPenzien <: AbstractPowerSpectralDensity
    ω::AbstractVector{<:Real}
    S_0::Real
    ω_f::Real
    ζ_f::Real
    ω_g::Real
    ζ_g::Real
    p::AbstractVector{<:Real}
end

"""
    CloughPenzien(ω::AbstractVector{<:Real}, S_0::Real, ω_f::Real, ζ_f::Real, ω_g::Real, ζ_g::Real)

Constructs a `CloughPenzien` instance representing a power spectral density function with the given parameters.

# Arguments / Parameters
- `ω::AbstractVector{<:Real}`: A vector of angular frequencies.
- `S_0::Real`: A scaling factor.
- `ω_f::Real`: Frequency parameter for the first oscillator.
- `ζ_f::Real`: Damping ratio for the first oscillator.
- `ω_g::Real`: Frequency parameter for the second oscillator.
- `ζ_g::Real`: Damping ratio for the second oscillator.

# Returns
A discretized `CloughPenzien` power spectral density function specified by given arguments (parameters).

# Example
```julia
w = 0:0.1:10
S_0 = 1.0
ω_f = 2.0
ζ_f = 0.05
ω_g = 3.0
ζ_g = 0.1
cp_psd = CloughPenzien(w, S_0, ω_f, ζ_f, ω_g, ζ_g)
```
"""
function CloughPenzien(
    ω::AbstractVector{<:Real}, S_0::Real, ω_f::Real, ζ_f::Real, ω_g::Real, ζ_g::Real
)
    p =
        S_0 * ((ω .^ 4) ./ ((ω_f^2 .- ω .^ 2) .^ 2 .+ 4 * ζ_f^2 * ω_f^2 * ω .^ 2)) .* (
            (ω_g^4 .+ 4 * ζ_g^2 * ω_g^2 * ω .^ 2) ./
            ((ω_g^2 .- ω .^ 2) .^ 2 .+ (4 * ζ_g^2 * ω_g^2 * ω .^ 2))
        )

    return CloughPenzien(ω, S_0, ω_f, ζ_f, ω_g, ζ_g, p)
end

struct KanaiTajimi <: AbstractPowerSpectralDensity
    ω::AbstractVector{<:Real}
    S_0::Real
    ω_0::Real
    ζ::Real
    p::AbstractVector{<:Real}
end

"""
    KanaiTajimi(ω::AbstractVector{<:Real}, S_0::Real, ω_0::Real, ζ::Real) -> KanaiTajimi

Constructs a `KanaiTajimi` instance representing a power spectral density function with the given parameters.

# Arguments
- `ω::AbstractVector{<:Real}`: A vector of angular frequencies.
- `S_0::Real`: A scaling factor.
- `ω_0::Real`: Natural frequency of the oscillator.
- `ζ::Real`: Damping ratio of the oscillator.

# Returns
A discretized `KanaiTajimi` power spectral density function specified by given arguments (parameters).

# Example
```julia
w = 0:0.1:10
S_0 = 1.0
ω_0 = 2.0
ζ = 0.05
kt = KanaiTajimi(w, S_0, ω_0, ζ)
```
"""
function KanaiTajimi(ω::AbstractVector{<:Real}, S_0::Real, ω_0::Real, ζ::Real)
    p =
        S_0 .* (1 .+ 4 * ζ^2 .* (ω ./ ω_0) .^ 2) ./
        ((1 .- (ω ./ ω_0) .^ 2) .^ 2 .+ 4 * ζ^2 * (ω ./ ω_0) .^ 2)

    return KanaiTajimi(ω, S_0, ω_0, ζ, p)
end

function evaluate(cp_psd::CloughPenzien)
    return cp_psd.p
end

function evaluate(kt::KanaiTajimi)
    return kt.p
end

struct ShinozukaDeodatis <: AbstractPowerSpectralDensity
    ω::AbstractVector{<:Real}
    σ::Real
    b::Real
    p::AbstractVector{<:Real}
end

"""
    ShinozukaDeodatis(ω::AbstractVector{<:Real}, σ::Real, b::Real)

Constructs a `ShinozukaDeodatis` instance representing a power spectral density function with the given parameters.

# Arguments
- `ω::AbstractVector{<:Real}`: A vector of angular frequencies.
- `σ::Real`: A hyperparamter related to the variance of the stochastic process.
- `b::Real`: A parameter related to the correlation length of the stochastic process.

# Returns
A discretized `ShinozukaDeodatis` instance with the power spectral density function specified by given arguments (parameters).

# Example
```julia
w = 0:0.1:10
σ = 1.0
b = 0.5
sd = ShinozukaDeodatis(w, σ, b)
```
"""
function ShinozukaDeodatis(ω::AbstractVector{<:Real}, σ::Real, b::Real)
    p = 1 / 4 * σ^2 * b^3 .* ω .^ 2 .* exp.(-b * abs.(ω))
    return ShinozukaDeodatis(ω, σ, b, p)
end

function evaluate(sd::ShinozukaDeodatis)
    return sd.p
end

"""
    EmpiricalPSD(ω::AbstractVector{<:Real}, p::AbstractVector{<:Real}) -> EmpiricalPSD

Constructs an `EmpiricalPSD` instance with the given angular frequencies and manually provided power spectral density values.

# Arguments
- `ω::AbstractVector{<:Real}`: A vector of angular frequencies.
- `p::AbstractVector{<:Real}`: A vector of power spectral density values corresponding to the frequencies in `ω`.

# Returns
A discretized `EmpiricalPSD` instance with manually pre-specified provided power spectral density values.

# Example
```julia
w = 0:0.1:10
p_values = rand(length(w))  # Example empirical PSD values
emp_psd = EmpiricalPSD(w, p_values)
```
"""
struct EmpiricalPSD <: AbstractPowerSpectralDensity
    ω::AbstractVector{<:Real}
    p::AbstractVector{<:Real}
end

function evaluate(ep::EmpiricalPSD)
    return ep.p
end

struct ImprecisePSD <: AbstractPowerSpectralDensity
    b::AbstractBasis
    p_lb::AbstractVector{<:Real}
    p_ub::AbstractVector{<:Real}
    ω::AbstractVector{<:Real}

    function ImprecisePSD(
        w::AbstractVector{<:Real},
        e::AbstractMatrix{<:Real},
        b::AbstractBasis,
        tol::Real=1e-12,
    )
        y_max = vec(maximum(e; dims=1))
        y_min = vec(minimum(e; dims=1))

        m = JuMP.Model(Clarabel.Optimizer)

        set_attribute(m, "tol_gap_abs", tol)
        set_attribute(m, "tol_gap_rel", tol)
        set_attribute(m, "tol_feas", tol)
        set_attribute(m, "tol_infeas_abs", tol)
        set_attribute(m, "tol_infeas_rel", tol)
        set_silent(m)

        n = length(b)

        φ = b(permutedims(w))
        @variable(m, p_lb[1:n])
        @variable(m, p_ub[1:n])

        @constraint(m, vec(p_lb' * φ) .<= y_min)
        # lower bound must be positive
        @constraint(m, vec(p_lb' * φ) .>= 0.0)
        @constraint(m, vec(p_ub' * φ) .>= y_max)

        @constraint(
            m, vec(p_ub' * ((φ - abs.(φ)) ./ 2) + p_lb' * ((φ + abs.(φ)) ./ 2)) .<= y_min
        )
        @constraint(
            m, vec(p_ub' * ((φ - abs.(φ)) ./ 2) + p_lb' * ((φ + abs.(φ)) ./ 2)) .>= 0.0
        )

        @constraint(
            m, vec(p_ub' * ((φ + abs.(φ)) ./ 2) + p_lb' * ((φ - abs.(φ)) ./ 2)) .>= y_max
        )

        @constraint(m, p_lb <= p_ub)

        @objective(m, Min, mean((p_ub - p_lb)' * abs.(φ)))

        JuMP.optimize!(m)

        return new(b, value.(p_lb), value.(p_ub), w)
    end
end

function evaluate(psd::ImprecisePSD)
    φ = psd.b(permutedims(psd.ω))
    lo = vec(psd.p_ub' * ((φ - abs.(φ)) ./ 2) + psd.p_lb' * ((φ + abs.(φ)) ./ 2))
    hi = vec(psd.p_ub' * ((φ + abs.(φ)) ./ 2) + psd.p_lb' * ((φ - abs.(φ)) ./ 2))
    return interval.(lo, hi)
end

function bounds(psd::ImprecisePSD)
    φ = psd.b(permutedims(psd.ω))
    lo = vec(psd.p_ub' * ((φ - abs.(φ)) ./ 2) + psd.p_lb' * ((φ + abs.(φ)) ./ 2))
    hi = vec(psd.p_ub' * ((φ + abs.(φ)) ./ 2) + psd.p_lb' * ((φ - abs.(φ)) ./ 2))
    return lo, hi
end
