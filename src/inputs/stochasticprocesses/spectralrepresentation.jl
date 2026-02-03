abstract type AbstractStochasticProcess <: RandomUQInput end

struct SpectralRepresentation <: AbstractStochasticProcess
    psd::AbstractPowerSpectralDensity
    time::AbstractVector{<:Real}
    ωt::AbstractMatrix{<:Real}
    Δω::Real
    A::AbstractVector{<:Real}
    name::Symbol
    ϕnames::Vector{Symbol}
end

"""
    SpectralRepresentation(psd::AbstractPowerSpectralDensity, time::AbstractVector{<:Real}, name::Symbol) -> SpectralRepresentation

Constructs a `SpectralRepresentation` instance representing a stochastic process generated using the spectral representation method.

# Arguments
- `psd::AbstractPowerSpectralDensity`: An instance of a power spectral density model.
- `time::AbstractVector{<:Real}`: A vector of time points.
- `name::Symbol`: A symbol representing the name of the process.

# Returns
A `SpectralRepresentation` instance with the given arguments (parameters).

# Example
```julia
w = 0:0.1:10
S_0 = 1.0
ω_f = 2.0
ζ_f = 0.05
ω_g = 3.0
ζ_g = 0.1
psd = CloughPenzien(w, S_0, ω_f, ζ_f, ω_g, ζ_g)
t = 0:0.1:10
name = :process1
sr = SpectralRepresentation(psd, t, name)
```

"""
function SpectralRepresentation(
    psd::AbstractPowerSpectralDensity, time::AbstractVector{<:Real}, name::Symbol
)
    Δω = psd.ω[2] - psd.ω[1]
    ω_u = psd.ω[end] + Δω
    Δt = time[2] - time[1]
    f_max = ω_u/(2π)        # Frequency of the signal
    f_s = 1/Δt              # Sampling Frequency
    f_ny = f_s/2            # Nyquist Frequency

    if f_max > f_ny
        @warn "The frequency of the signal ($f_max Hz) is bigger than the Nyquist frequency ($f_ny Hz). There is a risk for aliasing!"
    end

    A = sqrt.(2 * psd.p * Δω)
    A[iszero.(psd.ω)] .= 0.0

    return SpectralRepresentation(
        psd,
        time,
        psd.ω * time',
        Δω,
        A,
        name,
        [Symbol("$(name)_ϕ_$i") for i in 1:length(psd.ω)],
    )
end

"""
    sample(sr::SpectralRepresentation, n::Integer=1) -> DataFrame

Generates samples of random phase angles for a given `SpectralRepresentation` instance.

# Arguments
- `sr::SpectralRepresentation`: An instance of the `SpectralRepresentation` struct.
- `n::Integer=1`: The number of samples to generate (default is 1).

# Returns
A `DataFrame` containing the generated samples of random phase angles.

# Example
```julia
w = 0:0.1:10
psd = CloughPenzien(w, 1.0, 2.0, 0.05, 3.0, 0.1)
t = 0:0.1:10
name = :process1
sr = SpectralRepresentation(psd, t, name)
samples = sample(sr, 5)
```

"""
function sample(sr::SpectralRepresentation, n::Integer=1)
    return DataFrame(names(sr) .=> eachcol(rand(Uniform(0, 2π), (n, length(sr.psd.ω)))))
end

"""
    evaluate(sr::SpectralRepresentation, ϕ::AbstractVector{<:Real}) -> AbstractVector{<:Real}

Evaluates the stochastic process for a given `SpectralRepresentation` instance and a vector of random phase angles.

# Arguments
- `sr::SpectralRepresentation`: An instance of the `SpectralRepresentation` struct.
- `ϕ::AbstractVector{<:Real}`: A vector of random phase angles.

# Returns
A vector of real numbers representing the evaluated stochastic process.

# Example
```julia
w = 0:0.1:10
psd = CloughPenzien(w, 1.0, 2.0, 0.05, 3.0, 0.1)
t = 0:0.1:10
name = :process1
sr = SpectralRepresentation(psd, t, name)
ϕ = rand(Uniform(0, 2π), length(psd.ω))
process_values = evaluate(sr, ϕ)
```

"""
function evaluate(sr::SpectralRepresentation, ϕ::AbstractVector{<:Real})
    return sqrt(2) * vec(sr.A' * cos.(sr.ωt .+ ϕ))
end

function (sr::SpectralRepresentation)(ϕ::AbstractVector{<:Real})
    return evaluate(sr, ϕ)
end

# function evaluate(sr::SpectralRepresentation, row::DataFrameRow)
#     return evaluate(sr, collect(row[sr.ϕnames]))
# end

# function evaluate(sr::SpectralRepresentation, row::DataFrameRow)
#     return evaluate(sr, collect(row[sr.ϕnames]))
# end

# function (sr::SpectralRepresentation)(row::DataFrameRow)
#     return evaluate(sr, row)
# end

"""
    to_standard_normal_space!(sr::SpectralRepresentation, df::DataFrame) -> Nothing

Transforms the random phase angles in the given `DataFrame` from a uniform distribution to a standard normal distribution.

# Arguments
- `sr::SpectralRepresentation`: An instance of the `SpectralRepresentation` struct.
- `df::DataFrame`: A `DataFrame` containing the random phase angles to be transformed.

# Returns
Nothing. The `DataFrame` is modified in place.

# Example
```julia
w = 0:0.1:10
psd = CloughPenzien(w, 1.0, 2.0, 0.05, 3.0, 0.1)
t = 0:0.1:10
name = :process1
sr = SpectralRepresentation(psd, t, name)
samples = sample(sr, 5)
to_standard_normal_space!(sr, samples)
```

"""
function to_standard_normal_space!(sr::SpectralRepresentation, df::DataFrame)
    for v in names(sr)
        df[!, v] = quantile.(Normal(), cdf.(Uniform(0, 2π), df[:, v]))
    end
    return nothing
end

"""
    to_physical_space!(sr::SpectralRepresentation, df::DataFrame) -> Nothing

Transforms the random phase angles in the given `DataFrame` from a standard normal distribution to a uniform distribution.

# Arguments
- `sr::SpectralRepresentation`: An instance of the `SpectralRepresentation` struct.
- `df::DataFrame`: A `DataFrame` containing the random phase angles to be transformed.

# Returns
Nothing. The `DataFrame` is modified in place.

# Example
```julia
w = 0:0.1:10
psd = CloughPenzien(w, 1.0, 2.0, 0.05, 3.0, 0.1)
t = 0:0.1:10
name = :process1
sr = SpectralRepresentation(psd, t, name)
samples = sample(sr, 5)
to_standard_normal_space!(sr, samples)  # Transform to standard normal space
to_physical_space!(sr, samples)         # Transform back to physical space
```

"""
function to_physical_space!(sr::SpectralRepresentation, df::DataFrame)
    for v in names(sr)
        df[!, v] = quantile.(Uniform(0, 2π), cdf.(Normal(), df[:, v]))
    end
    return nothing
end

"""
    dimensions(sr::SpectralRepresentation) -> Int

Returns the number of dimensions (frequencies) in the given `SpectralRepresentation` instance.

# Arguments
- `sr::SpectralRepresentation`: An instance of the `SpectralRepresentation` struct.

# Returns
An integer representing the number of dimensions (frequencies).

# Example
```julia
w = 0:0.1:10
psd = CloughPenzien(w, 1.0, 2.0, 0.05, 3.0, 0.1)
t = 0:0.1:10
name = :process1
sr = SpectralRepresentation(psd, t, name)
num_dimensions = dimensions(sr)
```

"""
function dimensions(sr::SpectralRepresentation)
    return length(sr.psd.ω)
end

"""
    names(sr::SpectralRepresentation) -> Vector{Symbol}

Returns the names of the random phase angles for a given `SpectralRepresentation` instance.

# Arguments
- `sr::SpectralRepresentation`: An instance of the `SpectralRepresentation` struct.

# Returns
A vector of symbols representing the names of the random phase angles.

# Example
```julia
w = 0:0.1:10
psd = CloughPenzien(w, 1.0, 2.0, 0.05, 3.0, 0.1)
t = 0:0.1:10
name = :process1
sr = SpectralRepresentation(psd, t, name)
phase_angle_names = names(sr)
```

"""
function names(sr::SpectralRepresentation)
    return sr.ϕnames
end