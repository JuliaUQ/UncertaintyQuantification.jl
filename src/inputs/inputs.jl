"""
	sample(inputs::Vector{<:UQInput}, n::Integer=1)

Generates n correlated samples from a collection of inputs. Returns a DataFrame

See also: [`RandomVariable`](@ref), [`Parameter`](@ref)
"""
function sample(inputs::Vector{<:UQInput}, n::Integer=1)
    return mapreduce(i -> sample(i, n), hcat, inputs)
end

function to_physical_space!(inputs::Vector{<:UQInput}, x::DataFrame)
    for i in inputs
        to_physical_space!(i, x)
    end
    return nothing
end

function to_standard_normal_space!(inputs::Vector{<:UQInput}, x::DataFrame)
    for i in inputs
        to_standard_normal_space!(i, x)
    end
    return nothing
end

function names(inputs::Vector{<:UQInput})
    _names = Symbol[]

    for i in inputs
        if i isa JointDistribution || i isa SpectralRepresentation
            append!(_names, names(i))
        else
            push!(_names, getproperty(i, :name))
        end
    end

    return _names
end

names(input::UQInput) = only(names([input])) # need this to get the name of a single input in gps

function count_rvs(inputs::Vector{<:UQInput})
    random_inputs = filter(i -> isa(i, RandomUQInput) || isa(i, ProbabilityBox), inputs)
    return mapreduce(dimensions, +, random_inputs)
end

mean(inputs::Vector{<:UQInput}) = mapreduce(mean, vcat, inputs)

function sns_zero_point(inputs::AbstractVector{<:UQInput})
    random_inputs = filter(i -> isa(i, RandomUQInput), inputs)
    deterministic_inputs = filter(i -> isa(i, DeterministicUQInput), inputs)

    sns = DataFrame(names(random_inputs) .=> zeros(count_rvs(random_inputs)))

    if !isempty(deterministic_inputs)
        DataFrames.hcat!(sns, sample(deterministic_inputs, 1))
    end

    return sns
end

Base.broadcastable(i::T) where {T<:UQInput} = Ref(i)
