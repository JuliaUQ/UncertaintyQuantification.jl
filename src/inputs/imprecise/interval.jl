"""
    Interval(lb::Real, ub::Real)

Represents a closed numeric interval with a lower bound `lb` and an upper bound `ub`.

`Interval` is a data type primarily used for constructing probability boxes (p-boxes) and other uncertainty representations. It is **not** intended for direct use in simulations for that, see [`IntervalVariable`](@ref).

# Fields
- `lb::Real`: Lower bound of the interval.
- `ub::Real`: Upper bound of the interval.

# Examples

```jldoctest
julia> Interval(0.10, 0.14)
[0.1, 0.14]
```
"""
struct Interval
    lb::Real
    ub::Real
    function Interval(lb::Real, ub::Real)
        lb > ub && error(
            "Lower bound parameter must be smaller than upper bound parameter for Intervals.",
        )
        return new(lb, ub)
    end
end

function Base.show(io::IO, i::Interval)
    print(io, "[$(i.lb), $(i.ub)]")
    return nothing
end

Base.in(u, i::Interval) = i.lb <= u <= i.ub

Base.broadcastable(i::Interval) = Ref(i)

function isdegenerate(i::Interval)
    return i.lb == i.ub
end

function bounds(i::Interval)
    return i.lb, i.ub
end

"""
    IntervalVariable(lb::Real, ub::Real, name::Symbol)

Defines an interval variable with a lower bound `lb`, upper bound `ub`, and an identifying `name`.

`IntervalVariable` can be passed directly to analyses and simulations.
For other uses, such as building probability boxes (p-boxes) from interval parameters, use [`Interval`](@ref) instead.

# Fields
- `lb::Real`: Lower bound of the interval.
- `ub::Real`: Upper bound of the interval.
- `name::Symbol`: Name or identifier for the variable.

# Examples

```jldoctest
julia> IntervalVariable(0.10, 0.14, :x)
x ∈ [0.1, 0.14]
```
"""
struct IntervalVariable <: UQInput
    lb::Real
    ub::Real
    name::Symbol
    function IntervalVariable(lb::Real, ub::Real, name::Symbol)
        lb > ub && error(
            "Lower bound parameter must be smaller than upper bound parameter for Interval $name.",
        )
        return new(lb, ub, name)
    end
end

function Base.show(io::IO, i::IntervalVariable)
    print(io, "$(String(i.name)) ∈ [$(i.lb), $(i.ub)]")
    return nothing
end

function Interval(i::IntervalVariable)
    return Interval(i.lb, i.ub)
end

function sample(i::IntervalVariable, n::Integer=1)
    return DataFrame(i.name => fill(Interval(i), n))
end

function map_to_precise(x::Real, input::IntervalVariable)
    if !in(x, input)
        error("$x not in [$(input.lb), $(input.ub)] for Interval $(input.name).")
    end
    return Parameter(x, input.name)
end

to_standard_normal_space!(_::IntervalVariable, _::DataFrame) = nothing
to_physical_space!(_::IntervalVariable, _::DataFrame) = nothing

function bounds(i::IntervalVariable)
    return i.lb, i.ub
end

Base.in(u, i::IntervalVariable) = i.lb <= u <= i.ub

mean(i::IntervalVariable) = Interval(i)
var(i::IntervalVariable) = Interval(0.0, (i.ub - i.lb)^2 / 4)
