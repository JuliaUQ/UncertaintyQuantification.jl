function propagate_intervals!(
    m::Union{UQModel,AbstractVector{<:UQModel}}, df::DataFrame, bound::Symbol=:both
)
    if bound ∉ [:lb, :ub, :both]
        error("Invalid bound: $bound")
    end

    interval_cols = findall(eltype.(eachcol(df)) .== Interval)

    interval_names = propertynames(df[:, interval_cols])

    output = isa(m, AbstractVector) ? m[end].name : m.name

    y = map(eachrow(df)) do row
        degenerates = isdegenerate.(collect(row[interval_names]))
        pure = .!degenerates

        lb, ub = if any(degenerates)
            getproperty.(collect(row[interval_names[pure]]), :lb),
            getproperty.(collect(row[interval_names[pure]]), :ub)
        else
            getproperty.(collect(row[interval_names]), :lb),
            getproperty.(collect(row[interval_names]), :ub)
        end

        x0 = middle.(lb, ub)

        # create a  single-row DataFrame for evaluation
        precise_df = hcat(
            DataFrame([[0.0] for _ in 1:length(interval_names)], interval_names),
            select(DataFrame(row), Not(interval_names)),
        )

        # set degenerate intervals to their precise value
        if any(degenerates)
            precise_df[1, interval_names[degenerates]] .=
                getproperty.(collect(row[interval_names[degenerates]]), :lb)
        end

        function f(x)
            precise_df[1, interval_names[pure]] .= x

            evaluate!(m, precise_df)

            return precise_df[1, output]
        end

        if bound == :lb
            return minimize(
                OrthoMADS(length(x0)),
                f,
                x0;
                lowerbound=lb,
                upperbound=ub,
                min_mesh_size=1e-13,
            ).f
        elseif bound == :ub
            return -minimize(
                OrthoMADS(length(x0)),
                x -> -f(x),
                x0;
                lowerbound=lb,
                upperbound=ub,
                min_mesh_size=1e-13,
            ).f
        else
            result_lb = minimize(
                OrthoMADS(length(x0)),
                f,
                x0;
                lowerbound=lb,
                upperbound=ub,
                min_mesh_size=1e-13,
            )

            result_ub = minimize(
                OrthoMADS(length(x0)),
                x -> -f(x),
                x0;
                lowerbound=lb,
                upperbound=ub,
                min_mesh_size=1e-13,
            )

            return Interval(result_lb.f, -result_ub.f)
        end
    end

    df[!, output] = y

    return nothing
end
