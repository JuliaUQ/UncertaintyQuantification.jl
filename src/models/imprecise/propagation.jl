function propagate_intervals!(
    models::Union{UQModel,AbstractVector{<:UQModel}}, df::DataFrame, bound::Symbol=:both
)
    if bound ∉ [:lb, :ub, :both]
        error("Invalid bound: $bound")
    end

    models = wrap(models)

    evaluated_models = Symbol[]

    # evaluate any IPM that is not the last
    for (i, m) in enumerate(models)
        if isa(m, IntervalPredictorModel) && i != length(models)
            ipm_models = [filter(x -> name(x) in m.inputs, models)..., m]
            propagate_intervals!(ipm_models, df)
            append!(evaluated_models, names(ipm_models))
        end
    end

    interval_cols = findall(eltype.(eachcol(df)) .== Interval)

    if isempty(interval_cols) && !isimprecise(UQInput[], models)
        error("No intervals to propagate.")
    end

    # if all inputs are precise and the only model is an imprecise model we just evaluate it.
    if isempty(interval_cols) && length(models) == 1 && isimprecise(only(models))
        evaluate!(only(models), df)
        return nothing
    end

    interval_names = propertynames(df[:, interval_cols])

    output = name(models[end])

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
            precise_df[1, interval_names[degenerates]] .= getproperty.(
                collect(row[interval_names[degenerates]]), :lb
            )
        end

        function f(x)
            precise_df[1, interval_names[pure]] .= x

            if !isempty(evaluated_models)
                # skip models already evaluated
                evaluate!(filter(m -> !(name(m) in evaluated_models), models), precise_df)
            else
                evaluate!(models, precise_df)
            end

            return precise_df[1, output]
        end

        function f_lb(x)
            y = f(x)
            if y isa Interval
                return y.lb
            else
                return y
            end
        end

        function f_ub(x)
            y = f(x)
            if y isa Interval
                return y.ub
            else
                return y
            end
        end

        if bound == :lb
            return minimize(
                OrthoMADS(length(x0)),
                f_lb,
                x0;
                lowerbound=lb,
                upperbound=ub,
                min_mesh_size=1e-13,
            ).f
        elseif bound == :ub
            return -minimize(
                OrthoMADS(length(x0)),
                x -> -f_ub(x),
                x0;
                lowerbound=lb,
                upperbound=ub,
                min_mesh_size=1e-13,
            ).f
        else
            result_lb = minimize(
                OrthoMADS(length(x0)),
                f_lb,
                x0;
                lowerbound=lb,
                upperbound=ub,
                min_mesh_size=1e-13,
            )

            result_ub = minimize(
                OrthoMADS(length(x0)),
                x -> -f_ub(x),
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
