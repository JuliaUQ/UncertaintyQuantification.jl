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
            propagate_intervals!(ipm_models, df, bound)
            append!(evaluated_models, names(ipm_models))
        end
    end

    interval_cols = findall(eltype.(eachcol(df)) .== UncertaintyQuantification.Interval)

    interval_vec_cols = findall(
        eltype.(eachcol(df)) .== Vector{IntervalArithmetic.Interval{Float64}}
    )

    # if isempty(interval_cols) && isempty(interval_vec_cols) && length(models) > 1
    #     evaluate!(models[1], df)
    #     propagate_intervals!(models[2:end], df, bound)
    #     return nothing
    # end

    interval_names = propertynames(df[:, interval_cols])
    interval_vec_name = propertynames(df[:, interval_vec_cols])[1]

    output = name(models[end])

    y = map(eachrow(df)) do row
        degenerates = isdegenerate.(collect(row[interval_names]))
        pure = .!degenerates

        lb, ub = if any(degenerates)
            [
                getproperty.(collect(row[interval_names[pure]]), :lb)...,
                getproperty.(getproperty.(row[interval_vec_name], :bareinterval), :lo)...,
            ],
            [
                getproperty.(collect(row[interval_names[pure]]), :ub)...,
                getproperty.(getproperty.(row[interval_vec_name], :bareinterval), :hi)...,
            ]
        else
            [
                getproperty.(collect(row[interval_names]), :lb)...,
                getproperty.(getproperty.(row[interval_vec_name], :bareinterval), :lo)...,
            ],
            [
                getproperty.(collect(row[interval_names]), :ub)...,
                getproperty.(getproperty.(row[interval_vec_name], :bareinterval), :hi)...,
            ]
        end

        x0 = middle.(lb, ub)

        # create a  single-row DataFrame for evaluation
        precise_df = hcat(
            DataFrame([[0.0] for _ in 1:length(interval_names)], interval_names),
            DataFrame(
                [[[0.0]] for _ in 1:length([interval_vec_name])], [interval_vec_name]
            ),
            select(DataFrame(row), Not([interval_names..., interval_vec_name])),
        )

        # set degenerate intervals to their precise value
        if any(degenerates)
            precise_df[1, interval_names[degenerates]] .= getproperty.(
                collect(row[interval_names[degenerates]]), :lb
            )
        end

        function f(x)
            precise_df[1, interval_names[pure]] .= x[1:length(interval_cols)]
            precise_df[1, interval_vec_name] = x[(length(interval_cols) + 1):end]

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
            elseif y isa IntervalArithmetic.Interval
                return y.bareinterval.lo
            else
                return y
            end
        end

        function f_ub(x)
            y = f(x)
            if y isa Interval
                return y.ub
            elseif y isa IntervalArithmetic.Interval
                return y.bareinterval.hi
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
                min_mesh_size=1e-8,
            ).f
        elseif bound == :ub
            return -minimize(
                OrthoMADS(length(x0)),
                x -> -f_ub(x),
                x0;
                lowerbound=lb,
                upperbound=ub,
                min_mesh_size=1e-8,
            ).f
        else
            result_lb = minimize(
                OrthoMADS(length(x0)),
                f_lb,
                x0;
                lowerbound=lb,
                upperbound=ub,
                min_mesh_size=1e-8,
            )

            result_ub = minimize(
                OrthoMADS(length(x0)),
                x -> -f_ub(x),
                x0;
                lowerbound=lb,
                upperbound=ub,
                min_mesh_size=1e-8,
            )

            return Interval(result_lb.f, -result_ub.f)
        end
    end

    df[!, output] = y

    return nothing
end
