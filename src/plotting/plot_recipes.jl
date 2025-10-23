DEFAULT_ALPHA = 0.2
DEFAULT_LABEL = ""
DEFAULT_GRID = false
DEFAULT_LEGEND = false
DEFAULT_CDF = false
DEFAULT_DISTRIBUTION = :pdf
DEFAULT_FILL = :gray
DEFAULT_COLOUR_PDF = :blue
DEFAULT_COLOUR_UPPER = :red
DEFAULT_COLOUR_LOWER = :black

DEFAULT_INTERVAL_WIDTH=1.5
DEFAULT_INTERVAL_EDGE_ALPHA=1

DEFAULT_PLOT_RANGE_EXTEND_DENSITY = 0.2
DEFAULT_PLOT_RANGE_EXTEND = 0.2
DEFAULT_PLOT_RANGE_INTERVAL = 0.4
DEFAULT_PLOT_GRID_NUMBER = 500
DEFAULT_FONT_SIZE = 18
DEFAULT_TICK_SIZE = 12

###
#   Plots for UQInputs
###
@recipe function _plot(x::RandomVariable{T}; cdf_on = DEFAULT_CDF) where {T <: UnivariateDistribution}

    grid --> DEFAULT_GRID
    legend --> DEFAULT_LEGEND
    xlabel --> x.name
    cdf_on ? ylabel --> "cdf" : ylabel --> "pdf"

    # cdf_on --> DEFAULT_CDF

    lo_grid = quantile(x, 0.001)
    hi_grid = quantile(x, 0.999)

    width = hi_grid - lo_grid

    lo_grid = lo_grid - abs(width * DEFAULT_PLOT_RANGE_EXTEND_DENSITY)
    hi_grid = hi_grid + abs(width * DEFAULT_PLOT_RANGE_EXTEND_DENSITY)

    x_grid = range(lo_grid, hi_grid, DEFAULT_PLOT_GRID_NUMBER)

    if cdf_on
        distribution_evals = cdf.(Ref(x), x_grid)
    else
        distribution_evals = pdf.(Ref(x), x_grid)
    end
    
    if ~cdf_on
        @series begin
            fillrange := distribution_evals
            color := DEFAULT_FILL
            fillalpha := DEFAULT_ALPHA
            x_grid, zeros(length(x_grid))
        end
    end

    
    @series begin
        color --> DEFAULT_COLOUR_PDF
        alpha := 1
        label := ""
        x_grid, distribution_evals
    end
end

@recipe function _plot(x::IntervalVariable)

    grid --> DEFAULT_GRID
    legend --> DEFAULT_LEGEND
    ylabel --> "cdf"
    xlabel --> x.name

    lo_grid = x.lb
    hi_grid = x.ub

    width = hi_grid - lo_grid

    plot_lo = lo_grid - abs(width * DEFAULT_PLOT_RANGE_INTERVAL)
    plot_hi = hi_grid + abs(width * DEFAULT_PLOT_RANGE_INTERVAL)

    xlims := (plot_lo, plot_hi)

    x_grid = range(lo_grid, hi_grid, DEFAULT_PLOT_GRID_NUMBER)

    cdf_lo = x_grid .>= x.ub
    cdf_hi = x_grid .> x.lb
    
    @series begin
        fillrange := cdf_hi
        color := DEFAULT_FILL
        fillalpha := DEFAULT_ALPHA
        x_grid, cdf_lo
    end

    @series begin
        color --> DEFAULT_COLOUR_LOWER
        alpha := 1
        label := ""
        x_grid, cdf_lo
    end

    @series begin
        color --> DEFAULT_COLOUR_UPPER
        alpha := 1
        label := ""
        x_grid, cdf_hi
    end

end

@recipe function _plot(x::RandomVariable{T}) where {T <: ProbabilityBox}

    grid --> DEFAULT_GRID
    legend --> DEFAULT_LEGEND
    ylabel --> "cdf"
    xlabel --> x.name

    lo_grid = quantile(x, 0.001).lb
    hi_grid = quantile(x, 0.999).ub

    width = hi_grid - lo_grid

    lo_grid = lo_grid - abs(width * DEFAULT_PLOT_RANGE_EXTEND)
    hi_grid = hi_grid + abs(width * DEFAULT_PLOT_RANGE_EXTEND)

    x_grid = range(lo_grid, hi_grid, DEFAULT_PLOT_GRID_NUMBER)
    cdf_evals = cdf.(Ref(x), x_grid)

    @series begin
        fillrange := hi.(cdf_evals)
        color := DEFAULT_FILL
        fillalpha := DEFAULT_ALPHA
        x_grid, lo.(cdf_evals)
    end

    @series begin
        color --> DEFAULT_COLOUR_LOWER
        alpha := 1
        label := ""
        x_grid, lo.(cdf_evals)
    end

    @series begin
        color --> DEFAULT_COLOUR_UPPER
        alpha := 1
        label := ""
        x_grid, hi.(cdf_evals)
    end
end

@recipe function _plot(x::Vector{T}) where T <: UQInput

    x_no_params = filter(x -> !isa(x, Parameter), x)

    N_inputs = length(x_no_params)

    cols = ceil(Int, sqrt(N_inputs))  # Calculate the number of columns
    rows = ceil(Int, N_inputs / cols)  # Calculate the number of rows needed
    layout := (rows, cols)  # Create a grid layout

    for i = 1:N_inputs
        @series begin
            subplot := i
            x_no_params[i]
        end
    end
end
###
#   This code is a modified version of the plot recipe from IntervalArithmetic.jl
#       https://github.com/JuliaIntervals/IntervalArithmetic.jl       
###

# Plot a 2D IntervalBox:
@recipe function _plot(x::Interval, y::Interval)

    seriesalpha --> DEFAULT_ALPHA
    seriestype := :shape

    label := false

    linecolor --> :black                        # Explicitly set edge color
    linewidth --> DEFAULT_INTERVAL_WIDTH        # Make edges more visible
    linealpha --> DEFAULT_INTERVAL_EDGE_ALPHA

    x = [x.lb, x.ub, x.ub, x.lb]
    y = [y.lb, y.lb, y.ub, y.ub]

    x, y
end

# Plot a vector of 2D IntervalBoxes:
@recipe function _plot(xx::Vector{T}, yy::Vector{T}) where T<:Interval

    seriesalpha --> DEFAULT_ALPHA
    seriestype := :shape

    label := false

    linecolor := :black                         # Explicitly set edge color
    linewidth --> DEFAULT_INTERVAL_WIDTH        # Make edges more visible
    linealpha --> DEFAULT_INTERVAL_EDGE_ALPHA

    xs = Float64[]
    ys = Float64[]

    # build up coords:  # (alternative: use @series)
    for i = 1:length(xx)
        (x, y) = (xx[i], yy[i])

        # use NaNs to separate
        append!(xs, [x.lb, x.ub, x.ub, x.lb, NaN])
        append!(ys, [y.lb, y.lb, y.ub, y.ub, NaN])

    end

    xs, ys

end

###
# Plots for samples of data frames
###

@recipe function _plot(x::Vector{Interval})

    if length(unique(x))==1
        return x[1]
    else
        grid --> DEFAULT_GRID
        legend --> DEFAULT_LEGEND

        # xlabel --> x[1].name
        ylabel --> "cdf"
        N_samples = length(x)

        lows = sort(lo.(x))
        his = sort(hi.(x))

        is = range(0, 1, length = N_samples)

        @series begin
            seriestype := :steppre
            color --> DEFAULT_COLOUR_LOWER
            lows, is
        end

        @series begin
            seriestype := :steppost
            color --> DEFAULT_COLOUR_UPPER
            his, is
        end
    end
end