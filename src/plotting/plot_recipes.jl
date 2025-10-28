DEFAULT_ALPHA = 0.2
DEFAULT_LABEL = ""
DEFAULT_GRID = false
DEFAULT_LEGEND = true
DEFAULT_CDF = false
DEFAULT_FILL_DISTRIBUTION=true
DEFAULT_DISTRIBUTION = :pdf
DEFAULT_FILL = :gray
DEFAULT_COLOUR_PDF = :blue
DEFAULT_COLOUR_UPPER = :red
DEFAULT_COLOUR_LOWER = :black

DEFAULT_INTERVAL_WIDTH=1.5
DEFAULT_INTERVAL_EDGE_ALPHA=1

DEFAULT_DISTRIBUTION_WIDTH=2

DEFAULT_PLOT_RANGE_EXTEND_DENSITY = 0.2
DEFAULT_PLOT_RANGE_EXTEND = 0.2
DEFAULT_PLOT_RANGE_INTERVAL = 0.4
DEFAULT_PLOT_GRID_NUMBER = 500
DEFAULT_FONT_SIZE = 18
DEFAULT_TICK_SIZE = 12

###
#   Plots for UQInputs
###
@recipe function _plot(
    x::RandomVariable{T}; cdf_on=DEFAULT_CDF
) where {T<:UnivariateDistribution}
    grid --> DEFAULT_GRID
    legend --> DEFAULT_LEGEND
    label --> String(x.name)
    (cdf_on ? ylabel --> "cdf" : ylabel --> "pdf")
    seriescolor --> :auto

    lo = quantile(x, 0.001)
    hi = quantile(x, 0.999)
    w = hi - lo
    lo -= abs(w * DEFAULT_PLOT_RANGE_EXTEND_DENSITY)
    hi += abs(w * DEFAULT_PLOT_RANGE_EXTEND_DENSITY)

    xs = range(lo, hi, DEFAULT_PLOT_GRID_NUMBER)
    ys = cdf_on ? cdf.(Ref(x), xs) : pdf.(Ref(x), xs)

    # Primary line: either cdf or pdf
    @series begin
        seriestype := :path
        fill := nothing
        alpha --> 1
        linewidth --> DEFAULT_DISTRIBUTION_WIDTH
        label --> String(x.name)
        xs, ys
    end

    # Optional fill for PDF only: reuse color, don't advance palette
    if !cdf_on
        @series begin
            primary := false          # <-- reuse the color from the primary line
            seriestype := :path
            fillrange := 0              # fill down to baseline
            fillcolor := :match
            fillalpha --> DEFAULT_ALPHA
            linewidth --> 0             # fill only; no extra line
            label := ""
            xs, ys
        end
    end
end

@recipe function _plot(x::IntervalVariable)
    # --- plot-level defaults (soft) ---
    grid --> DEFAULT_GRID
    legend --> DEFAULT_LEGEND
    ylabel --> "cdf"
    label --> String(x.name)             # single legend entry

    seriescolor --> :auto

    lo_grid = x.lb
    hi_grid = x.ub

    width = hi_grid - lo_grid

    plot_lo = lo_grid - abs(width * DEFAULT_PLOT_RANGE_INTERVAL)
    plot_hi = hi_grid + abs(width * DEFAULT_PLOT_RANGE_INTERVAL)

    xlims := (plot_lo, plot_hi)

    x_grid = range(lo_grid, hi_grid, DEFAULT_PLOT_GRID_NUMBER)

    cdf_lo = x_grid .>= x.ub
    cdf_hi = x_grid .> x.lb

    # Plot upper cdf (primary, inherits colour and label)
    @series begin
        seriestype := :path
        alpha --> 1
        linewidth --> DEFAULT_DISTRIBUTION_WIDTH
        x_grid, cdf_hi
    end

    # Plot lower cdf
    @series begin
        primary := false
        seriestype := :path
        alpha --> 1
        label := ""
        linewidth --> DEFAULT_DISTRIBUTION_WIDTH
        x_grid, cdf_lo
    end

    # Plot fill
    @series begin
        primary := false
        seriestype := :path
        fillcolor := :match              # match this series' line color
        fillrange := cdf_hi
        color := DEFAULT_FILL
        fillalpha := DEFAULT_ALPHA
        linewidth --> 0                  # draw fill only here
        label := ""
        x_grid, cdf_lo
    end
end

@recipe function _plot(x::RandomVariable{T}) where {T<:ProbabilityBox}
    # --- plot-level defaults (soft) ---
    grid --> DEFAULT_GRID
    legend --> DEFAULT_LEGEND
    ylabel --> "cdf"
    label --> String(x.name)             # single legend entry

    seriescolor --> :auto

    lo_grid = quantile(x, 0.001).lb
    hi_grid = quantile(x, 0.999).ub
    width = hi_grid - lo_grid
    lo_grid = lo_grid - abs(width * DEFAULT_PLOT_RANGE_EXTEND)
    hi_grid = hi_grid + abs(width * DEFAULT_PLOT_RANGE_EXTEND)

    x_grid = range(lo_grid, hi_grid, DEFAULT_PLOT_GRID_NUMBER)
    cdf_evals = cdf.(Ref(x), x_grid)

    # Plot upper cdf (primary, inherits colour and label)
    @series begin
        seriestype := :path
        alpha --> 1
        linewidth --> DEFAULT_DISTRIBUTION_WIDTH
        x_grid, hi.(cdf_evals)
    end

    # Plot lower cdf
    @series begin
        primary := false
        seriestype := :path
        alpha --> 1
        linewidth --> DEFAULT_DISTRIBUTION_WIDTH
        label := ""
        x_grid, lo.(cdf_evals)
    end

    # Plot fill
    @series begin
        primary := false
        seriestype := :path
        fillcolor := :match              # match this series' line color
        fillrange := hi.(cdf_evals)
        fillalpha --> DEFAULT_ALPHA
        linewidth --> 0                  # draw fill only here
        label := ""
        x_grid, lo.(cdf_evals)
    end
end

using RecipesBase

@recipe function _plot(x::Vector{T}) where {T<:UQInput}
    # Filter out Parameter objects
    x_no_params = filter(xi -> !isa(xi, Parameter), x)

    N = length(x_no_params)
    cols = ceil(Int, sqrt(N))
    rows = ceil(Int, N / cols)
    layout := (rows, cols)

    # Choose a grid palette once (users can still override via plot(...; palette=...))
    # palette --> :default  # or :default, :Dark2_8, etc.

    for i in 1:N
        @series begin
            subplot := i
            seriescolor --> i  # <-- panel i uses the i-th color from the current palette
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
@recipe function _plot(xx::Vector{T}, yy::Vector{T}) where {T<:Interval}
    seriesalpha --> DEFAULT_ALPHA
    seriestype := :shape

    label := false

    linecolor := :black                         # Explicitly set edge color
    linewidth --> DEFAULT_INTERVAL_WIDTH        # Make edges more visible
    linealpha --> DEFAULT_INTERVAL_EDGE_ALPHA

    xs = Float64[]
    ys = Float64[]

    # build up coords:  # (alternative: use @series)
    for i in 1:length(xx)
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

        is = range(0, 1, length=N_samples)

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
