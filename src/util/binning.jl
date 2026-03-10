"""
    BinnedData(grid::AbstractVector{<:Real}, weights::AbstractVector{<:Real})

 Container for one dimensional binned data constructed from grid points and associated weights.
"""
struct BinnedData
	grid::AbstractVector{<:Real}
	weights::AbstractVector{<:Real}
end

"""
    linear_binning(x::AbstractVector, nbins::Integer)

Bin the data in `x` into `nbins` regular bins between `minimum(x)` and `maximum(x)` using linear binning.

In difference to regular binning, if a data point lies between two grid points, linear binning assigns weight to both points.
The closer the grid point is to the data the more weight it is assigned. Note, that points with  a weight of zero are discarded.

Returns a `BinnedData` object.

# References

[jonesErrorsInvolvedComputing1983](@cite)
"""
function linear_binning(x::AbstractVector, nbins::Integer)
	xmin = minimum(x)
	xmax = maximum(x)

	g = collect(range(xmin, xmax; length = nbins)) # grid points
	w = zeros(eltype(x), nbins) # weights
	Δ = (xmax - xmin) / (nbins - 1)
	for xi in x
		# Find the g index on the left
		idx = Int(clamp(floor((xi - xmin) / Δ) + 1, 1, nbins))
		if idx == nbins
			# If at the last bin, assign all weight there
			w[nbins] += 1.0
		else
			left = g[idx]
			# Linear weights
			w_right = (xi - left) / Δ
			w_left = 1.0 - w_right
			w[idx] += w_left
			w[idx+1] += w_right
		end
	end

	# find all nonzero weights
	idx = findall(!iszero, w)

	if length(nbins - length(idx)) > 0
		@info "Dropped $(nbins - length(idx)) zero-weight grid points"
	end

	return BinnedData(g[idx], w[idx])
end
