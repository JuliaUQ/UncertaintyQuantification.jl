# Map samples from standard normal space to physical space: each value becomes the quantile interval of the p-box at the corresponding u, as for ProbabilityBox
function to_physical_space!(rv::RandomVariable{<:DistributionFreeProbabilityBox}, x::DataFrame)
    x[!, rv.name] = quantile.(rv.dist, cdf.(Normal(), collect(x[:, rv.name])))
    return nothing
end
