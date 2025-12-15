const _kucherenko_table_types = [Symbol[], Float64[], Float64[]]
const _kucherenko_table_header = [
    "Variables", "FirstOrder", "TotalEffect"
]

"""
- `models::Vector{<:UQModel}`: Vector of UQ models to evaluate
- `inputs::Vector{<:UQInput}`: Vector of input distributions
- `output::Symbol`: Output quantity name
- `sim::AbstractMonteCarlo`: == N : Total MC samples = N*(2M+1)
"""
function kucherenkoindices(
    models::Vector{<:UQModel},
    inputs::Vector{<:UQInput},
    output::Symbol,
    sim::AbstractMonteCarlo
)
    indices = DataFrame(_kucherenko_table_types, _kucherenko_table_header)
    
    samples = sample(inputs, sim)
    evaluate!(models, samples)
    
    random_names = names(filter(i -> isa(i, RandomUQInput), inputs))
    
    Y_orig = Vector(samples[:, output])
    total_var = var(Y_orig)
    
    for i in random_names
        
        i_cond_samples = _generate_conditional_samples(samples, inputs[1], [i])
        evaluate!(models, i_cond_samples)
        S_i = _compute_first_order_kucherenko(samples, i_cond_samples, output, total_var)

        other_vars = setdiff(random_names, [i])
        other_cond_samples = _generate_conditional_samples(samples, inputs[1], other_vars)
        evaluate!(models, other_cond_samples)
        ST_i = length(random_names) == 1 ? S_i : _compute_total_effect_kucherenko(samples, other_cond_samples, output, total_var)
        
        push!(indices, [i, S_i, ST_i])
    end
    
    return indices
end


function _compute_first_order_kucherenko(
    samples::DataFrame,
    cond_samples::DataFrame,
    output::Symbol,
    total_var::Float64
)
    Y_orig = Vector(samples[:, output])
    Y_cond = Vector(cond_samples[:, output])
    
    S_i = (mean(Y_orig .* Y_cond) - mean(Y_orig)^2) / total_var  # Kucherenko et. al. 2012  Eq. 5.3
    
    return S_i
end

function _compute_total_effect_kucherenko(
    samples::DataFrame,
    cond_samples::DataFrame,
    output::Symbol,
    total_var::Float64
)
    Y_orig = Vector(samples[:, output])
    Y_cond = Vector(cond_samples[:, output])
    
    ST_i = mean((Y_orig .- Y_cond).^2) / (2 * total_var) # Kucherenko et. al. 2012 Eq. 5.4
    
    return ST_i
end

function _generate_conditional_samples(
    samples::DataFrame,
    joint_dist::JointDistribution,
    var_names::Vector{Symbol}
)
    input_var_names = [marginal.name for marginal in joint_dist.m]
    conditional_samples = map(1:nrow(samples)) do i
        x_values = [samples[i, var_name] for var_name in var_names]
        var_value_tuples = [(var_names[j], x_values[j]) for j in 1:length(var_names)]
        sample_conditional_copula(joint_dist, var_value_tuples, 1)
    end
    
    result = vcat(conditional_samples...)
    return select(result, input_var_names)
end

"""
- `models::Vector{<:UQModel}`: Vector of UQ models to evaluate
- `inputs::Vector{<:UQInput}`: Vector of input distributions
- `outputs::Vector{Symbol}`: Vector of output quantity names
- `sim::AbstractMonteCarlo`:  == N : Total MC samples = N*(2M+1)
"""
function kucherenkoindices(
    models::Vector{<:UQModel},
    inputs::Vector{<:UQInput},
    outputs::Vector{Symbol},
    sim::AbstractMonteCarlo
)
    
    random_names = names(filter(i -> isa(i, RandomUQInput), inputs))
    
    indices = Dict([
        (name, DataFrame(_kucherenko_table_types, _kucherenko_table_header)) for name in outputs
    ])
    
    for output in outputs
        
        output_indices = kucherenkoindices(models, inputs, output, sim)
        
        for (i, var_name) in enumerate(random_names)
            output_indices[i, :Variables] = var_name
        end
        
        indices[output] = output_indices
    end
    
    return length(outputs) > 1 ? indices : indices[outputs[1]]
end

function kucherenkoindices(
    models::Vector{<:UQModel},
    inputs::UQInput,
    outputs::Vector{Symbol},
    sim::AbstractMonteCarlo
)
    return kucherenkoindices(models, [inputs], outputs, sim)
end


function kucherenkoindices(
    models::UQModel,
    inputs::Vector{<:UQInput},
    outputs::Symbol,
    sim::AbstractMonteCarlo
)
    return kucherenkoindices([models], inputs, [outputs], sim)
end

function kucherenkoindices(
    models::Vector{<:UQModel},
    inputs::UQInput,
    outputs::Symbol,
    sim::AbstractMonteCarlo
)
    return kucherenkoindices(models, [inputs], [outputs], sim)
end

function kucherenkoindices(
    models::UQModel,
    inputs::UQInput,
    outputs::Symbol,
    sim::AbstractMonteCarlo
)
    return kucherenkoindices([models], [inputs], [outputs], sim)
end

function kucherenkoindices(
    models::UQModel,
    inputs::Vector{<:UQInput},
    outputs::Vector{Symbol},
    sim::AbstractMonteCarlo
)
    return kucherenkoindices([models], inputs, outputs, sim)
end



"""
- `X::Matrix`: Input sample matrix where each row is a sample point and each column is a variable
- `Y::Vector`: Output sample vector corresponding to the input samples
- `min_bin_sample::Int=25`: Minimum samples per bin for 1 dimension for conditioning; Recommended amount is at least 25 samples per bin with bin amount around 100
- `min_bin_sample_multi_dims::Int=25`: Minimum samples per bin-dimension in multiple dimensions for conditioning; Recommended amount is about 10-25 samples per bin
"""
function kucherenkoindices_bin(X::Matrix, Y::Vector; min_bin_sample=nothing, min_bin_sample_multi_dims::Int=25)
    n_samples, n_vars = size(X)

    num_bins = min_bin_sample === nothing ? min(100, floor(Int, n_samples / 25)) : floor(Int, n_samples / min_bin_sample)
    num_bins_multi = floor(Int, n_samples / min_bin_sample_multi_dims)
    
    if num_bins > 500 @warn "More than 500 bins in single dimension: $num_bins" end
    if length(Y) != n_samples throw(ArgumentError("Number of output samples must match number of input samples")) end
    
    indices = DataFrame(_kucherenko_table_types, _kucherenko_table_header)
    total_var = var(Y)
    
    for i in 1:n_vars
        S_i = _compute_first_order_kucherenko_bins(X, Y, i, num_bins, total_var)

        ST_i = n_vars == 1 ? S_i : _compute_total_effect_kucherenko_bins(X, Y, i, num_bins_multi, total_var)
        
        push!(indices, [Symbol("X$i"), S_i, ST_i])
    end
    
    return indices
end

function _compute_first_order_kucherenko_bins(X::Matrix, Y::Vector, var_idx::Int, num_bins::Int, total_var::Float64)
    n_samples = length(Y)
    x_var = X[:, var_idx]

    quantiles = range(0, 1; length=num_bins+1)
    bin_edges = quantile(x_var, quantiles)
    
    bin_means = Float64[]
    bin_weights = Float64[]
    
    for j in 1:num_bins
        if j == num_bins
            mask = (x_var .>= bin_edges[j]) .& (x_var .<= bin_edges[j+1])
        else
            mask = (x_var .>= bin_edges[j]) .& (x_var .< bin_edges[j+1])
        end
        
        n_bin = sum(mask)
        
        if n_bin > 0
            bin_mean = mean(Y[mask])
            push!(bin_means, bin_mean)
            push!(bin_weights, n_bin / n_samples)
        end
    end
    
    if isempty(bin_means)
        return 0.0
    end
    
    weighted_mean = sum(bin_weights .* bin_means)
    
    S_i = sum(bin_weights .* (bin_means .- weighted_mean).^2) / total_var
    
    return S_i
end

function _compute_total_effect_kucherenko_bins(X::Matrix, Y::Vector, var_idx::Int, num_bins::Int, total_var::Float64)
    n_samples, n_vars = size(X)
    
    other_vars = setdiff(1:n_vars, var_idx)
    X_other = X[:, other_vars]
    
    bin_assignments = _assign_multidimensional_bins(X_other, num_bins)
    
    weighted_var_sum = 0.0
    total_weight = 0.0
    
    for bin_id in unique(bin_assignments)
        mask = bin_assignments .== bin_id
        n_bin = sum(mask)
        
        if n_bin > 1
            bin_var = var(Y[mask])
            weight = n_bin / n_samples
            weighted_var_sum += weight * bin_var
            total_weight += weight
        end
    end
    
    ST_i = total_weight > 0 ? weighted_var_sum / total_var : 0.0
    
    return ST_i
end

function _assign_multidimensional_bins(X::Matrix, num_bins::Int)
    n_samples, n_dims = size(X)
    bins_per_dim = floor(Int, num_bins^(1/n_dims))

    quantile_edges = [quantile(X[:, j], range(0, 1; length=bins_per_dim+1)) for j in 1:n_dims]
    
    bin_assignments = zeros(Int, n_samples)
    
    for i in 1:n_samples
        bin_id = 0
        for j in 1:n_dims
            edges = quantile_edges[j]
            x = X[i, j]
            bin_idx = searchsortedlast(edges, x)
            bin_idx = clamp(bin_idx, 1, bins_per_dim)
            bin_id += (bin_idx - 1) * (bins_per_dim ^ (j - 1))
        end
        bin_assignments[i] = bin_id + 1
    end
    
    return bin_assignments
end

"""
- `models::Vector{<:UQModel}`: Vector of UQ models to evaluate
- `inputs::Vector{<:UQInput}`: Vector of input distributions
- `outputs::Vector{Symbol}`: Vector of output quantity names
- `sim::AbstractMonteCarlo`: Total MC samples
- `min_bin_sample::Int=25`: Minimum samples per bin for 1 dimension for conditioning; Recommended amount is at least 25 samples per bin but bin amount not higher than 100
- `min_bin_sample_multi_dims::Int=25`: Minimum samples per bin-dimension in multiple dimensions for conditioning; Recommended amount is about 10-25 samples per bin
"""
function kucherenkoindices_bin(
    models::Vector{<:UQModel},
    inputs::Vector{<:UQInput},
    outputs::Vector{Symbol},
    sim::AbstractMonteCarlo;
    min_bin_sample=nothing,
    min_bin_sample_multi_dims::Int=25
)
    samples = sample(inputs, sim)
    evaluate!(models, samples)
    
    random_names = names(filter(i -> isa(i, RandomUQInput), inputs))
    
    indices = Dict([
        (name, DataFrame(_kucherenko_table_types, _kucherenko_table_header)) for name in outputs
    ])
    
    X = Matrix(samples[:, random_names])
    
    for output in outputs
        Y = Vector(samples[:, output])
        
        output_indices = kucherenkoindices_bin(X, Y; min_bin_sample = min_bin_sample, min_bin_sample_multi_dims = min_bin_sample_multi_dims)
        
        for (i, var_name) in enumerate(random_names)
            output_indices[i, :Variables] = var_name
        end
        
        indices[output] = output_indices
    end
    
    return length(outputs) > 1 ? indices : indices[outputs[1]], samples
end

function kucherenkoindices_bin(
    models::Vector{<:UQModel},
    inputs::UQInput,
    outputs::Vector{Symbol},
    sim::AbstractMonteCarlo;
    min_bin_sample=nothing,
    min_bin_sample_multi_dims::Int=25
)
    return kucherenkoindices_bin(models, [inputs], outputs, sim; min_bin_sample=min_bin_sample, min_bin_sample_multi_dims=min_bin_sample_multi_dims)
end

function kucherenkoindices_bin(
    models::Vector{<:UQModel},
    inputs::Vector{<:UQInput},
    outputs::Symbol,
    sim::AbstractMonteCarlo;
    min_bin_sample=nothing,
    min_bin_sample_multi_dims::Int=25
)
    return kucherenkoindices_bin(models, inputs, [outputs], sim; min_bin_sample=min_bin_sample, min_bin_sample_multi_dims=min_bin_sample_multi_dims)
end

function kucherenkoindices_bin(
    models::UQModel,
    inputs::Vector{<:UQInput},
    outputs::Symbol,
    sim::AbstractMonteCarlo;
    min_bin_sample=nothing,
    min_bin_sample_multi_dims::Int=25
)
    return kucherenkoindices_bin([models], inputs, [outputs], sim; min_bin_sample=min_bin_sample, min_bin_sample_multi_dims=min_bin_sample_multi_dims)
end

function kucherenkoindices_bin(
    models::Vector{<:UQModel},
    inputs::UQInput,
    outputs::Symbol,
    sim::AbstractMonteCarlo;
    min_bin_sample=nothing,
    min_bin_sample_multi_dims::Int=25
)
    return kucherenkoindices_bin(models, [inputs], [outputs], sim; min_bin_sample=min_bin_sample, min_bin_sample_multi_dims=min_bin_sample_multi_dims)
end

function kucherenkoindices_bin(
    models::UQModel,
    inputs::UQInput,
    outputs::Symbol,
    sim::AbstractMonteCarlo;
    min_bin_sample=nothing,
    min_bin_sample_multi_dims::Int=25
)
    return kucherenkoindices_bin([models], [inputs], [outputs], sim; min_bin_sample=min_bin_sample, min_bin_sample_multi_dims=min_bin_sample_multi_dims)
end

function kucherenkoindices_bin(
    models::UQModel,
    inputs::Vector{<:UQInput},
    outputs::Vector{Symbol},
    sim::AbstractMonteCarlo;
    min_bin_sample=nothing,
    min_bin_sample_multi_dims::Int=25
)
    return kucherenkoindices_bin([models], inputs, outputs, sim; min_bin_sample=min_bin_sample, min_bin_sample_multi_dims=min_bin_sample_multi_dims)
end
