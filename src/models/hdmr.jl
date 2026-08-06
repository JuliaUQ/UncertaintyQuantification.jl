# High Dimensional Model Representation - Generic Implementation

struct HDMRRepresentation
    anchor::Vector{Float64}
    f0::Float64
    coefficients::Dict{Vector{Int}, Vector{Float64}}
    degree::Int
    inputs::Vector{RandomVariable}
    output::Symbol
    max_order::Int
end

function (hdmr::HDMRRepresentation)(h::Vector{Float64})
    result = hdmr.f0
    
    for (indices, coeffs) in hdmr.coefficients
        args = [h[idx] for idx in indices]
        result += _evaluate_polynomial_from_coeffs(coeffs, args, hdmr.degree)
    end
    
    return result
end

function cut_hdmr(model::Model, inputs::Vector{RandomVariable}, output::Symbol, anchor::DataFrame; order::Int=2, degree::Int=3, samples::Int=50)
    varnames = names(anchor)
    d = length(varnames)
    anchor_vec = [anchor[1, name] for name in varnames]
    anchor_copy = deepcopy(anchor)

    evaluate!(model, anchor_copy)
    f0 = anchor_copy[1, output]
    
    xs = range(0.01, 0.99; length=samples)
    
    all_points = [
        begin
            eval_point = copy(anchor_vec)
            for (i, var_idx) in enumerate(indices)
                eval_point[var_idx] = point_values[i]
            end
            eval_point
        end
        for current_order in 1:order
        for indices in _combinations(1:d, current_order)
        for point_values in Iterators.product([xs for _ in 1:current_order]...)
    ]
    
    unique_points_vec = unique(all_points)
    
    eval_df = DataFrame(
        [varname => [point[i] for point in unique_points_vec] for (i, varname) in enumerate(varnames)]
    )
    
    to_standard_normal_space!(inputs, eval_df)
    to_physical_space!(inputs, eval_df)
    evaluate!(model, eval_df)
    
    all_evaluation_points = Dict{Vector{Float64}, Float64}()
    for (i, point) in enumerate(unique_points_vec)
        all_evaluation_points[point] = eval_df[i, output]
    end
    
    hdmr_coefficients = Dict{Vector{Int}, Vector{Float64}}()
    
    for current_order in 1:order
        combinations = _combinations(1:d, current_order)
        
        for indices in combinations
            sample_points = Vector{Vector{Float64}}()
            
            for point_values in Iterators.product([xs for _ in 1:current_order]...)
                push!(sample_points, collect(point_values))
            end
            
            y_values = zeros(length(sample_points))
            
            for (k, point_values) in enumerate(sample_points)
                eval_point = copy(anchor_vec)
                for (i, var_idx) in enumerate(indices)
                    eval_point[var_idx] = point_values[i]
                end
                
                model_value = all_evaluation_points[eval_point]
                
                lower_contribution = _compute_lower_order_contribution(hdmr_coefficients, indices, f0, point_values, degree)
                # lower_contribution = _compute_lower_order_contribution(hdmr_coefficients, indices, point_values, f0, degree)
                y_values[k] = model_value - lower_contribution
            end
            
            X = _build_polynomial_basis(sample_points, degree, current_order)
            coefficients = X \ y_values
            
            hdmr_coefficients[indices] = coefficients
        end
    end
    
    return HDMRRepresentation(anchor_vec, f0, hdmr_coefficients, degree, inputs, output, order)
end

function _compute_lower_order_contribution(components_dict::Dict{Vector{Int}, Vector{Float64}}, 
                                         indices::Vector{Int}, 
                                         base_value::Float64,
                                         args::Vector{Float64},
                                         degree::Int)
    contribution = base_value
    
    for subset_size in 1:length(indices)-1
        for subset in _combinations(1:length(indices), subset_size)
            actual_indices = sort([indices[i] for i in subset])
            
            if haskey(components_dict, actual_indices)
                subset_args = [args[i] for i in subset]
                contribution += _evaluate_polynomial_from_coeffs(components_dict[actual_indices], subset_args, degree)
            end
        end
    end
    
    return contribution
end

function _evaluate_polynomial_from_coeffs(coeffs::Vector{Float64}, args::Vector{Float64}, degree::Int)
    num_vars = length(args)
    result = 0.0
    coeff_idx = 1
    
    for powers in Iterators.product([0:degree for _ in 1:num_vars]...)
        term_value = coeffs[coeff_idx]
        for (var_idx, power) in enumerate(powers)
            term_value *= args[var_idx]^power
        end
        result += term_value
        coeff_idx += 1
    end
    
    return result
end

function _build_polynomial_basis(sample_points::Vector{Vector{Float64}}, degree::Int, num_vars::Int)
    n_samples = length(sample_points)
    n_coeffs = (degree + 1)^num_vars
    X = zeros(n_samples, n_coeffs)
    
    for (sample_idx, point) in enumerate(sample_points)
        coeff_idx = 1
        
        for powers in Iterators.product([0:degree for _ in 1:num_vars]...)
            term_value = 1.0
            for (var_idx, power) in enumerate(powers)
                term_value *= point[var_idx]^power
            end
            X[sample_idx, coeff_idx] = term_value
            coeff_idx += 1
        end
    end
    
    return X
end

function _combinations(arr, k)
    n = length(arr)
    T = eltype(arr)
    k < 0 || k > n && return Vector{Vector{T}}()
    k == 0 && return [T[]]
    k == n && return [collect(arr)]

    result = Vector{Vector{T}}()
    combo = Vector{T}(undef, k)
    function _comb(start, depth)
        if depth > k
            push!(result, copy(combo))
            return
        end
        for i in start:(n - k + depth)
            combo[depth] = arr[i]
            _comb(i + 1, depth + 1)
        end
    end
    _comb(1, 1)
    return result
end