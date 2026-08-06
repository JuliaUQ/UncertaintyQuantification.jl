# NISS (Non-Intrusive Sensitivity Analysis)

abstract type NISS end


struct LEMCSCutHDMR <: NISS
    anchor::Dict{Symbol, Vector{Float64}}
    inputs::Vector{<:RandomVariable{<:ProbabilityBox}}
    output::Symbol
    order::Int
    samples::DataFrame
    sensitivity_indices::Dict{Symbol, Tuple{Float64, Float64}}
end

function LEMCSCutHDMR(
    anchor::Dict{Symbol, Vector{Float64}}, 
    inputs::Vector{<:RandomVariable{<:ProbabilityBox}}, 
    output::Symbol, 
    order::Int, 
    samples::DataFrame
)
    return LEMCSCutHDMR(anchor, inputs, output, order, samples, Dict{Symbol, Tuple{Float64, Float64}}())
end


struct GEMCS_RS_HDMR <: NISS
    anchor::Vector{Dict{Symbol, Vector{Float64}}}
    inputs::Vector{<:RandomVariable{<:ProbabilityBox}}
    output::Symbol
    order::Int
    samples::DataFrame
    sensitivity_indices::Dict{Symbol, Tuple{Float64, Float64}}
end

function GEMCS_RS_HDMR(
    anchor::Vector{Dict{Symbol, Vector{Float64}}}, 
    inputs::Vector{<:RandomVariable{<:ProbabilityBox}}, 
    output::Symbol, 
    order::Int, 
    samples::DataFrame
)
    return GEMCS_RS_HDMR(anchor, inputs, output, order, samples, Dict{Symbol, Tuple{Float64, Float64}}())
end

function (niss::NISS)(θ::Dict{Symbol, Vector{Float64}}; detailed::Bool=false)
    varnames = [pbox.name for pbox in niss.inputs]
    total_params = sum(length(pbox.dist.parameters) for pbox in niss.inputs)
    sample_matrix = Matrix(niss.samples[:, varnames])
    y_values = niss.samples[:, niss.output]
    N = size(sample_matrix, 1)

    hdmr_components, hdmr_variances = Dict{Vector{Int}, Float64}(), Dict{Vector{Int}, Float64}()
    hdmr_comp_second_moment, hdmr_comp_second_moment_variances = Dict{Vector{Int}, Float64}(), Dict{Vector{Int}, Float64}()

    hdmr_components[Int[]], hdmr_variances[Int[]] = mean(y_values), var(y_values) / N  # TODO recheck this later especially for variance
    hdmr_comp_second_moment[Int[]], hdmr_comp_second_moment_variances[Int[]] = mean(y_values.^2), var(y_values.^2) / N # same here
    
    for current_order in 1:niss.order
        combinations = _combinations(1:total_params, current_order)
        
        for indices in combinations
            hdmr_components[indices], hdmr_variances[indices], hdmr_comp_second_moment[indices], hdmr_comp_second_moment_variances[indices] = _compute_combination_component(
                sample_matrix, y_values, niss.inputs, indices, niss.anchor, θ)
        end
    end
    
    if detailed == false
        return sum(values(hdmr_components)), sum(values(hdmr_variances))
    end
    return hdmr_components, hdmr_variances, hdmr_comp_second_moment, hdmr_comp_second_moment_variances
end

function lemcs_cut_hdmr(
    model::Model, 
    inputs::Vector{<:RandomVariable{<:ProbabilityBox}},
    output::Symbol, 
    anchor::Dict{Symbol, Vector{Float64}}; 
    order::Int=2, 
    N::AbstractMonteCarlo = MonteCarlo(1000)
)
    anchor_inputs = [RandomVariable(map_to_precise(anchor[pbox.name], pbox.dist), pbox.name) for pbox in inputs]

    samples = sample(anchor_inputs, N)
    
    evaluate!(model, samples)

    lemcs_cut_hdmr = LEMCSCutHDMR(anchor, inputs, output, order, samples)
    sensitivity_indices = _compute_sensitivity_indices(lemcs_cut_hdmr)

    return LEMCSCutHDMR(anchor, inputs, output, order, samples, sensitivity_indices)
end

function gemcs_rs_hdmr(
    model::Model, 
    inputs::Vector{<:RandomVariable{<:ProbabilityBox}},
    output::Symbol;
    order::Int=2, 
    N::Int=5000  # TODO maybe not pass as Int ? not sure but currently not consistant with lemcs_cut_hdmr()
)
    anchor = _get_gemcs_anchorpoints(inputs, N)

    anchor_samples_list = map(anchor_point -> begin
        anchor_inputs = [RandomVariable(map_to_precise(anchor_point[pbox.name], pbox.dist), pbox.name) for pbox in inputs]
        sample(anchor_inputs, 1)
    end, anchor)
    samples = vcat(anchor_samples_list...)
    evaluate!(model, samples)

    gemcs_rs_hdmr = GEMCS_RS_HDMR(anchor, inputs, output, order, samples)
    sensitivity_indices = _compute_sensitivity_indices(gemcs_rs_hdmr)

    return GEMCS_RS_HDMR(anchor, inputs, output, order, samples, sensitivity_indices)
end

function _get_gemcs_anchorpoints(inputs::Vector{<:RandomVariable{<:ProbabilityBox}}, N::Int)
    total_params = sum(length(pbox.dist.parameters) for pbox in inputs)
    param_bounds = [(lb, ub) for pbox in inputs for (lb, ub) in zip(bounds(pbox.dist)...)]

    lhs = [shuffle(collect(0:N-1)) for _ in 1:total_params]
    lhs_points = [(lhs[j][i] + 0.5) / N for i in 1:N, j in 1:total_params]
    
    anchor = Vector{Dict{Symbol, Vector{Float64}}}(undef, N)
    for i in 1:N
        anchor_point = Dict{Symbol, Vector{Float64}}()
        idx = 1
        for pbox in inputs
            nparams = length(pbox.dist.parameters)
            vals = [param_bounds[idx+j-1][1] + lhs_points[i, idx+j-1] * (param_bounds[idx+j-1][2] - param_bounds[idx+j-1][1]) for j in 1:nparams]
            anchor_point[pbox.name] = vals
            idx += nparams
        end
        anchor[i] = anchor_point
    end
    return anchor
end

function _compute_combination_component(
    sample_matrix::Matrix{Float64}, 
    y_values::Vector{Float64},
    inputs::Vector{<:RandomVariable{<:ProbabilityBox}}, 
    combination_indices::Vector{Int}, 
    anchorpoint::T,
    θ::Dict{Symbol, Vector{Float64}},
    denominator_pdfs::Vector{Float64}=Vector{Float64}(),
    reference_pdfs::Matrix{Float64}=Matrix{Float64}(undef, 0, 0),
    input_params::Vector{Vector{Int}}=Vector{Vector{Int}}()
) where {T <: Union{Dict{Symbol, Vector{Float64}}, Vector{Dict{Symbol, Vector{Float64}}}}}
    N, current_order = size(sample_matrix, 1), length(combination_indices)
    if isempty(denominator_pdfs)
        denominator_pdfs = Vector{Float64}(undef, N)
        for k in 1:N
            anchor_ref = _get_anchor_reference(anchorpoint, k)
            denominator_pdfs[k] = _compute_joint_pdf(view(sample_matrix, k, :), anchor_ref, inputs)
        end
    end
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    for k in 1:N
        anchor_ref = _get_anchor_reference(anchorpoint, k)
        r = _compute_importance_ratio(view(sample_matrix, k, :), combination_indices, anchor_ref, θ, inputs, current_order, denominator_pdfs[k], reference_pdfs, input_params, k)
        y = y_values[k]
        wy = y * r
        wy2 = y^2 * r
        s1 += wy
        s2 += wy^2
        s3 += wy2
        s4 += wy2^2
    end

    mean_y = s1 / N
    second_moment_y = s3 / N
    variance_y = (s2 - N * mean_y^2) / (N * (N - 1))
    second_moment_variance_y = (s4 - N * second_moment_y^2) / (N * (N - 1))

    return mean_y, variance_y, second_moment_y, second_moment_variance_y
end

function _compute_importance_ratio(
    sample_point::AbstractVector{<:Real}, 
    combination_indices::Vector{Int}, 
    reference_point::Dict{Symbol, Vector{Float64}}, 
    θ::Dict{Symbol, Vector{Float64}}, 
    inputs::Vector{<:RandomVariable{<:ProbabilityBox}},
    order::Int,
    denominator_pdf::Float64=NaN,
    reference_pdfs::Matrix{Float64}=Matrix{Float64}(undef, 0, 0),
    input_params::Vector{Vector{Int}}=Vector{Vector{Int}}(),
    sample_idx::Int=0
)
    if order == 1
        return _compute_conditional_pdf_ratio(sample_point, combination_indices, reference_point, θ, inputs, denominator_pdf, reference_pdfs, input_params, sample_idx) - 1.0
    elseif order == 2
        c1 = _compute_conditional_pdf_ratio(sample_point, [combination_indices[1]], reference_point, θ, inputs, denominator_pdf, reference_pdfs, input_params, sample_idx) - 1.0
        c2 = _compute_conditional_pdf_ratio(sample_point, [combination_indices[2]], reference_point, θ, inputs, denominator_pdf, reference_pdfs, input_params, sample_idx) - 1.0
        c12 = _compute_conditional_pdf_ratio(sample_point, combination_indices, reference_point, θ, inputs, denominator_pdf, reference_pdfs, input_params, sample_idx) - 1.0
        return c12 - c1 - c2
    end

    rcut = Dict{Vector{Int}, Float64}()

    for k in 1:order
        for combo in _combinations(1:order, k)
            subset_combination_indices = [combination_indices[i] for i in combo]
            expr = _compute_conditional_pdf_ratio(sample_point, subset_combination_indices, reference_point, θ, inputs, denominator_pdf, reference_pdfs, input_params, sample_idx)

            for j in 1:k-1
                for subcombo in _combinations(combo, j)
                    expr -= rcut[collect(subcombo)]
                end
            end
        rcut[combo] = expr - 1
        end
    end
    
    return rcut[_combinations(1:order, order)[1]]
end

function _compute_conditional_pdf_ratio(
    sample_point::AbstractVector{<:Real}, 
    param_indices::Vector{Int}, 
    reference_point::Dict{Symbol, Vector{Float64}}, 
    θ::Dict{Symbol, Vector{Float64}}, 
    inputs::Vector{<:RandomVariable{<:ProbabilityBox}},
    denominator_pdf::Float64=NaN,
    reference_pdfs::Matrix{Float64}=Matrix{Float64}(undef, 0, 0),
    input_params::Vector{Vector{Int}}=Vector{Vector{Int}}(),
    sample_idx::Int=0
)
    use_ref_pdfs = !isempty(reference_pdfs) && sample_idx > 0

    if use_ref_pdfs
        numerator_pdf = 1.0
        for i in eachindex(inputs)
            pbox = inputs[i]
            input_modified = !isdisjoint(param_indices, input_params[i])
            if input_modified
                theta_vals = θ[pbox.name]
                all_modified = issubset(input_params[i], param_indices)
                if all_modified
                    precise_rv = map_to_precise(theta_vals, pbox.dist)
                    numerator_pdf *= pdf(precise_rv, sample_point[i])
                else
                    ref_vals = reference_point[pbox.name]
                    nparams = length(pbox.dist.parameters)
                    merged_vals = Vector{Float64}(undef, nparams)
                    for j in 1:nparams
                        if input_params[i][j] in param_indices
                            merged_vals[j] = theta_vals[j]
                        else
                            merged_vals[j] = ref_vals[j]
                        end
                    end
                    precise_rv = map_to_precise(merged_vals, pbox.dist)
                    numerator_pdf *= pdf(precise_rv, sample_point[i])
                end
            else
                numerator_pdf *= reference_pdfs[sample_idx, i]
            end
        end
    else
        modified_params = Dict{Symbol, Vector{Float64}}()
        global_param_idx = 1
        for pbox in inputs
            nparams = length(pbox.dist.parameters)
            vals = Vector{Float64}(undef, nparams)
            ref_vals = reference_point[pbox.name]
            theta_vals = θ[pbox.name]
            for local_param_idx in 1:nparams
                if global_param_idx in param_indices
                    vals[local_param_idx] = theta_vals[local_param_idx]
                else
                    vals[local_param_idx] = ref_vals[local_param_idx]
                end
                global_param_idx += 1
            end
            modified_params[pbox.name] = vals
        end
        numerator_pdf = _compute_joint_pdf(sample_point, modified_params, inputs)
    end

    if isnan(denominator_pdf)
        denominator_pdf = _compute_joint_pdf(sample_point, reference_point, inputs)
    end
    
    return denominator_pdf ≈ 0.0 ? 1.0 : numerator_pdf / denominator_pdf
end

function _compute_joint_pdf(
    sample_point::AbstractVector{<:Real}, 
    parameter_values::Dict{Symbol, Vector{Float64}}, 
    inputs::Vector{<:RandomVariable{<:ProbabilityBox}}
)
    result = 1.0
    for i in eachindex(inputs)
        pbox = inputs[i]
        precise_rv = map_to_precise(parameter_values[pbox.name], pbox.dist)
        result *= pdf(precise_rv, sample_point[i])
    end
    return result
end

function _compute_sensitivity_indices(niss::NISS)
    total_params = sum(length(pbox.dist.parameters) for pbox in niss.inputs)
    varnames = [pbox.name for pbox in niss.inputs]
    sample_matrix = Matrix(niss.samples[:, varnames])
    y_values = niss.samples[:, niss.output]

    param_bounds, param_names = _get_param_bounds_and_names(niss.inputs)

    N = length(y_values)
    n_inputs = length(niss.inputs)
    denominator_pdfs = Vector{Float64}(undef, N)
    reference_pdfs = Matrix{Float64}(undef, N, n_inputs)
    input_params = Vector{Vector{Int}}(undef, n_inputs)
    param_mapping = Vector{Tuple{Int, Int}}(undef, total_params)
    global_idx = 1
    for (i, pbox) in enumerate(niss.inputs)
        nparams = length(pbox.dist.parameters)
        input_params[i] = collect(global_idx:(global_idx + nparams - 1))
        for j in 1:nparams
            param_mapping[global_idx] = (i, j)
            global_idx += 1
        end
    end
    for k in 1:N
        anchor_ref = _get_anchor_reference(niss.anchor, k)
        denominator_pdfs[k] = _compute_joint_pdf(view(sample_matrix, k, :), anchor_ref, niss.inputs)
        for i in 1:n_inputs
            pbox = niss.inputs[i]
            precise_rv = map_to_precise(anchor_ref[pbox.name], pbox.dist)
            reference_pdfs[k, i] = pdf(precise_rv, sample_matrix[k, i])
        end
    end

    function get_component_function(indices::Vector{Int})
        return function(θ_dict::Dict{Symbol, Vector{Float64}})
            estimate, _, second_moment, _ = _compute_combination_component(
                sample_matrix, y_values, niss.inputs, indices, niss.anchor, θ_dict, denominator_pdfs, reference_pdfs, input_params
            )
            return estimate, second_moment
        end
    end

    component_vars = Dict{Vector{Int}, Float64}()
    component_vars_second_moment = Dict{Vector{Int}, Float64}()

    for order in 1:niss.order
        for indices in _combinations(1:total_params, order)
            lowers = [param_bounds[i][1] for i in indices]
            uppers = [param_bounds[i][2] for i in indices]
            component_func = get_component_function(indices)
            θ_template = _get_anchor_template(niss.anchor)
            modified_input_names = Set{Symbol}()
            for idx in indices
                input_idx, _ = param_mapping[idx]
                push!(modified_input_names, niss.inputs[input_idx].name)
            end
            θ_dict = Dict{Symbol, Vector{Float64}}()
            for pbox in niss.inputs
                θ_dict[pbox.name] = copy(θ_template[pbox.name])
            end
            function f(x)
                for (m, idx) in enumerate(indices)
                    input_idx, local_idx = param_mapping[idx]
                    θ_dict[niss.inputs[input_idx].name][local_idx] = x[m]
                end
                estimate, second_moment = component_func(θ_dict)
                return [estimate, estimate^2, second_moment, second_moment^2]
            end
            (integral, err) = hcubature(f, lowers, uppers; rtol=order==1 ? 1e-3 : 1e-2, maxevals=order==1 ? 10000 : 20000)
            volume = prod(uppers[i] - lowers[i] for i in 1:length(lowers))
            mean, mean_sq, mean_second, mean_second_sq = integral ./ volume
            component_vars[collect(indices)] = mean_sq - mean^2
            component_vars_second_moment[collect(indices)] = mean_second_sq - mean_second^2
        end
    end

    return _assign_sensitivity_indices(component_vars, component_vars_second_moment, param_names)
end

function _get_param_bounds_and_names(inputs::Vector{<:RandomVariable{<:ProbabilityBox}})
    param_bounds = Dict{Int, Tuple{Float64, Float64}}()
    param_names = String[]
    count = 0
    for pbox in inputs
        lbs, ubs = bounds(pbox.dist)
        for (i, param) in enumerate(pbox.dist.parameters)
            count += 1
            param_bounds[count] = (lbs[i], ubs[i])
            push!(param_names, "$(pbox.name)_$(param.first)")
        end
    end
    return param_bounds, param_names
end

function _get_anchor_template(anchor)
    if isa(anchor, Vector)
        return anchor[1]
    else
        return anchor
    end
end

@inline _get_anchor_reference(anchor::Dict{Symbol, Vector{Float64}}, k::Int) = anchor
@inline _get_anchor_reference(anchor::Vector{Dict{Symbol, Vector{Float64}}}, k::Int) = anchor[k]

function _copy_theta_template(template::Dict{Symbol, Vector{Float64}})
    return Dict{Symbol, Vector{Float64}}(k => copy(v) for (k, v) in template)
end

function _set_parameters!(θ_dict::Dict{Symbol, Vector{Float64}}, inputs::Vector{<:RandomVariable{<:ProbabilityBox}}, indices::Vector{Int}, values::AbstractVector)
    param_idx = 0
    for pbox in inputs
        for j in 1:length(pbox.dist.parameters)
            param_idx += 1
            for (k, idx) in enumerate(indices)
                if param_idx == idx
                    θ_dict[pbox.name][j] = values[k]
                end
            end
        end
    end
end

function _assign_sensitivity_indices(comp_vars::Dict{Vector{Int}, Float64}, comp_vars_second_mom::Dict{Vector{Int}, Float64}, param_names::Vector{String})
    total_var = sum(values(comp_vars))
    total_var_second_moment = sum(values(comp_vars_second_mom))
    sensitivity_indices = Dict{Symbol, Tuple{Float64, Float64}}()
    for (indices, comp_sensitivity) in comp_vars
        if length(indices) == 1
            param_name = Symbol(param_names[indices[1]])
        else
            names_combined = [param_names[i] for i in indices]
            param_name = Symbol(join(names_combined, "_"))
        end
        se_sensitivity_index = total_var > 0 ? comp_sensitivity / total_var : 0.0
        sv_sensitivity_index = total_var_second_moment > 0 ? comp_vars_second_mom[indices] / total_var_second_moment : 0.0
        sensitivity_indices[param_name] = (se_sensitivity_index, sv_sensitivity_index)
    end
    return sensitivity_indices
end

# Old version without hcubature kept for reference

# function _compute_sensitivity_indices(lemcs::LEMCSCutHDMR; Nt::Int=50) # TODO clean up this whole mess
#     total_params = sum(length(p.parameters) for p in lemcs.inputs)
#     varnames = [pbox.name for pbox in lemcs.inputs]
#     sample_matrix = Matrix(lemcs.samples[:, varnames])
#     y_values = lemcs.samples[:, lemcs.output]

#     param_ranges = Dict{Int, Vector{Float64}}()
#     param_names = String[]
    
#     count = 0
#     for (var_idx, pbox) in enumerate(lemcs.inputs)
#         for (param_idx, param) in enumerate(pbox.parameters)
#             count += 1
            
#             interval = pbox.parameters[param_idx]
#             param_min = interval.lb
#             param_max = interval.ub
            
#             param_ranges[count] = range(param_min, param_max, length=Nt)
#             push!(param_names, "$(pbox.name)_$(param.name)")
#         end
#     end

#     function get_component_function(indices::Vector{Int})
#         return function(θ_dict::Dict{Symbol, Vector{Float64}})
#             estimate, _, second_moment, _ = _compute_combination_component(
#                 sample_matrix, y_values, lemcs.inputs, indices, lemcs.anchor, θ_dict
#             )
#             return estimate, second_moment
#         end
#     end

#     component_vars = Dict{Vector{Int}, Float64}()
#     component_vars_second_moment = Dict{Vector{Int}, Float64}()

#     for i in 1:total_params
#         component_func = get_component_function([i])
#         vals = param_ranges[i]
        
#         expectation_values = zeros(Nt)
#         second_moment_values = zeros(Nt)
        
#         for k in 1:Nt
#             θ_dict = deepcopy(lemcs.anchor)

#             param_idx = 0
#             for pbox in lemcs.inputs
#                 for j in 1:length(pbox.parameters)
#                     param_idx += 1
#                     if param_idx == i
#                         θ_dict[pbox.name][j] = vals[k]
#                         break
#                     end
#                 end
#             end
            
#             estimate, second_moment = component_func(θ_dict)
#             expectation_values[k] = estimate
#             second_moment_values[k] = second_moment
#         end
        
#         component_vars[[i]] = var(expectation_values)
#         component_vars_second_moment[[i]] = var(second_moment_values)
#     end

#     if lemcs.order >= 2
#         for i in 1:total_params
#             for j in i+1:total_params
#                 component_func = get_component_function(sort([i,j]))
#                 vals_i = param_ranges[i]
#                 vals_j = param_ranges[j]
                
#                 coarse_Nt = min(Nt, 20)
#                 i_indices = round.(Int, range(1, Nt, length=coarse_Nt))
#                 j_indices = round.(Int, range(1, Nt, length=coarse_Nt))
                
#                 expectation_values = zeros(coarse_Nt, coarse_Nt)
#                 second_moment_values = zeros(coarse_Nt, coarse_Nt)
                
#                 for (ki, k) in enumerate(i_indices)
#                     for (lj, l) in enumerate(j_indices)
#                         θ_dict = deepcopy(lemcs.anchor)
#                         param_idx = 0
#                         for pbox in lemcs.inputs
#                             for m in 1:length(pbox.parameters)
#                                 param_idx += 1
#                                 if param_idx == i
#                                     θ_dict[pbox.name][m] = vals_i[k]
#                                 elseif param_idx == j
#                                     θ_dict[pbox.name][m] = vals_j[l]
#                                 end
#                             end
#                         end
#                         estimate, second_moment = component_func(θ_dict)
#                         expectation_values[ki, lj] = estimate
#                         second_moment_values[ki, lj] = second_moment
#                     end
#                 end
                
#                 component_vars[sort([i,j])] = var(vec(expectation_values))
#                 component_vars_second_moment[sort([i,j])] = var(vec(second_moment_values))
#             end
#         end
#     end

#     total_var = sum(values(component_vars))
#     total_var_second_moment = sum(values(component_vars_second_moment))
    
#     sensitivity_indices = Dict{Symbol, Tuple{Float64, Float64}}()
    
#     for (indices, variance) in component_vars
#         if length(indices) == 1
#             param_name = Symbol(param_names[indices[1]])
#         else
#             names_combined = [param_names[i] for i in indices]
#             param_name = Symbol(join(names_combined, "_"))
#         end

#         se_sensitivity_index = total_var > 0 ? variance / total_var : 0.0
#         sv_sensitivity_index = total_var_second_moment > 0 ? component_vars_second_moment[indices] / total_var_second_moment : 0.0
#         sensitivity_indices[param_name] = (se_sensitivity_index, sv_sensitivity_index)
#     end

#     return sensitivity_indices
# end

# function _compute_sensitivity_indices(gemcs::GEMCS_RS_HDMR) # TODO clean up this whole function
#     total_params = sum(length(p.parameters) for p in gemcs.inputs)
#     varnames = [pbox.name for pbox in gemcs.inputs]
#     sample_matrix = Matrix(gemcs.samples[:, varnames])
#     y_values = gemcs.samples[:, gemcs.output]
#     N = size(sample_matrix, 1)

#     param_names = String[]
#     for (pbox) in gemcs.inputs
#         for (param) in pbox.parameters
#             push!(param_names, "$(pbox.name)_$(param.name)")
#         end
#     end

#     mean_y, var_y = Vector{Dict{Vector{Int}, Float64}}(undef, N), Vector{Dict{Vector{Int}, Float64}}(undef, N)
#     sec_moment_y, var_sec_moment_y = Vector{Dict{Vector{Int}, Float64}}(undef, N), Vector{Dict{Vector{Int}, Float64}}(undef, N)

#     for k in 1:N
#         mean_y[k], var_y[k], sec_moment_y[k], var_sec_moment_y[k] = gemcs(gemcs.anchor[k]; detailed=true)
#         (k % 100 == 0 || k == N) && println("Eval $k / $N done.")
#     end

#     sens_comp_vars, sens_comp_variance = Dict{Vector{Int}, Float64}(), Dict{Vector{Int}, Float64}()
#     sens_second_moment, sens_second_moment_variance_y = Dict{Vector{Int}, Float64}(), Dict{Vector{Int}, Float64}()

#     for i in _combinations(1:total_params, 1)
#         expectation_values = [get(mean_y[k], i, NaN) for k in 1:N]
#         variance_values = [get(var_y[k], i, NaN) for k in 1:N]
#         second_moment_values = [get(sec_moment_y[k], i, NaN) for k in 1:N]
#         var_second_moment_values = [get(var_sec_moment_y[k], i, NaN) for k in 1:N]

#         sens_comp_vars[i] = mean(expectation_values.^2)
#         sens_comp_variance[i] = var(variance_values)
#         sens_second_moment[i] = mean(second_moment_values.^2)
#         sens_second_moment_variance_y[i] = var(var_second_moment_values)
#     end

#     if gemcs.order >= 2
#         for i in _combinations(1:total_params, 2)
#             expectation_values = [get(mean_y[k], i, NaN) for k in 1:N]
#             variance_values = [get(var_y[k], i, NaN) for k in 1:N]
#             second_moment_values = [get(sec_moment_y[k], i, NaN) for k in 1:N]
#             var_second_moment_values = [get(var_sec_moment_y[k], i, NaN) for k in 1:N]

#             sens_comp_vars[i] = mean(expectation_values.^2)
#             sens_comp_variance[i] = var(variance_values)
#             sens_second_moment[i] = mean(second_moment_values.^2)
#             sens_second_moment_variance_y[i] = var(var_second_moment_values)
#         end
#     end
#     return assign_sensitivity_indices(sens_comp_vars, sens_second_moment, param_names)
# end
