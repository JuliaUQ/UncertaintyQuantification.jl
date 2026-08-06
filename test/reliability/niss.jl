using Test
using UncertaintyQuantification
using DataFrames

# Delete these dependencies later TODO

@testset "NISS" begin
	function g_function(df::DataFrame)
		n_samples = nrow(df)
		outputs = Vector{Float64}(undef, n_samples)
		for i in 1:n_samples
			x1, x2 = df[i, :x1], df[i, :x2]
			outputs[i] = 1.0 - (x1 - 1.0)^2 / 9.0 - (x2 - 1.0)^3 / 16.0
		end
		return outputs
	end

	model = Model(g_function, :output)
	inputs = [
		RandomVariable(
			ProbabilityBox{Normal}(Dict(
				:μ => Interval(-0.2, 0.2),
				:σ => Interval(0.8, 1.2)
			)), :x1),
		RandomVariable(
			ProbabilityBox{Normal}(Dict(
				:μ => Interval(-0.2, 0.2),
				:σ => Interval(0.8, 1.2)
			)), :x2)
	]
	anchor = Dict(
		:x1 => [0.0, 1.0],
		:x2 => [0.0, 1.0]
	)

	N = LatinHypercubeSampling(5000)
	lemcs_result = lemcs_cut_hdmr(
		model,
		inputs,
		:output,
		anchor;
		order=1,
		N
	)
	gemcs_result = gemcs_rs_hdmr(
		model,
		inputs,
		:output;
		order=1,
		N=10000
	)


	@testset "LEMCSCutHDMR constructor" begin
		df = DataFrame(x1 = randn(10), x2 = randn(10), output = randn(10))
		obj = LEMCSCutHDMR(anchor, inputs, :output, 2, df)
		@test obj isa LEMCSCutHDMR
		@test obj.anchor == anchor
		@test obj.inputs == inputs
		@test obj.output == :output
		@test obj.order == 2
		@test obj.samples == df
	end

	@testset "GEMCS_RS_HDMR constructor" begin
		anchors = [anchor, anchor]
		df = DataFrame(x1 = randn(2), x2 = randn(2), output = randn(2))
		obj = GEMCS_RS_HDMR(anchors, inputs, :output, 2, df)
		@test obj isa GEMCS_RS_HDMR
		@test obj.anchor == anchors
		@test obj.inputs == inputs
		@test obj.output == :output
		@test obj.order == 2
		@test obj.samples == df
	end

	@testset "lemcs_cut_hdmr and gemcs_rs_hdmr" begin
		@test lemcs_result isa LEMCSCutHDMR
		θ = Dict(:x1 => [0.1, 1.0], :x2 => [0.0, 1.0])
		res = lemcs_result(θ; detailed=true)
		@test length(res) == 4
		@test all(x -> x isa Dict, res)

		@test gemcs_result isa GEMCS_RS_HDMR
		res2 = gemcs_result(θ; detailed=true)
		@test length(res2) == 4
		@test all(x -> x isa Dict, res2)

		anchor_inputs = [RandomVariable(UncertaintyQuantification.map_to_precise(θ[pbox.name], pbox.dist), pbox.name) for pbox in inputs]
		direct_samples = UncertaintyQuantification.sample(anchor_inputs, LatinHypercubeSampling(10000))
		evaluate!(model, direct_samples)
		direct_mean = mean(direct_samples[:, :output])
		local_niss_mean = sum(values(res[1]))
		global_niss_mean = sum(values(res2[1]))
		@test isfinite(direct_mean)
		@test isfinite(local_niss_mean)
		@test local_niss_mean ≈ direct_mean atol=0.01
		@test isfinite(global_niss_mean)
		@test global_niss_mean ≈ direct_mean atol=0.02 # more tolerance because of standard mc sampling
	end

	@testset "_get_gemcs_anchorpoints" begin
		anchors = UncertaintyQuantification._get_gemcs_anchorpoints(inputs, 5)
		@test anchors isa Vector{Dict{Symbol, Vector{Float64}}}
		@test length(anchors) == 5
		@test all(haskey.(anchors, :x1))
		@test all(haskey.(anchors, :x2))
	end

	@testset "_compute_importance_ratio (original example first term)" begin # currently stupid TODO get analytical formula for this one
		θ = Dict(:x1 => [0.1, 1.0], :x2 => [0.0, 1.0])

		anchor_inputs = [RandomVariable(UncertaintyQuantification.map_to_precise(anchor[pbox.name], pbox.dist), pbox.name) for pbox in inputs]
		samples = UncertaintyQuantification.sample(anchor_inputs, 1)
		sample_point = [samples[1, :x1], samples[1, :x2]]

		ratio = UncertaintyQuantification._compute_importance_ratio(sample_point, [1], anchor, θ, inputs, 1)
		@info "Importance ratio for first sample, first parameter" ratio sample_point anchor θ
		@test isfinite(ratio)
	end

	@testset "_compute_importance_ratio" begin
		θ_analytical = Dict(:x1 => [0.0, 1.0], :x2 => [0.0, 1.0])
		sample = [0.5, -0.5]
		ratio = UncertaintyQuantification._compute_importance_ratio(sample, [1], θ_analytical, θ_analytical, inputs, 1)
		@test isapprox(ratio, 0.0; atol=1e-10)
	end

	@testset "_compute_conditional_pdf_ratio" begin
		θ = Dict(:x1 => [0.0, 1.0], :x2 => [0.0, 1.0])
		θ_cond = Dict(:x1 => [0.2, 1.0], :x2 => [0.0, 1.0])
		sample = [0.5, 0.5]
        pdf_num = pdf(Normal(0.2,1), 0.5)
		pdf_den = pdf(Normal(0,1), 0.5)
		analytical_ratio = pdf_num / pdf_den
		ratio = UncertaintyQuantification._compute_conditional_pdf_ratio(sample, [1], θ, θ_cond, inputs)
		@test isapprox(ratio, analytical_ratio; atol=1e-10)
	end

	@testset "_compute_joint_pdf" begin
		stdnorm = Normal(0, 1)
		inputs_analytical = [
			RandomVariable(
				ProbabilityBox{Normal}(Dict(
					:μ => Interval(0.0, 0.0),
					:σ => Interval(1.0, 1.0)
				)), :x1),
			RandomVariable(
				ProbabilityBox{Normal}(Dict(
					:μ => Interval(0.0, 0.0),
					:σ => Interval(1.0, 1.0)
				)), :x2)
		]
		θ_analytical = Dict(:x1 => [0.0, 1.0], :x2 => [0.0, 1.0])
		sample = [0.3, -0.5]
		pdf_expected = pdf(stdnorm, 0.3) * pdf(stdnorm, -0.5)
		pdfval = UncertaintyQuantification._compute_joint_pdf(sample, θ_analytical, inputs_analytical)
		@test isapprox(pdfval, pdf_expected; atol=1e-10)
	end

	@testset "First-order sensitivity indices vs analytical" begin
		# Analytical values from the paper
		analytical_SEcut = [0.1293, 0.3712, 0.1293, 0.3682]
		analytical_SVcut = [0.0252, 0.2892, 0.0129, 0.6656]
		analytical_SERS  = [0.1283, 0.3732, 0.1283, 0.3654]
		analytical_SVRS  = [0.0247, 0.2966, 0.0127, 0.6492]

		names = ["x1_μ", "x2_μ", "x1_σ", "x2_σ"]

		le_indices = [lemcs_result.sensitivity_indices[Symbol(n)][1] for n in names]
		lv_indices = [lemcs_result.sensitivity_indices[Symbol(n)][2] for n in names]
		ge_indices = [gemcs_result.sensitivity_indices[Symbol(n)][1] for n in names]
		gv_indices = [gemcs_result.sensitivity_indices[Symbol(n)][2] for n in names]

		for (i, n) in enumerate(names)
			@test isapprox(le_indices[i], analytical_SEcut[i]; atol=0.03)
			@test isapprox(lv_indices[i], analytical_SVcut[i]; atol=0.05)
			@test isapprox(ge_indices[i], analytical_SERS[i]; atol=0.1)  # no latin hypercube sampling -> more tolerance
			@test isapprox(gv_indices[i], analytical_SVRS[i]; atol=0.125) # no latin hypercube sampling -> more tolerance
		end
	end

	@testset "_get_param_bounds_and_names" begin
		bounds, names = UncertaintyQuantification._get_param_bounds_and_names(inputs)
		@test length(bounds) == 4
		@test length(names) == 4
		@test all(x -> x isa Tuple{Float64, Float64}, values(bounds))
		@test all(x -> x isa String, names)
	end

	@testset "_get_anchor_template" begin
		t1 = UncertaintyQuantification._get_anchor_template(anchor)
		@test t1 == anchor
		t2 = UncertaintyQuantification._get_anchor_template([anchor, anchor])
		@test t2 == anchor
	end

	@testset "_set_parameters!" begin
		θ = Dict(:x1 => [0.0, 1.0], :x2 => [0.0, 1.0])
		UncertaintyQuantification._set_parameters!(θ, inputs, [1,3], [0.1, 0.2])
		@test θ[:x1][1] == 0.1
		@test θ[:x2][1] == 0.2
	end

	@testset "_assign_sensitivity_indices" begin
		comp_vars = Dict([1] => 0.5, [2] => 0.5)
		comp_vars_second = Dict([1] => 0.3, [2] => 0.7)
		names = ["x1_μ", "x1_σ"]
		sens = UncertaintyQuantification._assign_sensitivity_indices(comp_vars, comp_vars_second, names)
		@test sens[:x1_μ] == (0.5, 0.3)
		@test sens[:x1_σ] == (0.5, 0.7)
	end
	# Possible extension: lemcs with 500 samples and second order - takes about 4 seconds
end
