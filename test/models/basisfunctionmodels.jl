@testset "BasisFunctionModel" begin

	@testset "MonomialBasis" begin

		x = RandomVariable.(Uniform(-5, 5), [:x1, :x2])
		himmelblau = Model(
			df -> (df.x1 .^ 2 .+ df.x2 .- 11) .^ 2 .+ (df.x1 .+ df.x2 .^ 2 .- 7) .^ 2, :y,
		)

		data = sample(x, FullFactorial([5, 5]))
		evaluate!(himmelblau, data)

		basis = MonomialBasis(2, 4)
		bfm = BasisFunctionModel(data, :y, basis)

		test_data = sample(x, SobolSampling(1024))
		validate_data = copy(test_data)

		evaluate!(himmelblau, test_data)
		evaluate!(bfm, validate_data)

		mse = mean((test_data.y .- validate_data.y) .^ 2)

		@test mse < 1e-25
	end

	@testset "AdditiveBasis" begin
		x = RandomVariable.(Uniform(-π, π), [:x1, :x2, :x3])
		a = Parameter(7, :a)
		b = Parameter(0.1, :b)

		inputs = [x; a; b]

		ishigami = Model(
			df ->
				sin.(df.x1) .+ df.a .* sin.(df.x2) .^ 2 .+ df.b .* (df.x3 .^ 4) .* sin.(df.x1),
			:y,
		)

		Random.seed!(8128)

		data = sample(inputs, 100)
		evaluate!(ishigami, data)

		Random.seed!()
		phrbasis = PolyharmonicRadialBasis(permutedims(Matrix(data[:, [:x1, :x2, :x3]])), 2)
		monobasis = MonomialBasis(3, 1)
		basis = AdditiveBasis([phrbasis, monobasis])

		bfm = BasisFunctionModel(data, :y, basis, [:x1, :x2, :x3])

		test_data = select(data, Not(:y))

		evaluate!(bfm, test_data)

		mse = mean((data.y .- test_data.y) .^ 2)
		@test isapprox(mse, 0; atol = eps(Float64))
	end
end
