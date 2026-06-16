@testsnippet InputsSetup begin
    using Copulas

    p = Parameter(3.14, :π)
    x = RandomVariable(Normal(0, 1), :x)
    y = RandomVariable(Normal(1, 1), :y)
    z = RandomVariable(Normal(0, 1), :z)
    jd = JointDistribution(GaussianCopula([1 0; 0 1]), [x, y])

    inputs = [p, jd, z]
end

@testitem "Inputs: sample" setup = [InputsSetup] begin
    samples = sample(inputs, 10)

    @test size(samples) == (10, 4)
    @test names(samples) == ["π", "x", "y", "z"]
    @test propertynames(samples) == [:π, :x, :y, :z]
end

@testitem "Inputs: mean" setup = [InputsSetup] begin
    means = mean(inputs)
    @test size(means) == (4,)
    @test means[1] == 3.14
    @test means[2] == 0
    @test means[3] == 1
    @test means[4] == 0
end

@testitem "Inputs: names" setup = [InputsSetup] begin
    @test names(inputs) == [:π, :x, :y, :z]
end

@testitem "Inputs: count_rvs" setup = [InputsSetup] begin
    @test count_rvs(inputs) == 3
end

@testitem "Inputs: broadcasting" setup = [InputsSetup] begin
    x = RandomVariable(Normal(), :x)
    u = rand(10)
    @test pdf.(x.dist, u) == pdf.(x, u)
end
