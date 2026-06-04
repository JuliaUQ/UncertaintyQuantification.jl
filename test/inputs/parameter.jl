@testitem "Parameter" setup = [Import] begin
    π = Parameter(3.14, :π)
    x = DataFrame(; π=[3.14, 3.14, 3.14, 3.14, 3.14])

    @test π isa Parameter
    @test π.value == 3.14
    @test π.name == :π

    @test sample(π, 5) == x

    @test mean(π) == 3.14

    to_standard_normal_space!(π, x)
    @test x == DataFrame(; π=[3.14, 3.14, 3.14, 3.14, 3.14])

    to_physical_space!(π, x)
    @test x == DataFrame(; π=[3.14, 3.14, 3.14, 3.14, 3.14])
end
