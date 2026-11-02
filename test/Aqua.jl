@testsnippet AquaSetup begin
    using Aqua
    using UncertaintyQuantification
end

@testitem "Package quality checks" setup = [AquaSetup] begin
    Aqua.test_all(UncertaintyQuantification)
end
