abstract type FiniteDifferencesMethod end

struct CentralFiniteDifferences <: FiniteDifferencesMethod
    order::Int
    derivative::Int

    function CentralFiniteDifferences(order::Int, derivative::Int = 1)
        return new(order, derivative)
    end
end

struct ForwardFiniteDifferences <: FiniteDifferencesMethod
    order::Int
    derivative::Int

    function ForwardFiniteDifferences(order::Int, derivative::Int = 1)
        return new(order, derivative)
    end
end

struct BackwardFiniteDifferences <: FiniteDifferencesMethod
    order::Int
    derivative::Int

    function BackwardFiniteDifferences(order::Int, derivative::Int = 1)
        return new(order, derivative)
    end
end
