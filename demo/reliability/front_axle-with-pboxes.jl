using UncertaintyQuantification

a = RandomVariable(
    ProbabilityBox{Normal}(Dict(:μ => Interval(11, 13), :σ => 1.2), 0, Inf), :a
)
t = RandomVariable(
    ProbabilityBox{Normal}(Dict(:μ => Interval(13, 15), :σ => 1.4), 0, Inf), :t
)

b = RandomVariable(truncated(Normal(65, 6.5), 0, Inf), :b)
h = RandomVariable(truncated(Normal(85, 8.5), 0, Inf), :h)
μ_M, σ_M = distribution_parameters(3.5, 0.35, LogNormal)
M = RandomVariable(LogNormal(μ_M, σ_M), :M)
μ_T, σ_T = distribution_parameters(3.1, 0.31, LogNormal)
T = RandomVariable(LogNormal(μ_T, σ_T), :T)

function wx(a, h, t, b) # [mm^3]
    return a * (h - 2 * t)^3 / (6 * h) + b / (6 * h) * (h^3 - (h - 2 * t)^3)
end

function wz(a, h, t, b) # [mm^3]
    return 0.8 * b * t^2 + 0.4 * (a^3 * (h - 2 * t) / t)
end

Wx = Model(df -> wx.(df.a, df.h, df.t, df.b) .* 10^(-9), :Wx) # [m^3]
Wz = Model(df -> wz.(df.a, df.h, df.t, df.b) .* 10^(-9), :Wz) # [m^3]
σ = Model(df -> df.M .* 10^3 ./ df.Wx, :σ) # [N/m^2] = [Pa]
τ = Model(df -> df.T .* 10^3 ./ df.Wz, :τ) # [N/m^2] = [Pa]
σ_s = 680 * 10^6

p = Model(df -> sqrt.(df.σ .^ 2 .+ 3 .* df.τ .^ 2), :p)

inputs = [a, t, b, h, M, T]
model = [Wx, Wz, σ, τ, p]
performance = df -> σ_s .- df.p

pf_dl, x_lb, x_ub = probability_of_failure(
    model, performance, inputs, DoubleLoop(MonteCarlo(10^6), MonteCarlo(10^5))
)

rs = RandomSlicing(ImportanceSampling(4000))

pf_rs, out_lb, out_ub = probability_of_failure(model, performance, inputs, rs)
