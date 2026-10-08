using Plots

include("subset-benchmark-base.jl")

pf = 10^-4
N_dimensions = [100, 200, 500, 1000, 2000, 5000, 10000]

N_samples = 2000
N_runs = 100

pf_MH = zeros(length(N_dimensions), N_runs)
pf_CS = zeros(length(N_dimensions), N_runs)
pf_aCS = zeros(length(N_dimensions), N_runs)

for (i, N_d) in enumerate(N_dimensions)
    pf_MH[i, :], pf_CS[i, :], pf_aCS[i, :] = run_sim(N_d, pf, N_samples, N_runs)
end

pf_MH_lo = [quantile(pf_MH[i, :], 0.025) for i in 1:length(N_dimensions)]
pf_MH_hi = [quantile(pf_MH[i, :], 0.975) for i in 1:length(N_dimensions)]

pf_CS_lo = [quantile(pf_CS[i, :], 0.025) for i in 1:length(N_dimensions)]
pf_CS_hi = [quantile(pf_CS[i, :], 0.975) for i in 1:length(N_dimensions)]

pf_aCS_lo = [quantile(pf_aCS[i, :], 0.025) for i in 1:length(N_dimensions)]
pf_aCS_hi = [quantile(pf_aCS[i, :], 0.975) for i in 1:length(N_dimensions)]

p = plot(
    N_dimensions,
    pf_MH_hi;
    fillrange=pf_MH_lo,
    label="SubSet-MH",
    alpha=0.2,
    color=theme_palette(:auto)[1],
)
plot!(
    p,
    N_dimensions,
    pf_CS_hi;
    fillrange=pf_CS_lo,
    label="SubSet-CS",
    alpha=0.2,
    color=theme_palette(:auto)[2],
)
plot!(
    p,
    N_dimensions,
    pf_aCS_hi;
    fillrange=pf_aCS_lo,
    label="SubSet-aCS",
    alpha=0.2,
    color=theme_palette(:auto)[3],
)

plot!(
    p,
    N_dimensions,
    mean(pf_MH; dims=2);
    color=theme_palette(:auto)[1],
    label=false,
    lw=2,
    seriestype=:scatter,
)
plot!(
    p,
    N_dimensions,
    mean(pf_CS; dims=2);
    color=theme_palette(:auto)[2],
    label=false,
    lw=2,
    seriestype=:scatter,
)
plot!(
    p,
    N_dimensions,
    mean(pf_aCS; dims=2);
    color=theme_palette(:auto)[3],
    label=false,
    lw=2,
    seriestype=:scatter,
)

plot!(
    p,
    N_dimensions,
    fill(pf, length(N_dimensions));
    minorgrid=true,
    xscale=:log10,
    yscale=:log10,
    label="target pf",
    legend=:bottomright,
    xlabel="number of dimensions",
    ylabel="pf",
    lw=2,
    color=theme_palette(:auto)[4],
)
