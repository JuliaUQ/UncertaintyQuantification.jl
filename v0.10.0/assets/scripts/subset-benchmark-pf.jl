
using Plots

include("subset-benchmark-base.jl")

pfs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

N_dimensions = 200
N_samples = 2000
N_runs = 100

pf_MH = zeros(length(pfs), N_runs)
pf_CS = zeros(length(pfs), N_runs)
pf_aCS = zeros(length(pfs), N_runs)

for (i, pf) in enumerate(pfs)
    pf_MH[i, :], pf_CS[i, :], pf_aCS[i, :] = run_sim(N_dimensions, pf, N_samples, N_runs)
end

l = @layout [a b; c _]

p_MH = plot(
    pfs,
    mean(pf_MH; dims=2);
    yerr=std(pf_MH; dims=2),
    xscale=:log10,
    yscale=:log10,
    lw=2,
    minorgrid=true,
    color=theme_palette(:auto)[1],
    legend=:topleft,
    label="SubSet-MH",
)

p_CS = plot(
    pfs,
    mean(pf_CS; dims=2);
    yerr=std(pf_CS; dims=2),
    xscale=:log10,
    yscale=:log10,
    lw=2,
    minorgrid=true,
    color=theme_palette(:auto)[2],
    legend=:topleft,
    label="SubSet-CS",
)

p_aCS = plot(
    pfs,
    mean(pf_aCS; dims=2);
    yerr=std(pf_aCS; dims=2),
    xscale=:log10,
    yscale=:log10,
    lw=2,
    minorgrid=true,
    color=theme_palette(:auto)[3],
    legend=:topleft,
    label="SubSet-aCS",
)

p = plot(p_MH, p_CS, p_aCS; layout=l)
