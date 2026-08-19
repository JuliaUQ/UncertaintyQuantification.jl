@testitem "EmpiricalDistribution" setup = [TestSetup] begin
    data = [
        21.37,
        19.435,
        20.363,
        20.632,
        20.404,
        19.893,
        21.511,
        19.905,
        22.018,
        19.93,
        31.304,
        32.286,
        28.611,
        29.721,
        29.866,
        30.635,
        29.715,
        27.343,
        27.559,
        31.32,
        39.693,
        38.218,
        39.828,
        41.214,
        41.895,
        39.569,
        39.742,
        38.236,
        40.46,
        39.36,
        50.455,
        50.704,
        51.035,
        49.391,
        50.504,
        48.282,
        49.215,
        49.149,
        47.585,
        50.03,
    ]

    ed = EmpiricalDistribution(data)

    @test ed.h ≈ 1.0870436775976158

    @test mean(ed) ≈ 34.95964999999962
    @test var(ed) ≈ 120.7690583345086

    @test all(insupport.(ed, data))
    @test all(pdf.(ed, data) .>= 0)

    samples = rand(ed, 10000)

    @test all(insupport.(ed, samples))
    @test all(pdf.(ed, samples) .>= 0)

    @test pdf(ed, minimum(ed)) ≈ 0.0 atol = 1.0e-10
    @test pdf(ed, maximum(ed)) ≈ 0.0 atol = 1.0e-10

    pdf_area, _ = hquadrature(x -> pdf(ed, x), minimum(ed), maximum(ed))

    @test pdf_area ≈ 1 atol = 0.01
    @test logpdf.(ed, samples) ≈ log.(pdf.(ed, samples))

    @test quantile.(ed, cdf.(ed, samples)) ≈ samples

    # high magnitude data

    data = [
        -0.3634,
        0.2517,
        -0.315,
        -0.3113,
        0.8163,
        0.4767,
        -0.8596,
        -1.4693,
        -2.1143,
        0.0438,
        -0.8253,
        0.8403,
        0.4339,
        -0.3954,
        0.5171,
        1.4472,
        -0.0931,
        -1.2726,
        -0.3107,
        -0.0966,
        -1.3828,
        -1.4624,
        1.301,
        2.2663,
        1.5124,
        0.6488,
        0.8999,
        0.2842,
        1.1036,
        0.6639,
        0.1713,
        1.1199,
        1.9208,
        0.8006,
        -0.3856,
        -0.4397,
        0.6299,
        -0.0696,
        -0.7789,
        -0.8787,
    ]

    ed = EmpiricalDistribution(2000 .* data)      # large-scale: pre-fix this threw DomainError

    @test ed.h ≈ 851.4141601336096
    @test mean(ed) ≈ 216.2649999999922
    @test var(ed) ≈ 4.513752626845226e6

    @test all(insupport.(ed, 2000 .* data))
    @test all(pdf.(ed, 2000 .* data) .>= 0)

    samples = rand(ed, 10000)
    @test all(insupport.(ed, samples))
    @test all(pdf.(ed, samples) .>= 0)

    @test pdf(ed, minimum(ed)) ≈ 0.0 atol = 1.0e-10
    @test pdf(ed, maximum(ed)) ≈ 0.0 atol = 1.0e-10

    pdf_area, _ = hquadrature(x -> pdf(ed, x), minimum(ed), maximum(ed))
    @test pdf_area ≈ 1 atol = 0.01
    @test logpdf.(ed, samples) ≈ log.(pdf.(ed, samples))
    @test quantile.(ed, cdf.(ed, samples)) ≈ samples
end

@testitem "EmpiricalDiistribution: Linear binning" setup = [TestSetup] begin
    x = [randn(10_000)..., (5 .+ randn(10_000))...]

    ed = EmpiricalDistribution(x; nbins=2^12)

    @test mean(ed) ≈ 2.5 atol = 0.1

    @test all(insupport.(ed, x))
    @test all(pdf.(ed, x) .>= 0)

    samples = rand(ed, 10000)

    @test all(insupport.(ed, samples))
    @test all(pdf.(ed, samples) .>= 0)

    @test pdf(ed, minimum(ed)) ≈ 0.0 atol = 1.0e-10
    @test pdf(ed, maximum(ed)) ≈ 0.0 atol = 1.0e-10

    pdf_area, _ = hquadrature(x -> pdf(ed, x), minimum(ed), maximum(ed))

    @test pdf_area ≈ 1 atol = 0.01
    @test logpdf.(ed, samples) ≈ log.(pdf.(ed, samples))

    @test quantile.(ed, cdf.(ed, samples)) ≈ samples
end
