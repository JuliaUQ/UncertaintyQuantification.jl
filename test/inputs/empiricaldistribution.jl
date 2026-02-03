@testset "EmpiricalDistribution" begin
    data = [
        21.370,
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
        40.460,
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

    @test pdf(ed, minimum(ed)) ≈ 0.0 atol = 1e-10
    @test pdf(ed, maximum(ed)) ≈ 0.0 atol = 1e-10

    pdf_area, _ = hquadrature(x -> pdf(ed, x), minimum(ed), maximum(ed))

    @test pdf_area ≈ 1 atol = 0.01
    @test logpdf.(ed, samples) ≈ log.(pdf.(ed, samples))

    @test quantile.(ed, cdf.(ed, samples)) ≈ samples

    @testset "Linear binning" begin
        x = [randn(10_000)..., (5 .+ randn(10_000))...]

        ed = EmpiricalDistribution(x; nbins=2^12)

        @test mean(ed) ≈ 2.5 atol = 0.1

        @test all(insupport.(ed, x))
        @test all(pdf.(ed, x) .>= 0)

        samples = rand(ed, 10000)

        @test all(insupport.(ed, samples))
        @test all(pdf.(ed, samples) .>= 0)

        @test pdf(ed, minimum(ed)) ≈ 0.0 atol = 1e-10
        @test pdf(ed, maximum(ed)) ≈ 0.0 atol = 1e-10

        pdf_area, _ = hquadrature(x -> pdf(ed, x), minimum(ed), maximum(ed))

        @test pdf_area ≈ 1 atol = 0.01
        @test logpdf.(ed, samples) ≈ log.(pdf.(ed, samples))

        @test quantile.(ed, cdf.(ed, samples)) ≈ samples
    end
end
