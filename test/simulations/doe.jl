@testset "Design of Experiments" begin
    for doe in subtypes(AbstractDesignOfExperiments)
        @test hasfield(doe, :q)
    end

    @testset "TwoLevelFactorial" begin
        tlf = TwoLevelFactorial()

        @test doe_samples(tlf, 2) == [
            0.0 0.0
            1.0 0.0
            0.0 1.0
            1.0 1.0
        ]

        @test doe_samples(tlf, 3) == [
            0.0 0.0 0.0
            1.0 0.0 0.0
            0.0 1.0 0.0
            1.0 1.0 0.0
            0.0 0.0 1.0
            1.0 0.0 1.0
            0.0 1.0 1.0
            1.0 1.0 1.0
        ]

        @test doe_samples(tlf, 4) == [
            0.0 0.0 0.0 0.0
            1.0 0.0 0.0 0.0
            0.0 1.0 0.0 0.0
            1.0 1.0 0.0 0.0
            0.0 0.0 1.0 0.0
            1.0 0.0 1.0 0.0
            0.0 1.0 1.0 0.0
            1.0 1.0 1.0 0.0
            0.0 0.0 0.0 1.0
            1.0 0.0 0.0 1.0
            0.0 1.0 0.0 1.0
            1.0 1.0 0.0 1.0
            0.0 0.0 1.0 1.0
            1.0 0.0 1.0 1.0
            0.0 1.0 1.0 1.0
            1.0 1.0 1.0 1.0
        ]
    end

    @testset "FullFactorial" begin
        @test_throws ErrorException("Levels must be >= 2") FullFactorial([2, 0])

        ff = FullFactorial([3, 4, 2])
        @test ff.levels == [3, 4, 2]
        @test doe_samples(ff, 3) == [
            0.0 0.0 0.0
            0.5 0.0 0.0
            1.0 0.0 0.0
            0.0 0.3333333333333333 0.0
            0.5 0.3333333333333333 0.0
            1.0 0.3333333333333333 0.0
            0.0 0.6666666666666666 0.0
            0.5 0.6666666666666666 0.0
            1.0 0.6666666666666666 0.0
            0.0 1.0 0.0
            0.5 1.0 0.0
            1.0 1.0 0.0
            0.0 0.0 1.0
            0.5 0.0 1.0
            1.0 0.0 1.0
            0.0 0.3333333333333333 1.0
            0.5 0.3333333333333333 1.0
            1.0 0.3333333333333333 1.0
            0.0 0.6666666666666666 1.0
            0.5 0.6666666666666666 1.0
            1.0 0.6666666666666666 1.0
            0.0 1.0 1.0
            0.5 1.0 1.0
            1.0 1.0 1.0
        ]

        ff = FullFactorial([4, 2])
        @test doe_samples(ff, 2) == [
            0.0 0.0
            0.3333333333333333 0.0
            0.6666666666666666 0.0
            1.0 0.0
            0.0 1.0
            0.3333333333333333 1.0
            0.6666666666666666 1.0
            1.0 1.0
        ]
    end

    @testset "FractionalFactorial" begin
        frac = FractionalFactorial(["a", "b", "c", "ac"])
        @test frac.columns == ["a", "b", "c", "ac"]
        @test doe_samples(frac) == [
            0.0 0.0 0.0 1.0
            1.0 0.0 0.0 0.0
            0.0 1.0 0.0 1.0
            1.0 1.0 0.0 0.0
            0.0 0.0 1.0 0.0
            1.0 0.0 1.0 1.0
            0.0 1.0 1.0 0.0
            1.0 1.0 1.0 1.0
        ]

        frac = FractionalFactorial(["a", "b", "ba"])
        @test doe_samples(frac) == [
            0.0 0.0 1.0
            1.0 0.0 0.0
            0.0 1.0 0.0
            1.0 1.0 1.0
        ]
    end

    @testset "BoxBehnken" begin
        @test_throws ErrorException("centers must be nonnegative") BoxBehnken(-1)

        bb = BoxBehnken()

        @test doe_samples(bb, 2) == [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0; 0.5 0.5]
        @test doe_samples(bb, 5) == [
            0.0 0.0 0.5 0.5 0.5
            1.0 0.0 0.5 0.5 0.5
            0.0 1.0 0.5 0.5 0.5
            1.0 1.0 0.5 0.5 0.5
            0.0 0.5 0.0 0.5 0.5
            1.0 0.5 0.0 0.5 0.5
            0.0 0.5 1.0 0.5 0.5
            1.0 0.5 1.0 0.5 0.5
            0.0 0.5 0.5 0.0 0.5
            1.0 0.5 0.5 0.0 0.5
            0.0 0.5 0.5 1.0 0.5
            1.0 0.5 0.5 1.0 0.5
            0.0 0.5 0.5 0.5 0.0
            1.0 0.5 0.5 0.5 0.0
            0.0 0.5 0.5 0.5 1.0
            1.0 0.5 0.5 0.5 1.0
            0.5 0.0 0.0 0.5 0.5
            0.5 1.0 0.0 0.5 0.5
            0.5 0.0 1.0 0.5 0.5
            0.5 1.0 1.0 0.5 0.5
            0.5 0.0 0.5 0.0 0.5
            0.5 1.0 0.5 0.0 0.5
            0.5 0.0 0.5 1.0 0.5
            0.5 1.0 0.5 1.0 0.5
            0.5 0.0 0.5 0.5 0.0
            0.5 1.0 0.5 0.5 0.0
            0.5 0.0 0.5 0.5 1.0
            0.5 1.0 0.5 0.5 1.0
            0.5 0.5 0.0 0.0 0.5
            0.5 0.5 1.0 0.0 0.5
            0.5 0.5 0.0 1.0 0.5
            0.5 0.5 1.0 1.0 0.5
            0.5 0.5 0.0 0.5 0.0
            0.5 0.5 1.0 0.5 0.0
            0.5 0.5 0.0 0.5 1.0
            0.5 0.5 1.0 0.5 1.0
            0.5 0.5 0.5 0.0 0.0
            0.5 0.5 0.5 1.0 0.0
            0.5 0.5 0.5 0.0 1.0
            0.5 0.5 0.5 1.0 1.0
            0.5 0.5 0.5 0.5 0.5
            0.5 0.5 0.5 0.5 0.5
            0.5 0.5 0.5 0.5 0.5
            0.5 0.5 0.5 0.5 0.5
            0.5 0.5 0.5 0.5 0.5
            0.5 0.5 0.5 0.5 0.5
        ]

        bb = BoxBehnken(8)
        @test doe_samples(bb, 2) == [
            0.0 0.0
            1.0 0.0
            0.0 1.0
            1.0 1.0
            0.5 0.5
            0.5 0.5
            0.5 0.5
            0.5 0.5
            0.5 0.5
            0.5 0.5
            0.5 0.5
            0.5 0.5
        ]

        bb = BoxBehnken()
        factors = [3, 4, 6, 7, 9, 10, 11, 12, 16]
        designsizes = [15, 27, 54, 62, 130, 170, 188, 204, 396]

        for (f, s) in zip(factors, designsizes)
            m = doe_samples(bb, f)
            @test size(m) == (s, f)
        end
    end

    @testset "CentralComposite" begin
        @test_throws ErrorException("type must be :inscribed or :face.") CentralComposite(
            :type
        )

        cc = CentralComposite(:face)
        @test doe_samples(cc, 3) == [
            0.0 0.0 0.0
            1.0 0.0 0.0
            0.0 1.0 0.0
            1.0 1.0 0.0
            0.0 0.0 1.0
            1.0 0.0 1.0
            0.0 1.0 1.0
            1.0 1.0 1.0
            1.0 0.5 0.5
            0.0 0.5 0.5
            0.5 1.0 0.5
            0.5 0.0 0.5
            0.5 0.5 1.0
            0.5 0.5 0.0
            0.5 0.5 0.5
        ]

        cc = CentralComposite(:inscribed)
        @test doe_samples(cc, 3) == [
            0.21132486540518708 0.21132486540518708 0.21132486540518708
            0.7886751345948129 0.21132486540518708 0.21132486540518708
            0.21132486540518708 0.7886751345948129 0.21132486540518708
            0.7886751345948129 0.7886751345948129 0.21132486540518708
            0.21132486540518708 0.21132486540518708 0.7886751345948129
            0.7886751345948129 0.21132486540518708 0.7886751345948129
            0.21132486540518708 0.7886751345948129 0.7886751345948129
            0.7886751345948129 0.7886751345948129 0.7886751345948129
            1.0 0.5 0.5
            0.0 0.5 0.5
            0.5 1.0 0.5
            0.5 0.0 0.5
            0.5 0.5 1.0
            0.5 0.5 0.0
            0.5 0.5 0.5
        ]
    end

    @testset "PlackettBurman" begin
        pb = PlackettBurman()

        @test_throws ErrorException("number of random variables must be in 2:47") doe_samples(
            pb, 48
        )
        @test_throws ErrorException("number of random variables must be in 2:47") doe_samples(
            pb, 1
        )

        nvars = [7 17 rand(2:47) rand(2:47) rand(2:47)]

        for n in nvars
            m = doe_samples(pb, n)
            combinations = [0 0 0 0]
            for i in 1:(n - 1)
                for j in (i + 1):n
                    map(
                        row -> combinations[(Int)(2 * row[i] + row[j] + 1)] += 1, eachrow(m)
                    )
                end
            end
            @test combinations[1] == combinations[2] == combinations[3] == combinations[4]
        end

        factors = [3, 5, 9, 14, 19, 20, 25, 31, 32, 37, 42, 47] # one per hardcoded design, varying the amount of omitted columns
        designsizes = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]

        for (f, s) in zip(factors, designsizes)
            m = doe_samples(pb, f)
            @test size(m) == (s, f)
        end
    end

    @testset "Variables and Generators" begin
        v, i, g, gi = UncertaintyQuantification.variables_and_generators(["a", "b", "ab"])
        @test v == "ab"
        @test i == [1, 2]
        @test g == ["ab"]
        @test gi == [3]

        v, i, g, gi = UncertaintyQuantification.variables_and_generators([
            "a", "b", "ab", "c", "ac"
        ])
        @test v == "abc"
        @test i == [1, 2, 4]
        @test g == ["ab", "ac"]
        @test gi == [3, 5]

        v, i, g, gi = UncertaintyQuantification.variables_and_generators([
            "c", "abc", "b", "a"
        ])
        @test v == "cba"
        @test i == [1, 3, 4]
        @test g == ["abc"]
        @test gi == [2]

        @test_throws ErrorException(
            "Each String in columns must hold at least one character"
        ) UncertaintyQuantification.variables_and_generators(["a", "", "b", "ab"])

        @test_throws ErrorException(
            "All combinations of columns must also be individual columns."
        ) UncertaintyQuantification.variables_and_generators(["a", "b", "ac", "ab"])
    end

    @testset "Sampling" begin
        x = RandomVariable(Uniform(-1, 1), :x)
        y = RandomVariable(Uniform(0, 1), :y)

        ff = FullFactorial([3, 3], 1.0 - 1e-12)

        samples = sample([x, y], ff)

        @test cdf.(x, samples.x) ≈ [0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
        @test cdf.(y, samples.y) ≈ [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]

        x = RandomVariable(Normal(1, 1), :x)
        y = RandomVariable(Exponential(2), :y)

        ff = FullFactorial([5, 5], 1.0 - 1e-12)

        samples = sample([x, y], ff)

        @test all(isfinite.(samples.x))
        @test all(isfinite.(samples.y))

        @test cdf.(x, samples.x[1:5]) ≈ [0.0, 0.25, 0.5, 0.75, 1.0]
        @test cdf.(y, samples.y[1:5:end]) ≈ [0.0, 0.25, 0.5, 0.75, 1.0]
    end
end
