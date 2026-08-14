using Literate

"""
    function fuseConvert(path::String, out::IOStream, r::String, dir::String)

Groups all files from a directory into a temp file and creates one markdown-file (saved to ./docs/src/examples) using Literate.jl.

"""
function fuseConvert(r::String, dir::String; documenter::Bool = true)
    path = tempname(; cleanup = false)

    # merge all example files of a subfolder into one
    open(path, "a") do out
        for (root, _, files) in walkdir(joinpath(r, dir))
            for file in files
                write(out, read(joinpath(root, file), String))
            end
        end
    end

    Literate.markdown(path, "./docs/src/examples"; documenter = documenter, name = dir)

    rm(path) # delete the temporary file

    #remove @meta block created by literate
    if documenter
        example_file = joinpath("./docs/src/examples", "$dir.md")
        lines = readlines(example_file; keep = true)
        open(example_file, "w") do file
            write.(file, lines[5:end])
        end
    end

    return nothing
end

for (r, d, f) in walkdir("./docs/literate/")
    for dir in d
        if dir == "hpc" || dir == "external"
            fuseConvert(r, dir; documenter = false)
        else
            fuseConvert(r, dir)
        end
    end
end
