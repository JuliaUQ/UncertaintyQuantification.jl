using Literate

for (root, dirs, files) in walkdir("./docs/literate/")
    sp = splitpath(root)
    jl_files = filter(f -> endswith(f, ".jl"), files)
    if (!isempty(jl_files))
        Literate.script.(
            joinpath.(root, jl_files), joinpath.(pwd(), "./demo/", joinpath(sp[4:end]))
        )
    end
end