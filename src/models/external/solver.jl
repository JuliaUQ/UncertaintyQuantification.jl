struct Solver
    path::String
    source::String
    args::String
    function Solver(path::String, source::String, args::String)
        if !isabspath(path)
            @warn "Solver path is not absolute. Make sure $path is on your PATH."
        end
        return new(path, source, args)
    end
end

Solver(path::String, source::String; args::String = "") = Solver(path, source, args)

function run(solver::Solver, folder::String)
    binary = solver.path
    source = solver.source
    args = solver.args

    out = basename(binary) * "UncertaintyQuantification.out"
    err = basename(binary) * "UncertaintyQuantification.err"

    p = pipeline(
        !isempty(args) ? `$binary $args $source` : `$binary $source`; stdout = out, stderr = err
    )

    cd(() -> run(p), folder)

    return nothing
end
