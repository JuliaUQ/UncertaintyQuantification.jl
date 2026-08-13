# Function to check if (exact) line exits in file
function isline(file, string_check)
    for (i, line) in enumerate(eachline(file))
        if (line == string_check)
            return true
        end
    end

    return false
end

# Checks the pattern doesn't exist anywhere
function isnotanywhere(file, string_check)
    for (i, line) in enumerate(eachline(file))
        if (m = match(Regex(string_check), line); m !== nothing)
            return false
        end
    end

    return true
end
