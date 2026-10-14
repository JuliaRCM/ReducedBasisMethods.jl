# The words a generic package must not write: ReducedBasisMethods knows no model, so a physical
# word in its sources or its documentation is a physical example that does not belong to it (T7.15,
# section 3.6 of the plan). One definition, two callers: `docs/make.jl` greps the site it has built,
# and `test/quality/docs.jl` greps the sources, so a word fails the Documentation workflow and the
# suite alike.
const PHYSICAL_WORDS = r"vlasov|plasma|particle"i

"""
    forbidden_files(path)

Every file under `path` whose text holds one of `PHYSICAL_WORDS`, as paths relative to `path`. A
directory is walked recursively; a file is taken as it is. The words are pure ASCII, and reading a
file as a `String` does not check its encoding, so a binary of the built site is searched too.
"""
function forbidden_files(path)
    files = isdir(path) ?
            sort!([joinpath(root, name) for (root, _, names) in walkdir(path)
                   for name in names]) :
            String[path]
    return [relpath(file, path)
            for file in files if occursin(PHYSICAL_WORDS, read(file, String))]
end
