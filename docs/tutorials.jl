using Literate

src_dir = joinpath(@__DIR__, "..", "test", "Tutorials")
out_dir = joinpath(@__DIR__, "src", "Tutorials")

mkpath(out_dir)

tutorial_pages = Pair{String,String}[]

for file in sort(readdir(src_dir))
    if !endswith(file, ".jl")
        continue
    end
    Literate.markdown(
        joinpath(src_dir, file), out_dir;
        codefence = "````julia " => "````"
    )
    name = splitext(file)[1]
    push!(tutorial_pages, name => "Tutorials/$(name).md")
end
