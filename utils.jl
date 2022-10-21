"""
    hfun_makeheader()

Make the header list for the website.
"""
function hfun_makeheader()
    current_page = "/$(splitext(locvar("fd_rpath"))[1])"
    current_page = (endswith(current_page, "index")) ?
        current_page[1:end - 5] : "$current_page/"
    io = IOBuffer()
    write(io, "<ul>")
    for (url, name) in globvar("headers")
        is_active = (url == current_page) ? "active" : ""
        write(io, """<li><a href="$url" class="$is_active">$name</a></li>\n""")
    end
    write(io, "</ul>")
    return String(take!(io))
end

"""
    lx_news(com, _)

Get the `n` most recent news entries.
"""
function lx_news(com, _)
    n = parse(Int64, Franklin.content(com.braces[1]))
    lines = Vector{String}(undef, n + 4)
    lines[1] = "@@news"
    lines[2] = "| Date       | Description |"
    lines[3] = "|:-----------|-------------|"
    lines[end] = "@@"
    i = -1
    open("news.md") do io
        for line in eachline(io)
            if line == "@@news"
                i = 0
            end
            i >= 0 && (i += 1)
            1 <= i - 3 <= n && (lines[i] = line)
        end
    end
    return join(lines, "\n")
end

