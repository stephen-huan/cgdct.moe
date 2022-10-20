"""
    hfun_makeheader

Make the header list for the website.
"""
function hfun_makeheader()
    current_page = splitext(locvar("fd_rpath"))[1]
    current_page == "index" && (current_page = "")
    io = IOBuffer()
    write(io, "<ul>")
    for (url, name) in globvar("headers")
        is_active = (url[2:end - 1] == current_page) ? "active" : ""
        write(io, """<li><a href="$url" class="$is_active">$name</a></li>\n""")
    end
    write(io, "</ul>")
    return String(take!(io))
end

