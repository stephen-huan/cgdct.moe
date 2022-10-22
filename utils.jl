"""
    getvar(page, var; default)

Get `var` from `page`, using the simplier `Franklin.locvar` if possible.
"""
function getvar(page, var; default)
    (page == locvar(:fd_rpath)) ?
        locvar(var; default) : pagevar(page, var; default)
end

"""
    robust_title(page)

Get the title field for `page`, defaulting to the path if not defined.
"""
robust_title(page) = getvar(page, :title; default="/$page/")

"""
    robust_date(page)

Get the date field for `page`, defaulting to date of creation if not defined.
"""
robust_date(page) = getvar(page, :date;
    default=Date(Dates.unix2datetime(stat(page * ".md").ctime))
)

"""
    write_header!(io, page; rss=true)

Render the metadata about the blog post `page` to `io`.
"""
function write_header!(io, page; rss=true)
    # short description
    description = pagevar(page, :rss_description)
    if !isnothing(description) && rss
        write(io, "<p>$description</p>")
    end
    # date
    date = robust_date(page)
    write(io, """<span class="post-meta">$date</span>""")
    # tags
    tags = pagevar(page, :tags; default=String[])
    if length(tags) > 0
        tag_path = globvar(:tag_page_path)
        write(io, """<span class="post-tags">&nbsp; &middot;""")
        for tag in tags
            tag_url = "/$tag_path/$tag/"
            write(io, """&nbsp; <a href="$tag_url"><b>#</b> $tag</a>""")
        end
        write(io, "</span>")
    end
end

"""
    hfun_makeheader()

Make the header list for the website.
"""
function hfun_makeheader()
    current_url = get_url(locvar(:fd_rpath))
    io = IOBuffer()
    write(io, "<ul>")
    for (url, name) in globvar(:headers)
        is_active = (url == current_url) ? "active" : ""
        write(io, """<li><a href="$url" class="$is_active">$name</a></li>\n""")
    end
    write(io, "</ul>")
    return String(take!(io))
end

"""
    hfun_lastupdated()

Return the modification time, ignoring automatically generated pages.
"""
function hfun_lastupdated()
    url = get_url(locvar(:fd_rpath))
    exclude = Set(["/blog/", "/404/"])
    (in(url, exclude)) ?  "" : "Last updated: $(locvar(:fd_mtime))."
end

"""
    hfun_custom_taglist()

Generate a custom tag list.

See: https://tlienart.github.io/FranklinTemplates.jl/templates/basic/menu3/
"""
function hfun_custom_taglist()
    # -------------------------------------------------------------
    # Part1: Retrieve all pages associated with the tag & sort them
    # -------------------------------------------------------------
    # retrieve the tag string
    tag = locvar(:fd_tag)
    # recover the relative paths to all pages that
    # have that tag, these are paths like /blog/page1
    rpaths = globvar(:fd_tag_pages)[tag]
    # you might want to sort these pages by chronological order
    # you could also only show the most recent 5 etc...
    sort!(rpaths, by=robust_date, rev=true)

    # --------------------------------
    # Part2: Write the HTML to plug in
    # --------------------------------
    # instantiate a buffer in which we will
    # write the HTML to plug in the tag page
    io = IOBuffer()
    write(io, """<div class="tagged-posts"><table><tbody>\n""")
    # go over all paths
    for rpath in rpaths
        write(io, "<tr>")
        # recover the url corresponding to the rpath
        url = get_url(rpath)
        title, date = robust_title(rpath), robust_date(rpath)
        # write some appropriate HTML
        write(io, """<th scope="row">$date</th>""")
        write(io, """<td><a href="/$rpath/">$title</a></td>""")
        write(io, "</tr>\n")
    end
    # finish the HTML
    write(io, "</table></tbody></div>")
    # return the HTML string
    return String(take!(io))
end

"""
    hfun_gettags()

Render the list of tags for blog posts.
"""
function hfun_gettags()
    path = globvar(:tag_page_path)
    tags = locvar(:displaytags)
    io = IOBuffer()
    write(io, "<ul>\n")
    for (i, tag) in enumerate(tags)
        url = "/$path/$tag/"
        write(io, """<li><a href="$url"><b>#</b> $tag</a></li>\n""")
        i != length(tags) && write(io, "<p>&nbsp; &middot; &nbsp;</p>\n")
    end
    write(io, "</ul>")
    return String(take!(io))
end

"""
    hfun_getposts()

Render the list of blog posts in reverse chronological order.

See: https://franklinjl.org/demos/#007_delayed_hfun
"""
@delay function hfun_getposts()
    io = IOBuffer()
    write(io, "<ul>\n")
    for post in sort(readdir("blog"; join=true), by=robust_date, rev=true)
        post == "blog/index.md" && continue
        write(io, "<li>")
        url, title = get_url(post), robust_title(post)
        write(io, """<h3><a href="$url">$title</a></h3>""")
        write_header!(io, post)
        write(io, "</li>\n")
    end
    write(io, "</ul>")
    return String(take!(io))
end

"""
    hfun_maketitle()

Make the title for blog posts.
"""
function hfun_maketitle()
    io = IOBuffer()
    post = locvar(:fd_rpath)
    title = robust_title(post)
    write(io, "<h1>$title</h1>\n")
    write_header!(io, post; rss=false)
    return String(take!(io))
end

"""
    lx_news(com, _)

Get the `n` most recent news entries.
"""
function lx_news(com, _)
    n = parse(Int64, Franklin.content(com.braces[1]))
    io = IOBuffer()
    write(io, "@@news", "\n")
    write(io, "| Date       | Description |", "\n")
    write(io, "|:-----------|-------------|", "\n")
    i = -1
    open("news.md") do news
        for line in eachline(news)
            if line == "@@news"
                i = 0
            end
            i >= 0 && (i += 1)
            1 <= i - 3 <= n && write(io, line, "\n")
        end
    end
    write(io, "@@", "\n")
    return String(take!(io))
end

