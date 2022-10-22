#!/bin/bash

libs="_libs"
root="node_modules"
# copy files from node_modules to _libs
mkdir -p "$libs/katex/"
cp -r "$root/katex/dist/fonts/"        "$libs/katex/"
cp    "$root/katex/dist/katex.min.js"  "$libs/katex/"
cp    "$root/katex/dist/katex.min.css" "$libs/katex/"
mkdir -p "$libs/highlight/styles/"
cp    "$root/highlight.js/styles/atom-one-light.css" \
      "$libs/highlight/styles/atom-one-light.min.css"

# build site
# julia -O0 -e "using Franklin: optimize; \
#     optimize(prerender=true, minify=false)"

# publish

