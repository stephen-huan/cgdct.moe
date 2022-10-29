#!/bin/bash
# script to update dependencies

libs="_libs"
root="node_modules"

julia --project="@." -O0 -e "using Pkg: update; update()"

npm update
# copy files from node_modules to _libs
mkdir -p "$libs/katex/"
cp -r "$root/katex/dist/fonts/"        "$libs/katex/"
cp    "$root/katex/dist/katex.min.js"  "$libs/katex/"
cp    "$root/katex/dist/katex.min.css" "$libs/katex/"
mkdir -p "$libs/highlight/styles/"
cp    "$root/highlight.js/styles/atom-one-light.css" \
      "$libs/highlight/styles/atom-one-light.min.css"

