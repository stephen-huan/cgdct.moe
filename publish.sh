#!/bin/bash

libs="_libs"
root="node_modules"

commit="$(git rev-parse --short HEAD)"
main_branch="master"
build_branch="build"
push_branch="gh-pages"

# copy files from node_modules to _libs
mkdir -p "$libs/katex/"
cp -r "$root/katex/dist/fonts/"        "$libs/katex/"
cp    "$root/katex/dist/katex.min.js"  "$libs/katex/"
cp    "$root/katex/dist/katex.min.css" "$libs/katex/"
mkdir -p "$libs/highlight/styles/"
cp    "$root/highlight.js/styles/atom-one-light.css" \
      "$libs/highlight/styles/atom-one-light.min.css"

git checkout "$build_branch"
# sync changes
git merge "$main_branch"

# build site
julia --project="@." -O0 -e "using Franklin: optimize; \
    optimize(clear=true, prerender=true, minify=false)"

# publish
git add --force __site
git commit --message "Deploying to $push_branch from @ $commit"
# https://gist.github.com/cobyism/4730490
git subtree push --prefix __site origin "$push_branch"

git checkout "$main_branch"

