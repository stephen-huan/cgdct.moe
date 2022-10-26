#!/bin/bash
# script to deploy the site to GitHub pages

main_branch="master"
build_branch="build"
push_branch="gh-pages"
commit="$(git rev-parse --short HEAD)"

git checkout "$build_branch" &&
# sync changes
git merge --no-verify "$main_branch" &&

# build site
julia --project="@." -O0 -e "using Franklin: optimize; \
    optimize(clear=true, prerender=true, minify=false)" &&

# publish
git add --force __site &&
git commit --message "Deploying to $push_branch from @ $commit" &&
# https://gist.github.com/cobyism/4730490
git subtree push --prefix __site origin "$push_branch"

git checkout "$main_branch"

