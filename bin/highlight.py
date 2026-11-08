#!/usr/bin/env python3
"""
Modify a pandoc highlight style.
"""

import json

name = "atom-one-light.theme"

# `pandoc --print-highlight-style tango > atom-one-light.theme`
with open(name) as f:
    theme = json.load(f)
    styles = theme["text-styles"]

colors1 = {
    attribute["text-color"]
    for attribute in styles.values()
    if attribute["text-color"] is not None
}
# https://github.com/stephen-huan/polar.nvim
colors2 = {
    styles["Alert"]["text-color"]: "#d70000",
    styles["BuiltIn"]["text-color"]: "#c18401",
    styles["Comment"]["text-color"]: "#808080",
    styles["Error"]["text-color"]: "#d70000",
    styles["Float"]["text-color"]: "#986801",
    styles["Function"]["text-color"]: "#2f6aea",
    styles["Import"]["text-color"]: "#ca1243",
    styles["Keyword"]["text-color"]: "#a626a4",
    styles["Operator"]["text-color"]: "#a626a4",
    styles["String"]["text-color"]: "#2a871f",
    styles["Variable"]["text-color"]: "#0184bc",
}

theme["text-color"] = None
theme["background-color"] = None
theme["line-number-color"] = None
theme["line-number-background-color"] = None
for value in theme["text-styles"].values():
    value["text-color"] = colors2.get(value["text-color"], value["text-color"])

with open(name, "w") as f:
    json.dump(theme, f, indent=4)

print(sorted(colors1), len(colors1))
print(sorted(colors2.values()), len(colors2))
