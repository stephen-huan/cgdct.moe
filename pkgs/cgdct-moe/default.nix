{ lib
, buildNpmPackage
, writeShellScriptBin
, julia
, python3
, glibcLocales
, gnugrep
, which
}:

let
  site = "__site";
in
buildNpmPackage rec {
  name = "cgdct-moe";

  src = ../..;

  npmDepsHash = "sha256-VNu10Jd2LbuHVmwIVJHmN/JsK5BtANYdLZPIbsPGgEI=";

  julia' = julia.withPackages [
    "Dates"
    "Franklin"
    "JSON"
    "JuliaFormatter"
    "SHA"
  ];

  python' = python3.withPackages (ps: with ps; [
    beautifulsoup4
    lxml
  ]);

  # spoof `git log --pretty=%at -1`
  git' = writeShellScriptBin "git" ''
    echo 0
  '';

  # filter spurious warning messages during build
  filter = writeShellScriptBin "filter" ''
    ${lib.getExe gnugrep} \
      --invert-match \
      --regexp 'Unicode text character "[女時金悪]" used in math mode' \
      "$@"
  '';

  configurePhase = ''
    runHook preConfigure

    export LOCALE_ARCHIVE="${glibcLocales}/lib/locale/locale-archive"
    export LANG=en_US.UTF-8
    # give a valid node binary to Franklin.jl
    # https://github.com/tlienart/Franklin.jl/pull/1069
    export NODE="$(which node)"

    runHook postConfigure
  '';

  strictDeps = true;
  nativeBuildInputs = [ julia' python' git' which ];

  buildPhase = ''
    runHook preBuild

    julia --optimize=0 --color=yes --eval "
      using Franklin: optimize
      optimize(clear=true, prerender=true, minify=false)
    " 2> >("${lib.getExe filter}" >&2)

    runHook postBuild
  '';

  postBuild = ''
    python src/postprocess --verbose ${site}
    # don't ignore files in .gitignore
    npx prettier --write ${site} --ignore-path=.prettierignore
  '';

  installPhase = ''
    runHook preInstall

    cp -r ${site} -T $out

    runHook postInstall
  '';

  meta = with lib; {
    description = "Stephen Huan's personal website";
    homepage = "https://cgdct.moe/";
    license = licenses.unlicense;
    maintainers = [ ];
  };
}
