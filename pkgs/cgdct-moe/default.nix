{ lib
, buildNpmPackage
, writeShellScriptBin
, julia
, python3
, glibcLocales
, gnugrep
, which
, validator-nu
, lychee
}:

buildNpmPackage rec {
  name = "cgdct-moe";

  src = ../..;

  npmDepsHash = "sha256-iqUwXXAQSjKgghyavvwqoTukqSMG57OUFJFDDmMsmSg=";

  site = "__site";

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
    NODE="$(which node)"
    export NODE

    runHook postConfigure
  '';

  strictDeps = true;
  nativeBuildInputs = [ julia' python' which ];

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

  doCheck = true;
  nativeCheckInputs = [ validator-nu lychee ];
  checkPhase = ''
    runHook preCheck

    source bin/vnu ${site}

    runHook postCheck
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
