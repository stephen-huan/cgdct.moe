{ lib
, stdenvNoCC
, python3
, haskellPackages
, pandoc
, pandoc-katex
, prettier
, validator-nu
, lychee
, testers
, cgdct-moe
}:

let
  site = "build";
  python' = python3.withPackages (ps: with ps; [
    beautifulsoup4
    lxml
    pyyaml
    watchfiles
  ]);
in
stdenvNoCC.mkDerivation {
  name = "cgdct-moe";

  strictDeps = true;
  __structuredAttrs = true;

  src = ../.;

  nativeBuildInputs = [
    haskellPackages.pandoc-crossref
    pandoc
    pandoc-katex
    python'
    prettier
  ];

  buildPhase = ''
    runHook preBuild

    python bin/build.py

    runHook postBuild
  '';

  postBuild = ''
    python bin/postprocess.py --verbose ${site}
    # don't ignore files in .gitignore
    prettier --write ${site} --ignore-path=.prettierignore
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

  passthru.tests = {
    links = testers.lycheeLinkCheck {
      site = cgdct-moe;
      relocatable = false;
    };
  };

  meta = with lib; {
    description = "Stephen Huan's personal website";
    homepage = "https://cgdct.moe/";
    license = licenses.unlicense;
    maintainers = with maintainers; [ stephen-huan ];
  };
}
