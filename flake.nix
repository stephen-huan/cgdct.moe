{
  description = "cgdct.moe";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;
      systems = lib.systems.flakeExposed;
      eachDefaultSystem = f: builtins.foldl' lib.attrsets.recursiveUpdate { }
        (map f systems);
    in
    eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        # site = self.packages.${system}.default;
        # inherit (site) filter julia';
        # site-dependencies = site.nativeBuildInputs;
        site-dependencies = [ ];
        python' = pkgs.python3.withPackages (ps: with ps; [
          pyyaml
          watchfiles
        ]);
      in
      {
        packages.${system} = let inherit (pkgs) callPackage; in {
          # default = callPackage ./pkgs/cgdct-moe { };
        };

        formatter.${system} = pkgs.writeShellScriptBin "formatter" ''
          npx prettier --write "$@"
          {julia'}/bin/julia --eval "using JuliaFormatter; format(\"$1\")"
          ${lib.getExe pkgs.isort} "$@"
          ${lib.getExe pkgs.black} "$@"
          ${lib.getExe pkgs.shfmt} --write "$@"
          ${lib.getExe pkgs.nixpkgs-fmt} "$@"
        '';

        checks.${system}.lint = pkgs.buildNpmPackage {
          name = "lint";
          src = ./.;
          # inherit (site) npmDepsHash;
          dontNpmBuild = true;
          doCheck = true;
          nativeCheckInputs = site-dependencies;
          checkPhase = ''
            npx prettier --check .
            ${lib.getExe pkgs.isort} --check --diff .
            ${lib.getExe pkgs.black} --check --diff .
            ${lib.getExe pkgs.ruff} check .
            ${lib.getExe pkgs.shfmt} --diff .
            ${lib.getExe pkgs.statix} check
            source ./bin/vnu _assets
            source ./bin/vnu _css
            source ./bin/vnu _libs
          '';
          installPhase = "touch $out";
        };

        apps.${system} = {
          build = {
            type = "app";
            program = lib.getExe (pkgs.writeShellApplication {
              name = "build";
              runtimeInputs = site-dependencies;
              text = ''
                python bin/build.py
              '';
            });
          };
          watch = {
            type = "app";
            program = lib.getExe (pkgs.writeShellApplication {
              name = "watch";
              runtimeInputs = site-dependencies;
              text = ''
                python bin/build.py --watch
              '';
            });
          };
          publish = {
            type = "app";
            program = lib.getExe (pkgs.writeShellApplication {
              name = "publish";
              runtimeInputs = site-dependencies;
              text = builtins.readFile bin/publish;
            });
          };
          serve = {
            type = "app";
            program = lib.getExe (pkgs.writeShellApplication {
              name = "serve";
              runtimeInputs = site-dependencies;
              text = builtins.readFile bin/serve;
            });
          };
          update = {
            type = "app";
            program = lib.getExe (pkgs.writeShellApplication {
              name = "update";
              runtimeInputs = site-dependencies;
              text = builtins.readFile bin/update;
            });
          };
        };

        devShells.${system}.default = (pkgs.mkShellNoCC.override {
          # https://discourse.nixos.org/t/minimal-nix-shell/14373/3
          stdenv = pkgs.stdenvNoCC.override {
            initialPath = [ pkgs.coreutils ];
          };
        }) {
          packages = [
            pkgs.haskellPackages.pandoc-crossref
            pkgs.pandoc
            pkgs.pandoc-katex
            python'
          ];
        };
      }
    );
}
