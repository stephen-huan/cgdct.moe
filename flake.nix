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
        site = self.packages.${system}.default;
        site-dependencies = site.nativeBuildInputs;
      in
      {
        packages.${system} = let inherit (pkgs) callPackage; in rec {
          default = callPackage ./pkgs { cgdct-moe = default; };
        };

        formatter.${system} = pkgs.writeShellScriptBin "formatter" ''
          prettier --write "$@"
          ${lib.getExe pkgs.isort} "$@"
          ${lib.getExe pkgs.black} "$@"
          ${lib.getExe pkgs.shfmt} --write "$@"
          ${lib.getExe pkgs.nixpkgs-fmt} "$@"
        '';

        checks.${system}.lint = pkgs.stdenvNoCC.mkDerivation {
          name = "lint";
          src = ./.;
          doCheck = true;
          nativeCheckInputs = site-dependencies;
          checkPhase = ''
            prettier --check .
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
      }
    );
}
