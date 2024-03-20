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
        inherit (site) filter julia';
        site-dependencies = site.nativeBuildInputs;
      in
      {
        packages.${system} = let inherit (pkgs) callPackage; in {
          default = callPackage ./pkgs/cgdct-moe { };
        };

        formatter.${system} = pkgs.writeShellScriptBin "formatter" ''
          npx prettier --write "$@"
          ${julia'}/bin/julia --eval "using JuliaFormatter; format(\"$1\")"
          ${lib.getExe pkgs.isort} "$@"
          ${lib.getExe pkgs.black} "$@"
          ${lib.getExe pkgs.shfmt} --write "$@"
          ${lib.getExe pkgs.nixpkgs-fmt} "$@"
        '';

        checks.${system}.lint = pkgs.buildNpmPackage {
          name = "lint";
          src = ./.;
          inherit (site) npmDepsHash;
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
          publish = {
            type = "app";
            program = "${lib.getExe (pkgs.writeShellApplication {
              name = "publish";
              runtimeInputs = site-dependencies;
              text = builtins.readFile bin/publish;
            })}";
          };
          serve = {
            type = "app";
            program = "${lib.getExe (pkgs.writeShellApplication {
              name = "serve";
              runtimeInputs = site-dependencies;
              text = ''
                NODE="$(which node)"
                export NODE
                # https://unix.stackexchange.com/a/7080
                julia --optimize=0 --color=yes --eval "
                  using Franklin: serve; serve(prerender=true)
                " 2> >("${lib.getExe filter}" >&2)
              '';
            })}";
          };
          update = {
            type = "app";
            program = "${lib.getExe (pkgs.writeShellApplication {
              name = "update";
              runtimeInputs = site-dependencies;
              text = builtins.readFile bin/update;
            })}";
          };
        };
      }
    );
}
