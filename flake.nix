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
        inherit (self.packages.${system}.default)
          filter
          julia'
          python';
        linters = [ pkgs.validator-nu pkgs.lychee ];
        node-packages = [ pkgs.nodejs ];
        site-builders = [ julia' python' ];
      in
      {
        packages.${system} = let inherit (pkgs) callPackage; in {
          default = callPackage ./pkgs/cgdct-moe { };
        };

        formatter.${system} = pkgs.writeShellScriptBin "prettier" ''
          npx prettier --write "$@"
        '';

        checks.${system}.lint = pkgs.stdenvNoCC.mkDerivation {
          name = "lint";
          src = ./.;
          doCheck = true;
          nativeCheckInputs = linters ++ node-packages ++ site-builders;
          checkPhase = ''
            source .envrc || true
            prettier --check .
            source ./bin/build
            source ./bin/vnu __site
          '';
          installPhase = "touch $out";
        };

        apps.${system} = {
          publish = {
            type = "app";
            program = "${lib.getExe (pkgs.writeShellApplication {
              name = "publish";
              runtimeInputs = node-packages ++ site-builders;
              text = builtins.readFile bin/publish;
            })}";
          };
          serve = {
            type = "app";
            program = "${lib.getExe (pkgs.writeShellApplication {
              name = "serve";
              runtimeInputs = site-builders;
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
              runtimeInputs = node-packages;
              text = builtins.readFile bin/update;
            })}";
          };
        };
      }
    );
}
