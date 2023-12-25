{
  description = "dotfiles";

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
        generated = import ./default.nix { inherit pkgs system; };
        inherit (generated) nodeDependencies;
        node-env =
          "${nixpkgs.outPath}/pkgs/development/node-packages/node-env.nix";
        python' = pkgs.python3.withPackages (ps: with ps; [
          beautifulsoup4
          lxml
        ]);
        linters = [ pkgs.validator-nu pkgs.lychee ];
        node-packages = [ pkgs.nodejs pkgs.node2nix nodeDependencies ];
        site-builders = [ pkgs.julia-bin python' ];
      in
      {
        formatter.${system} = pkgs.writeShellScriptBin "prettier" ''
          ${nodeDependencies}/bin/prettier --write "$@"
        '';
        checks.${system}.lint = pkgs.stdenvNoCC.mkDerivation {
          name = "lint";
          src = ./.;
          doCheck = true;
          nativeCheckInputs = linters ++ node-packages ++ site-builders;
          checkPhase = ''
            export NODE_PATH=${nodeDependencies}/lib/node_modules
            source .envrc || true
            prettier --check .
            source ./bin/build
            source ./bin/vnu __site
          '';
          installPhase = "touch $out";
        };
        apps.${system} = {
          build = {
            type = "app";
            program = "${lib.getExe (pkgs.writeShellApplication {
              name = "build";
              runtimeInputs = node-packages ++ site-builders;
              text = builtins.readFile bin/build;
            })}";
          };
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
              text = builtins.readFile bin/serve;
            })}";
          };
          update = {
            type = "app";
            program = "${lib.getExe (pkgs.writeShellApplication {
              name = "update";
              runtimeInputs = node-packages ++ lib.singleton pkgs.poetry;
              text = builtins.readFile bin/update;
            })}";
          };
        };
        devShells.${system}.default = (pkgs.mkShellNoCC.override {
          stdenv = pkgs.stdenvNoCC.override {
            initialPath = [ pkgs.coreutils ];
          };
        }) {
          packages = linters
            ++ node-packages
            ++ site-builders;
          shellHook = ''
            # clear nodejs and node2nix from $NODE_PATH
            export NODE_PATH=${nodeDependencies}/lib/node_modules
            ln -sf "${node-env}" node-env.nix
          '';
        };
      }
    );
}
