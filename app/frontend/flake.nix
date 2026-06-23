{
  description = "NestJS development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: {
    devShells = {
      x86_64-linux =
        let
          pkgs = import nixpkgs { system = "x86_64-linux"; };
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.nodejs_22
              pkgs.yarn
              pkgs.pnpm
              # pkgs.nodePackages_latest."@nestjs/cli"
            ];
            shellHook = ''
              echo "Node enviroment loaded"
            '';
          };
        };
    };
  };
}
