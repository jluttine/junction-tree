let
  pkgs = import <nixpkgs> {};

  python = pkgs.python3Full;
  pythonPackages = pkgs.python3Packages;

  versioneer = pythonPackages.buildPythonPackage rec {
    name = "${pname}-${version}";
    pname = "versioneer";
    version = "0.18";
    src = pythonPackages.fetchPypi {
      inherit pname version;
      sha256 = "0dgkzg1r7mjg91xp81sv9z4mabyxl39pkd11jlc1200md20zglga";
    };
  };

in
pythonPackages.buildPythonPackage rec {
  name = "junctionTree";
  checkPhase = "true"; # Skip unit tests..
  src = ./.;
  depsBuildBuild = with pythonPackages; [
    # Console
    jupyter
    ipython
    pytest
    sphinx
    pip
  ];
  buildInputs = with pythonPackages; [

    # Core
    python
    numpy
    attrs

    # Tests
    networkx

    versioneer
  ];
}
