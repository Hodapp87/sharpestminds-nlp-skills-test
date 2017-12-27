{ pkgs ? import <nixpkgs> {} }:

let stdenv = pkgs.stdenv;
    optional = stdenv.lib.optional;
    # N.B. Jupyter and TensorFlow can only coexist on Python 3.6, at
    # least on latest nixpkgs.
    #tf = pkgs.python36Packages.tensorflowWithCuda;
    tf = pkgs.python36Packages.tensorflow;
    python_with_deps = pkgs.python36.withPackages
      (ps: [ps.scipy tf ps.matplotlib ps.pandas ps.scikitlearn ps.h5py
            ps.Keras
            ps.easydict ps.pillow ps.pyyaml
            ps.pyqt4 # Needed only for matplotlib backend
            ps.pycallgraph ps.graphviz
            ps.jupyter ps.gensim ps.nltk ps.requests
            ]);
in stdenv.mkDerivation rec {
  name = "sharpestminds-nlp-skill-test";

  buildInputs = with pkgs; [ python_with_deps ];
}

