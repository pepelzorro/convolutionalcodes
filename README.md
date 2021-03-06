# Building blocks for convolutional codes in [[Amaranth HDL]]

This repository contains implementations for convolutional encoders and decoders
in [[Amaranth HDL]]. There is an (albeit trivial) encoder, and a r=1/2 viterbi
decoder.

## Getting started.

Install dependencies (in a python3 virtual environment)

```
pip install numpy scipy jupyterlab matplotlib amaranth
```

To install an editable version of this repository for development, use:

```
pip install -e [path to checkout]
```

This will allow you to depend on this package from another project and edit this
package without installing it again.

## Documentation / Performance

Performance plots can be found in a jupyter notebook:

- [Viterbi-BER](notebooks/Viterbi-BER.ipynb) - Viterbi decoding with hard and
  soft symbols

![Block Diagram of r=1/2 Viterbi Decoder](notebooks/r2_viterbi.png)

## Running tests

### Unit Tests

These are self-contained tests using the pysim backend

```
python -m unittest discover -t . -s convolutionalcodes -p "*py" -k unit -v
```

To generate waves, run the tests with the `GENERATE_VCDS` environment variable
set.

### CXXRTL Models

These tests generate CXXRTL binaries that are then used for the performance
plots

```
python -m convolutionalcodes.viterbi_decode -k cxx -v
```

## License

[2-Clause BSD License](LICENSE)

[Amaranth HDL]: https://github.com/amaranth-lang/amaranth
