# Open Fine-Grained Activity Detection Scorer (OpenFAD Scorer)

## Overview

Official scorer release for OpenFAD 2022 _Activity Classification_ (AC) Task
and _Temporal Acticity Detection Task_ (TAD).  For official evaluation
information, evaluation plan as well as documentation please visit the [OpenFAD
Website](https://openfad.nist.gov).

## Installing Python and Setuptools

NOTE: Please ensure to _call pip from your local python interpreter_ in order
to install the scorer package and dependecies in your _local_ python
environment by using `python -m pip` instead of `pip`.

- python 3.7.9
- setuptools: `python -m pip install setuptools`

### Installing FAD Scorer

Running install will also install dependent python packages.

```bash
python -m pip install .
```

After running the install above there should be a `fad-scorer` CLI tool in your
python environment.

### (Optional) Installing HDF5 Introspection Tools

The scorer uses HDF5 format for storing results. There are several tools to
introspect hdf5 files. In order to introspect HDF5 files on the Linux command
line install system level hdf5 tools using your package manager.  For example
for ubunut/debian based platforms use: `sudo apt-get install h5utils`

## Running Tests

```bash
python tests.py
```

## Usage Examples 

```bash
# Help
fad-scorer -h

# Activity Classification Task (AC Scorer)
fad-scorer score-ac -r tests/testdata/ac_ref_2x3.csv -y tests/testdata/ac_hyp_2x3_1fp.csv -o tmp

# - using verbose flag
fad-scorer -v score-ac -r tests/testdata/ac_ref_2x3.csv -y tests/testdata/ac_hyp_2x3_1fp.csv -o tmp


# Score Temporal Activity Detection Task (TAD Scorer)
fad-scorer score-tad -r tests/testdata/tad_ref_smoothcurve.csv -y tests/testdata/tad_hyp_smoothcurve.csv
```

# Authors

- 2021-2022 Lukas Diduch (lukas.diduch@nist.gov)

# Licensing Statement

See file: `LICENSE.txt`
