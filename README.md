# Open Fine-Grained Activity Detection Scorer (OpenFAD Scorer)

## Overview

Official scorer release for OpenFAD 2022 _Activity Classification_ (AC) Task and
_Temporal Acticity Detection Task_ (TAD).  For evaluation information see
[https://openfad.nist.gov](OpenFAD Website).

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

## OpenFAD Scorer Usage Examples 

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

# License

AUTHORS
- Lukas Diduch

```
COPYRIGHT
---------

Full details can be found at: http://nist.gov/data/license.cfm

This software was developed at the National Institute of Standards and
Technology by employees of the Federal Government in the course of their
official duties.  Pursuant to Title 17 Section 105 of the United States Code
this software is not subject to copyright protection within the United States
and is in the public domain. This evaluation framework is an experimental
system.  NIST assumes no responsibility whatsoever for its use by any party,
and makes no guarantees, expressed or implied, about its quality, reliability,
or any other characteristic.

We would appreciate acknowledgement if the software is used.  This software can
be redistributed and/or modified freely provided that any derivative works bear
some notice that they are derived from it, and any modified versions bear some
notice that they have been modified.

THIS SOFTWARE IS PROVIDED "AS IS."  With regard to this software, NIST MAKES NO
EXPRESS OR IMPLIED WARRANTY AS TO ANY MATTER WHATSOEVER, INCLUDING
MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
```
