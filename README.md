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

Full details can be found at: http://nist.gov/data/license.cfm

```
NIST-developed software is provided by NIST as a public service. You may use,
copy, and distribute copies of the software in any medium, provided that you
keep intact this entire notice. You may improve, modify, and create derivative
works of the software or any portion of the software, and you may copy and
distribute such modifications or works. Modified works should carry a notice
stating that you changed the software and should note the date and nature of
any such change. Please explicitly acknowledge the National Institute of
Standards and Technology as the source of the software. 

NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY
OF ANY KIND, EXPRESS, IMPLIED, IN FACT, OR ARISING BY OPERATION OF LAW,
INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, AND DATA ACCURACY. NIST NEITHER
REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE
UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES
NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR
THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY,
RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

You are solely responsible for determining the appropriateness of using and
distributing the software and you assume all risks associated with its use,
including but not limited to the risks and costs of program errors, compliance
with applicable laws, damage to or loss of data, programs or equipment, and the
unavailability or interruption of operation. This software is not intended to
be used in any situation where a failure could cause risk of injury or damage
to property. The software developed by NIST employees is not subject to
copyright protection within the United States.
```
