import os, sys, pathlib
import pytest

os.chdir( pathlib.Path.cwd() / 'tests' )
pytest.main(sys.argv[1:])
