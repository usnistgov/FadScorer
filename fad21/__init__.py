"""
Python package `fad21` for scoring.
.. include:: ../README.md
"""

from .scoring import *
from .metrics import *
from .scoring import score_ac, score_tad
from .validation import validate_gt, validate_pred, validate_ac, validate_tad
from .validation import detect_missing_video_id, detect_out_of_scope_hyp_video_id

from .io import *
from .datatypes import *
from .plot import *