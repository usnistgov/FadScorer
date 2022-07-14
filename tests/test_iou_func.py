import fad21
from fad21.scoring import score_tad
from fad21.validation import validate_tad,tad_add_noscore_region, remove_out_of_scope_activities
from fad21.io import *
from fad21.metrics import segment_iou
from fad21.datatypes import Dataset
from pathlib import Path
import numpy as np
import pytest
import io
import collections
import logging
log = logging.getLogger(__name__)

# Validate TAD scorer-implementation w/ edge-cases

def get_root():
    return Path(__file__).parent

def load_data(refFile, hypFile, act):
    """ Loader Func """
    ds = Dataset(load_tad_ref(get_root()/refFile), load_tad_hyp(get_root()/hypFile))
    # Apply scorer pre-processing
    tad_add_noscore_region(ds)    
    remove_out_of_scope_activities(ds)
    ref=ds.ref.loc[(ds.ref.activity_id == act) | (ds.ref.activity_id == 'NO_SCORE_REGION')].reset_index(drop=True)
    hyp=ds.hyp.loc[(ds.hyp.activity_id == act)].reset_index(drop=True)
    return ref, hyp

def test_temp_iou_100():
    ref, hyp = load_data('testdata/tad_ref_smoothcurve.csv', 'testdata/tad_hyp_smoothcurve.csv', "Closing")    
    for idx, myhyp in hyp.iterrows():        
        ious = segment_iou(myhyp.frame_start, myhyp.frame_end, [ref.frame_start, ref.frame_end])
        assert(ious[idx] == 1.0)        

def test_temp_iou_vari():
    ref, hyp = load_data('testdata/tad_ref_smoothcurve.csv', 'testdata/tad_hyp_smoothcurve.csv', "Closing-Temp-IOU")    
    vresult=[1.0, 0.8, 0.6,0.4,0.2,0.1,0.05,0.01]
    for idx, myhyp in hyp.iterrows():        
        ious = segment_iou(myhyp.frame_start, myhyp.frame_end, [ref.frame_start, ref.frame_end])
        assert(ious[idx] == vresult[idx])