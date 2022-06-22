import numpy as np
import pandas as pd
import csv
from sklearn import metrics as skm
import json
import logging
import sys
from .datatypes import Dataset
from .io import *

log = logging.getLogger(__name__)

class ValidationError(Exception):
    """Class for error reporting."""
    pass

def validate_gt(ds):
    """
    Validate Ground Truth Dataframe
    :params DataSet ds: Dataset w/ ref and hyp data
    :raises ValidationError: if GT data has duplicate entries
    """
    d_raw, d_dedup = ds.ref, ds.ref.drop_duplicates()
    l_raw, l_dedup = len(d_raw), len(d_dedup)
    if l_raw > l_dedup:
        log.debug("~~~ Duplicates List ~~~")
        diff_index = d_raw.index.difference(d_dedup.index)
        log.debug(d_raw.loc[diff_index].to_csv())
        raise ValidationError("Duplicates in Ground-Truth file detected. {} duplicates found.".format(l_raw-l_dedup))

def detect_missing_video_id(ds):
    """ Validate System Output for missing video-id against REF
    :params DataSet ds: Dataset w/ ref and hyp data    
    """
    ds.ref['video_file_id'] = pd.Categorical(ds.ref.video_file_id)
    ds.hyp['video_file_id'] = pd.Categorical(ds.hyp.video_file_id)
    ref_labels = ds.ref['video_file_id'].unique()
    hyp_labels = ds.hyp['video_file_id'].unique()
    label_distance = len(set(ref_labels) - set(hyp_labels))
    if label_distance > 0:
        log.warning("System output is missing {} video-file-id labels.".format(label_distance))
        # Past behavior
        # --:raises ValidationError: Hyp is missing video-id from REF
        # raise ValidationError("Missing video-file-id in system output.")

def detect_out_of_scope_hyp_video_id(ds):
    """ Validate System Output for video_id not present in reference data.
    :params DataSet ds: Dataset w/ ref and hyp data
    :raises ValidationError: Out-of-scope video-id detected
    """
    ref_labels = ds.ref['video_file_id'].unique()
    hyp_labels = ds.hyp['video_file_id'].unique()
    label_distance = len(set(hyp_labels) - set(ref_labels))
    if label_distance > 0:        
        out_of_scope_vid = ds.hyp[np.logical_not(ds.hyp.video_file_id.isin(ds.ref.video_file_id))] 
        oounique = out_of_scope_vid.video_file_id.unique()
        log.error("{} entries in system output using {} video-file-id".format(label_distance, len(oounique)))
        log.error("Out of scope video_file_id:")
        log.error(oounique)
        raise ValidationError("Unknown video-file-id in system output.")

def validate_pred(ds):
    """ Validate Prediction Dataframe.
    - Shows warning if PRED has more activity_id classes then GT
    :params DataSet ds: Dataset w/ ref and hyp data
    :raises ValidationError: Pred data contains labels outside of GT labels on relevant video_id's
    """
    gt_labels = ds.ref['activity_id'].unique()
    pred_labels = ds.hyp['activity_id'].unique()
        
    # check for unknown labels in pred
    label_distance = len(set(pred_labels) - set(gt_labels))
    if label_distance > 0:
        log.warning("System output contains {} extra activites not in ground-truth.".format(label_distance))        

    # Check for N/A activities
    hyp_na_sum = ds.hyp.activity_id.isna().sum()
    if hyp_na_sum > 0:
        log.warning("{} entries missing activity_id labels.".format(hyp_na_sum))

    # Create subset of vids from reference
    pred_vid_in_gt = ds.hyp[ds.hyp.video_file_id.isin(ds.ref.video_file_id.unique())]
    # Check MD's where PRED:activity-id does do not exist in GT:activity_id
    #if label_distance > 0:
    #    log.debug(pred_vid_in_gt[~pred_vid_in_gt.activity_id.isin(ds.ref.activity_id)].activity_id)
    matching_pred = pred_vid_in_gt[pred_vid_in_gt.activity_id.isin(ds.ref.activity_id.unique())]
    if len(matching_pred) != len(pred_vid_in_gt):
        log.warning("{} entries will be marked as missed !".format(len(pred_vid_in_gt) - len(matching_pred)))
    #    raise ValidationError("System output contains unknown activity_id. {} mismatched activities found."
    #        .format(len(pred_vid_in_gt) - len(matching_pred)))

def validate_ac(ds):
    """ Top-Level AC Validation Method 
    :params DataSet ds: Dataset w/ ref and hyp data    
    """    
    validate_gt(ds)
    detect_out_of_scope_hyp_video_id(ds)
    detect_missing_video_id(ds)
    validate_pred(ds)

def validate_ac_via_index(ds):
    """ Top-Level AC Validation Method 
    :params DataSet ds: Dataset w/ VALIDATION file (#video_id,frame_rate) and hyp data    
    """    
    detect_missing_video_id(ds)    

def validate_tad(ds):
    """ Top-Level TAD Validation Method
    :params DataSet ds: Dataset w/ ref and hyp data
    """    
    validate_gt(ds)
    detect_out_of_scope_hyp_video_id(ds)    
    validate_pred(ds)