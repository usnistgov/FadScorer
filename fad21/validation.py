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

def tad_add_noscore_region(ds):
    """ De-NaN GT and PRED (should ideally not have nan activity_id)
    Parameters
    ----------
    ds: Dataset object
        Reference and Hypothesis dataframes.    
    """    
    # Check for NaN Activity Id and remove. Throw a warning now, might be an exception later.
    gtnan = ds.ref.isna().activity_id
    gtnanl = len(gtnan.loc[gtnan == True])
    if gtnanl > 0:
        log.warning("Reference contains {} no-score regions.".format(gtnanl))
        gtnan = ds.ref.isna().activity_id
        ds.ref.loc[gtnan, 'activity_id']= "NO_SCORE_REGION"

    prednan = ds.hyp.isna().activity_id
    prednanl = len(prednan.loc[prednan == True])
    if prednanl > 0:
        log.warning("NaN activity_id in system-output detected. Dropping {} NaN entries".format(prednanl))
        #raise MetricsValidationError("NaN Activity-Id in system output !")
        ds.hyp.dropna(inplace=True)
    # Check for NaN - END

def remove_out_of_scope_activities(ds):
    """ If there are any activity-id which are out of scope or NA, whole entry is
    removed from hypothesis. 

    Parameters
    ----------
    ds: Dataset object
        Reference and Hypothesis dataframes.

    """
    # ref.activity_id will already include NO_SCORE_REGION activity 
    ds.hyp.drop(ds.hyp[~ds.hyp.activity_id.isin(ds.ref.activity_id.unique())].index, inplace = True)
    # Usecase: video_id,,,
    ds.hyp.drop(ds.hyp[ds.hyp.activity_id.isna()].index, inplace = True)
    
    # If not using no-score region:
    # ds.ref.drop(ds.ref[ds.ref.activity_id.isna()].index, inplace = True)    

def validate_gt(ds):
    """ Reference Validation

    Parameters
    ----------
    ds: Dataset object
        Reference and Hypothesis dataframes.

    Exceptions
    ----------
    ValidationError:
        If reference data has duplicate entries
    """
    d_raw, d_dedup = ds.ref, ds.ref.drop_duplicates()
    l_raw, l_dedup = len(d_raw), len(d_dedup)
    if l_raw > l_dedup:
        log.debug("- Duplicates List")
        diff_index = d_raw.index.difference(d_dedup.index)
        log.debug(d_raw.loc[diff_index].to_csv())
        raise ValidationError("Duplicates in Ground-Truth file detected. {} duplicates found.".format(l_raw-l_dedup))

def detect_missing_video_id(ds):
    """ Validate System Output for missing video-id against reference.

    Parameters
    ----------
    ds: Dataset object
        Reference and Hypothesis dataframes.
    """
    ds.ref['video_file_id'] = pd.Categorical(ds.ref.video_file_id)
    ds.hyp['video_file_id'] = pd.Categorical(ds.hyp.video_file_id)
    ref_labels = ds.ref['video_file_id'].unique()
    hyp_labels = ds.hyp['video_file_id'].unique()
    label_distance = len(set(ref_labels) - set(hyp_labels))
    if label_distance > 0:
        log.warning("System output is missing {} video-file-id labels.".format(label_distance))

def detect_out_of_scope_hyp_video_id(ds):
    """ Validate System Output for video_id not present in reference data.
    :params DataSet ds: Dataset w/ ref and hyp data
    Parameters
    ----------    
    ds: DataSet
        Dataset w/ ref and hyp data

    Exceptions
    ----------
    ValidationError:
        Out-of-scope video-id detected
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
    """ Validate Hypothesis.

    Parameters
    ----------    
    ds: Dataset object
        Reference and Hypothesis dataframes.
    """
    gt_labels = ds.ref['activity_id'].unique()
    pred_labels = ds.hyp['activity_id'].unique()
        
    # check for unknown labels in pred
    label_distance = len(set(pred_labels) - set(gt_labels))
    if 'nan' in pred_labels:
        print("Nan detetced")
        print(set(pred_labels))
    if label_distance > 0:
        log.warning("System output contains {} extra activites not in ground-truth.".format(label_distance))

    # Check for N/A activities
    hyp_na_sum = ds.hyp.activity_id.isna().sum()
    if hyp_na_sum > 0:
        log.warning("{} entries missing activity_id labels.".format(hyp_na_sum))

    # Create subset of vids from reference
    pred_vid_in_gt = ds.hyp[ds.hyp.video_file_id.isin(ds.ref.video_file_id.unique())]
    
    # Check MD's where PRED:activity-id does do not exist in GT:activity_id
    matching_pred = pred_vid_in_gt[pred_vid_in_gt.activity_id.isin(ds.ref.activity_id.unique())]
    if len(matching_pred) != len(pred_vid_in_gt):
        log.warning("{} entries will be marked as missed.".format(len(pred_vid_in_gt) - len(matching_pred)))
    
def validate_ac(ds):
    """ AC Validation Wrapper

    Parameters
    ----------    
    ds: Dataset object
        Reference and Hypothesis dataframes.
    """    
    validate_gt(ds)
    detect_out_of_scope_hyp_video_id(ds)
    detect_missing_video_id(ds)
    validate_pred(ds) 

def validate_tad(ds):
    """ TAD Validation Wrapper

    Parameters
    ----------    
    ds: Dataset object
        Reference and Hypothesis dataframes.
    """    
    validate_gt(ds)
    detect_out_of_scope_hyp_video_id(ds)    
    validate_pred(ds)

def process_subset_args(args, ds):
    """ Apply activity-id and video-id inclusion-only filters to dataset
    reducing ref and hyp to a subset.    
    
    Parameters
    ----------
    args: argparse args
        Argument object from CLI with activity_list_file: str
            List of activities to include
        video_list_file: str
            List of video-id to include            
    ds: Dataset object
        Reference and Hypothesis dataframes.
    """
    if args.activity_list_file:        
        raw_al = load_list_file(args.activity_list_file)
        activity_list = list(filter(None, raw_al))
        log.info("Using {} activity-id from '{}' activities-file.".format(len(activity_list), args.activity_list_file))
        #log.debug(activity_list)
        # Exclude all classes but include relevant video-id in reference        
        ds.hyp = ds.hyp.loc[ds.hyp.activity_id.isin(activity_list)]
        ds.ref = ds.ref.loc[ds.ref.activity_id.isin(activity_list) | ds.ref.activity_id.isna()]
    if args.video_list_file:
        raw_vl = load_list_file(args.video_list_file)
        video_list = list(filter(None, raw_vl))
        log.info("Using {} video-id from '{}' video-id-file.".format(len(video_list), args.video_list_file))
        log.debug(video_list)
        ds.ref = ds.ref.loc[ds.ref.video_file_id.isin(video_list)]
        ds.hyp = ds.hyp.loc[ds.hyp.video_file_id.isin(video_list)]
    log.debug(ds)