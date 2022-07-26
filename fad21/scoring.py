# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import csv
import json
import logging
import os

from .datatypes import Dataset
from .io import *
from .validation import detect_out_of_scope_hyp_video_id,tad_add_noscore_region, remove_out_of_scope_activities
from .metrics import *

log = logging.getLogger(__name__)  

def score_ac(ds, metrics=['map'], filter_top_n=0, output_dir=None, argstr = "{}", no_clamp = False):
    """ Score System output of Activity Classification Task (AC)

    Parameters
    ----------
    ds: Dataset
        Dataset Object w/ REF + HYP
    metrics: list [str] 
        Array of metrics to include. Currently only ['map'] is supported.
    filter_top_n: int
        DEPRECATED
    output_dir: str 
        Path to a directory (created on demand) for output files

    Output
    ------
    Tuple with followin values:
        - **pr_scores** (??)
            multi-class pr for all activities
        - **results** (??)
            metrics for system level
        - **al_results**: ??
            metrics for activity level
    """
    # Safety check in case this is called from somewhere else than main.
    detect_out_of_scope_hyp_video_id(ds)
    # Fix out of scope and NA's
    remove_out_of_scope_activities(ds) 
    
    # Handle empty files/content
    if len(ds.hyp) > 0:
        pr_scores = compute_multiclass_pr(ds)
    else:
        pr_scores = generate_zero_scores(ds)
    
    results    = ac_system_level_score(metrics, pr_scores)
    al_results = ac_activity_level_scores(metrics, pr_scores)
    
    # Writeout if not running from a notebook/not output specified
    if (output_dir != None):
        fn = os.path.join(output_dir, "scoring_results.h5")
        wipe_scoring_file(fn)
        h5f = h5_create_archive(fn, 'w')
        h5_add_info(h5f, argstr, "AC")
        h5_add_system_scores(h5f, results)
        h5_add_activity_scores(h5f, al_results)
        h5_add_activity_prt(h5f, pr_scores)
        h5f.close()

    return pr_scores, results, al_results    
  
def score_tad(ds, iou_thresholds, metrics=['map'], output_dir=None, nb_jobs = -1, argstr = "{}"):
    """ Score System output of Temporal Activity Detection Task (TAD)
    
    Parameters
    ----------
    ds: Dataset
        Dataset Object w/ REF + HYP
    iou_thresholds: list[float]
        List of IoU Thresholds to use
    metrics: list[str] 
        Array of metrics to include
    output_dir: str
        Path to a directory (created on demand) for output files    
    
    Returns
    -------
    Tuple with following values:
        - **pr_iou_scores** (dict of df)
            multi-class pr for all activities and iou
        - **results** (df)
            metrics for system level
        - **al_results** (df)
            metrics for activity level
    """    

    # FIXME: Use a No score-region parameter
    tad_add_noscore_region(ds)
    # Fix out of scope and NA's
    remove_out_of_scope_activities(ds) 
    
    if len(ds.hyp) > 0:
        pr_iou_scores = compute_multiclass_iou_pr(ds, iou_thresholds, nb_jobs)
    else:
        pr_iou_scores = {}
        [ pr_iou_scores.setdefault(iout, generate_zero_scores(ds)) for iout in iou_thresholds ]    

    results = _sumup_tad_system_level_scores(metrics, pr_iou_scores, iou_thresholds)
    al_results = _sumup_tad_activity_level_scores(metrics, pr_iou_scores, iou_thresholds)

    if (output_dir != None):
        fn = os.path.join(output_dir, "scoring_results.h5")
        wipe_scoring_file(fn)
        h5f = h5_create_archive(fn, 'a')
        h5_add_info(h5f, argstr, "TAD") 
        h5_add_iou_system_scores(h5f, results)    
        h5_add_iou_activity_scores(h5f, al_results)
        activities = pd.unique(ds.ref.activity_id)
        h5_add_iou_activity_prt(h5f, pr_iou_scores, iou_thresholds, activities)
        h5f.close()        
    return pr_iou_scores, results, al_results

def ac_system_level_score(metrics, pr_scores):
    """ Map internal scores to a standartized output format.
    Parameters
    ----------
    metrics: list [str]
        List of metrics
    pr_scores: DataFrame
        DataFrame w/ scores per activity.
    """    
    co = []
    if 'map'        in metrics: co.append(['mAP',     round(np.mean(pr_scores.ap), 4)])    
    return co

def ac_activity_level_scores(metrics, pr_scores):
    """ Map internal scores to a standartized output format.
    Parameters
    ----------
    metrics: list [str]
        List of metrics
    pr_scores: DataFrame
        DataFrame w/ scores per activity.
    """        
    act = {}
    for index, row in pr_scores.iterrows():
        co = {}
        if 'map'        in metrics:        co['AP'] = round(row['ap'], 4)     
        act[row['activity_id']] = co
    return act

def _sumup_tad_system_level_scores(metrics, pr_iou_scores, iou_thresholds):
    """ Map internal scores to a standartized output format.
    Parameters
    ----------
    metrics: list[str]
        List of metrics
    pr_iou_scores: dict[Dataframe]
        Dictionary of DataFrames w/ iout as keys
    iou_thresholds: list [float]
        List of thresholds
    """    
    ciou = {}
    for iout in iou_thresholds:
        pr_scores = pr_iou_scores[iout]
        co = {}
        if 'map'         in metrics: co['mAP']        = round(np.mean(pr_scores.ap), 4)        
        ciou[iout] = co
    return ciou
        
def _sumup_tad_activity_level_scores(metrics, pr_iou_scores, iou_thresholds):
    """ Map internal scores to a standartized output format.
    Parameters
    ----------
    metrics: list[str]
        List of metrics
    pr_iou_scores: dict[Dataframe]
        Dictionary of DataFrames w/ iout as keys
    iou_thresholds: list [float]
        List of thresholds
    """    
    metrics = metrics    
    act = {}    
    for iout in iou_thresholds:        
        prs = pr_iou_scores[iout]        
        for index, row in prs.iterrows():            
            co = {}
            if 'map'         in metrics: co[        "AP"] = round(row['ap'], 4)
            activity = row['activity_id']
            if activity not in act.keys():
                act[activity] = {}
            act[activity][iout] = co
    return act