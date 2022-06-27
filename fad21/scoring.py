# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import csv
from sklearn import metrics as skm
import json
import logging
import os

from .datatypes import Dataset
from .io import *
from .filters import *
from .validation import detect_out_of_scope_hyp_video_id
from .metrics import *

log = logging.getLogger(__name__)

def score_ac(ds, metrics=['map'], filter_top_n=0, output_dir=None, argstr = "{}", no_clamp = False):
    """ Score System output (hypothesis) of Activity Classification Task (AC) incl.
    - no activity labels
    - missing video-id
    
    :param fad21.Dataset ds  : Dataset Object w/ REF + HYP
    :param list[str] metrics : Array of metrics to include ['map', 'map_interp']
    :param int filter_top_n  : Subselect by top confidence scores before scoring
    :param str output_dir    : Path to a directory (created on demand) for output files
    :returns tuple: __pr_scores__, __results__, __al_results__
    
    > - pr_scores: multi-class pr for all activities
    > - results     metrics for system level
    > - al_results  metrics for activity level
    """
    # Safety check in case this is called from somewhere else than main.
    detect_out_of_scope_hyp_video_id(ds)

    # Fix out of scope and NA's
    remove_out_of_scope_activities(ds)
 
    # Remove out of bound activities
    fhyp = ds.hyp[ds.hyp.activity_id.isin(ds.activity_ids)]
    if len(fhyp) != len(ds.hyp):
        log.info("[xform] {} reference activities matching {} in system output.".
            format(len(fhyp.activity_id.unique()), len(ds.hyp.activity_id.unique())))
    ds.hyp = fhyp

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
        aggPR = pr_curve_aggregator(h5f, interp=False)        
        aggPRinterp = pr_curve_aggregator(h5f, interp=True)        
        h5_add_aggregated_pr(h5f, aggPR, interp=False)            
        h5_add_aggregated_pr(h5f, aggPRinterp, interp=True)    
        h5f.close()

    return pr_scores, results, al_results    
  
def score_tad(ds, metrics=['map'], iou_thresholds=[0.5], output_dir=None, argstr = "{}"):    
    """ Score System output (hypothesis) of Temporal Activity Detection Task (TAD)
    
    :param fad21.Dataset ds:          Dataset Object w/ REF + HYP
    :param list[str] metrics:     Array of metrics to include
    :param list[float] iou_threshold: List of IoU Thresholds to use
    :param str output_dir:  Path to a directory (created on demand) for output files    
    :returns tuple: __pr_scores__, __results__, __al_results__
    
    > - pr_iou_scores: multi-class pr for all activities and iou
    - results     metrics for system level
    - al_results  metrics for activity level
    """
    mpred = prep_tad_data(ds)
    pr_iou_scores = {}
    for iout in iou_thresholds:        
        pr_iou_scores[iout] = compute_pr_scores_at_iou(mpred, iout)  
    results = _sumup_tad_system_level_scores(metrics, pr_iou_scores, iou_thresholds)
    al_results = _sumup_tad_activity_level_scores(metrics, pr_iou_scores, iou_thresholds)
    if (output_dir != None):
        fn = os.path.join(output_dir, "scoring_results.h5")
        wipe_scoring_file(fn)
        if len(mpred) > 0:
            h5_add_alignment(fn, mpred)
        h5f = h5_create_archive(fn, 'a')
        if len(mpred) > 0:
            h5f['alignments'].attrs['ftype'] = "LExtra"
        h5_add_info(h5f, argstr, "TAD")        
        h5_add_iou_system_scores(h5f, results)         
        h5_add_iou_activity_scores(h5f, al_results)
        activities = pd.unique(ds.ref.activity_id)
        h5_add_iou_activity_prt(h5f, pr_iou_scores, iou_thresholds, activities)
        if len(mpred) > 0:
            for iou_thr in iou_thresholds:
                h5_iou_aggregator(h5f, "{}".format(iou_thr))
        h5f.close()    
        
    return pr_iou_scores, results, al_results

def ac_system_level_score(metrics, pr_scores):
    """ Map internal to public representation. """
    co = []
    if 'map'        in metrics: co.append(['mAP',     round(np.mean(pr_scores.ap), 4)])
    #if 'map_interp' in metrics: co.append(['mAP_interp', round(np.mean(pr_scores.ap_interp), 4)])
    return co

def ac_activity_level_scores(metrics, pr_scores):
    """ Map internal to public representation. """
    act = {}
    for index, row in pr_scores.iterrows():
        co = {}
        if 'map'        in metrics:        co['ap'] = round(row['ap'], 4)
     #   if 'map_interp' in metrics: co['ap_interp'] = round(row['ap_interp'], 4)
        act[row['activity_id']] = co
    return act

def _sumup_tad_system_level_scores(metrics, pr_iou_scores, iou_thresholds):
    """ Map internal to public representation. """
    ciou = {}
    for iout in iou_thresholds:
        pr_scores = pr_iou_scores[iout]
        co = {}
        if 'map'         in metrics: co['mAP']        = round(np.mean(pr_scores.ap), 3)
        if 'map_interp'  in metrics: co['mAP_interp'] = round(np.mean(pr_scores.ap_interp), 3)
        ciou[iout] = co
    return ciou
        
def _sumup_tad_activity_level_scores(metrics, pr_iou_scores, iou_thresholds):
    """ Map internal to public representation. Scores per Class and IoU Level """
    metrics = metrics    
    act = {}    
    for iout in iou_thresholds:        
        prs = pr_iou_scores[iout]        
        for index, row in prs.iterrows():            
            co = {}
            if 'map'         in metrics: co[        "ap"] = round(row['ap'], 3)
            if 'map_interp'  in metrics: co[ "ap_interp"] = round(row['ap_interp'], 3)
            activity = row['activity_id']
            if activity not in act.keys():
                act[activity] = {}
            act[activity][iout] = co
    return act