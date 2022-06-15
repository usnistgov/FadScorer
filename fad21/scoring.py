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
from .metrics import _sumup_ac_system_level_scores, _sumup_ac_activity_level_scores, _sumup_tad_activity_level_scores, _sumup_tad_system_level_scores

log = logging.getLogger(__name__)

def score_ac(ds, metrics=['map'], filter_top_n=0, output_dir=None, argstr = "{}"):
    """ Score System output (hypothesis) of Activity Classification Task (AC)
    
    :param fad21.Dataset ds:          Dataset Object w/ REF + HYP
    :param list[str] metrics:     Array of metrics to include
    :param int topk:        Subselect topk matches
    :param str output_dir:  Path to a directory (created on demand) for output files    
    :returns tuple: __pr_scores__, __results__, __al_results__
    
    > - pr_scores: multi-class pr for all activities
    - results     metrics for system level
    - al_results  metrics for activity level
    """
    # Just a safety check in case this is called from somewhere else than main.
    detect_out_of_scope_hyp_video_id(ds)

    # Sequence matters here !

    # Fix out of scope and  NA's
    remove_out_of_scope_activities(ds)
    # Subselect by confidence scores
    ds.hyp = filter_by_top_k_confidence(ds.hyp,filter_top_n)
    # Fix missing entries (adds __missed_detection__ label)
    append_missing_video_id(ds)
    # Rectify 
    prep_ac_data(ds)
    data = ds.register
    
    #cm, labels, pr_stats = score_pr(data)
    if len(data) > 0:
        pr_scores = compute_precision_score(data)
    else:
        pr_scores = generate_zero_scores(ds.register)

    results = _sumup_ac_system_level_scores(metrics, pr_scores)
    al_results = _sumup_ac_activity_level_scores(metrics, pr_scores)

    # Writeout if not running from a notebook/not output specified
    if (output_dir != None):
        fn = os.path.join(output_dir, "scoring_results.h5")
        wipe_scoring_file(fn)
        h5f = h5_create_archive(fn, 'w')
        h5_add_info(h5f, argstr, "AC")
        h5_add_system_scores(h5f, results)
        h5_add_activity_scores(h5f, al_results)
        h5_add_activity_prt(h5f, pr_scores)
        h5_aggregator(h5f)
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