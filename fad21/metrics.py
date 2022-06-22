import numpy as np
import pandas as pd
import logging
from .datatypes import Dataset
from .io import *
from .aggregators import *

# for aggregators, but FS/Format/FileIO stuff should not be here
import math
import h5py

log = logging.getLogger(__name__)

class MetricsValidationError(Exception):
    """Custom exception for error reporting."""
    pass

def generate_zero_scores(labels):
    y = []
    if len(labels)>0:
        for i in labels:
            y.append([i, 0, 0, 0, 0, 0, [ 0.0, 0.0 ], [ 0.0, 0.0 ], [0.0], [] ])
    else:
        log.error("No matching activities found in system output.")
        y.append(['no_macthing_activity', 0, 0, 0, 0, 0, [ 0.0, 0.0 ], [ 0.0, 0.0 ], [0.0] ])
    return pd.DataFrame(y, columns=['activity_id', 'ap', 'ap_interp', 'precision', 'recall'])

def compute_multiclass_pr(data, no_clamp = False):    
    activities = data.activity_id_ref.unique()

    # Don't do this, as it will always gloss over threshold details potentially
    # mapping them onto a single point in P/R space. This can result in a high
    # loss of detail in extreme cases where there is lots of support points
    # within the thershold interval:
    # trange = np.linspace(1, 0, num=101, retstep=True)[0]
    # trange = np.array([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0])
    #
    # To create a smaller threshold range in case of an excessive amount of thresholds, 
    # and preserve more detail it is better to reduce using available thresholds.

    y = []    
    
    # Iterate over all activity-id isolating them into a binary detection
    for act in activities:
        # Include only relevant TP/FP (all relevant retrievals)
        subsel = (data.activity_id_ref == act)
        subdata = data.loc[subsel]        
                
        # Count and drop missed detections
        mlen = len(subdata.loc[subdata.activity_id_hyp == '0'])
        subdata = subdata.drop(subdata[subdata.activity_id_hyp == '0'].index)

        # There is at least 1 datapoint + MD
        if len(subdata) > 0:
            # Use all thresholds available.
            trange = np.sort(subdata['confidence_score'].unique())
            if trange[0] != 0.0:
                trange = np.append(0.0, trange)
            if trange[-1] != 1.0:
                trange = np.append(trange, 1.0)

            # Reverse the range to be always running from high thr to low thr to
            # preserve order of p/r. This is needed to handle edgecases w/ 1 or 2
            # points.
            trange = trange[::-1]            
            alabels = [act, '0']        
            precision,recall = [],[]        

            # computing p/r for each threshold
            for thr in trange:        
                tdata = subdata[(subdata.confidence_score >= thr)]         
                tp, fp, fn = cm_binary(tdata.activity_id_ref, tdata.activity_id_hyp, act)
                # all retrievals above threshold
                fp += fn             
                # retrievals below threshold + md as they are always excluded above
                fn = len(subdata) - len(tdata) + mlen
                
                # counting missed detections in
                # if (mlen > 0) & (thr == 0): fp += mlen                

                # Clamp to 1 (on by default)
                if (no_clamp == False) & (thr == 1.0):
                    prec = 1.0 if (tp+fp == 0) else tp/(tp+fp)
                else:
                    prec = tp/(tp+fp) if (tp+fp >0) else 0.0            
                
                # account for MD's: clamp to 0 for correct aP computation
                if (mlen > 0) & (thr == 0): prec = 0
                recl = tp/(tp+fn) if (tp+fn >0) else 0.0
                    
                #print("act {}, thr {}, tp {}, fp {}, fn {}, p/r {}/{}, lent {}, ldata {}, mdata {}".format(
                #   act, thr, tp, fp, fn, prec, recl, len(tdata), len(subdata), mlen))            
                precision.append(prec)
                recall.append(recl)
        # One MD, No datapoints.
        else:
            precision = [0.0, 0.0]
            recall = [0.0, 1.0]
            trange = []

        # Build output
        y.append([ act, precision, recall, trange ])
    return pd.DataFrame(y, columns=['activity_id', 'precision', 'recall', 'thresholds' ])

def compute_temp_iou(gstart, gend, pstart, pend):
    """ Compute __Temporal__ Intersection over Union (__IoU__) as defined in [1], Eq.(1) given start and endpoints of intervals __g__ and __p__.    

    :param float gstart: start first interval
    :param float gend: end first interval
    :param float pstart: start second interval
    :param float pend: end second interval
    :return float: IoU Value
    :raises MetricsValidationError: if input parameters are out of range
    """

    # Workaround for Gitlab-CI version
    pd.isna = pd.isnull

    # Validate for edge cases
    if (gstart > gend) or (pstart > pend):
        raise MetricsValidationError("Start frame after End frame or End frame before Start frame ! FIX SYSTEM OUTPUT !")

    if pd.isna(gstart) or pd.isna(gend):
        raise MetricsValidationError("NaN not allowed for system frame output ! FIX SYSTEM OUTPUT !")

    # If NaN set to 0 (not needed, data is supposedly clean)
    # if pd.isna(pstart):
    #     pstart = 0
    # if pd.isna(pend):
    #     pend = 0;

    
    s0 = min(gstart, pstart)    
    spoint = max(gstart-s0, pstart-s0) 
    epoint = min(gend-s0, pend-s0)    
    try:
        # Area of overlap (normalized over GT area)
        aoo = max(0, (epoint - spoint) / (gend-gstart))    
        # Area of union
        aou = (max(gend, pend) - min(gstart, pstart)) / (gend-gstart)
        iou = aoo/aou
    except:
        iou = 0
    #log.debug("[{},{}]:[{},{}] aoo: {}, aou: {}, iou: {}".format(
    #    gstart, pstart, gend, pend, aoo, aou, iou))
    return round(iou,3)

def compute_pr_scores_at_iou(mpred, iou_threshold):
    """ Given dataframe of predictions compute P/R Metrics using
    'compute_precision_score' at a specific IoU threshold.

    :param dataframe mpred: scoring-ready hypothesis data.
    :param float iou_threshold: Intersection Over Union threshold used to subset
        `mpred` for scoring.
        
    :returns dataframe: See output of #compute_precision_score
    """
    if len(mpred) > 0:
        # MD is -1
        data = pd.DataFrame(mpred.loc[(mpred.alignment != '-1') & (mpred.IoU >= iou_threshold)])
        data.rename(columns={'alignment':'ground_truth', 'activity_id':'activity_id_ref'}, inplace=True)
        return compute_precision_score(data, tad_mode=True)
    else:
        return generate_zero_scores(['zero_score'])

def ac_system_level_score(metrics, pr_scores):
    """ Map internal to public representation. """
    co = []
    if 'map'        in metrics: co.append(['mAP',     round(np.mean(pr_scores.ap), 4)])
    if 'map_interp' in metrics: co.append(['mAP_interp', round(np.mean(pr_scores.ap_interp), 4)])
    return co

def ac_activity_level_scores(metrics, pr_scores):
    """ Map internal to public representation. """
    act = {}
    for index, row in pr_scores.iterrows():
        co = {}
        if 'map'        in metrics:        co['ap'] = round(row['ap'], 4)
        if 'map_interp' in metrics: co['ap_interp'] = round(row['ap_interp'], 4)
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

def cm_binary(refl_df, hypl_df, activity):
    tp,fp,fn = 0,0,0
    refl = refl_df.to_numpy()
    hypl = hypl_df.to_numpy()    
    if len(refl) > 0:
        if len(refl) != len(hypl):
            raise "ref activity len != hyp activity len"
        for i in range(0, len(refl)):
            if (refl[i] == activity):
                if (hypl[i] == activity):
                    tp += 1
                else:
                    fp += 1
            else:
                if (hypl[i] == activity):
                    fn += 1
    return tp,fp,fn

# For aP computation not graph
def rectify_pr_curves(results):
    return results.apply(_rectify_pr_curve, axis=1)

def compute_aps(results):    
    results = results.apply(_compute_ap, axis=1)    
    return results    

#def _rectify_pr_curve(row, at_start=True, at_end=True):
def _rectify_pr_curve(row, at_start=False, at_end=False):
    prec,recl = row.precision.copy(), row.recall.copy()
    if not np.array(row.precision).any():
            prec, recl = [0.0,0.0], [0.0, 1.0]
    else:
        # When all thresholds are mapped to the same point in P/R space
        if (np.sum(np.diff(prec)) == 0) & (np.sum(np.diff(recl)) == 0):
            recl[0] = 0.0            
            recl[-1] = 1.0
        # When there is MD's @end or no support at start and recall points are
        # missing at 1 or 0.
        if (recl[0] != 0.0):
            recl.insert(0, 0.0)
            prec.insert(0, row.precision[0])
        if (recl[-1] != 1.0):
            recl.append(1.0)
            prec.append(row.precision[-1])            
    row.precision = prec
    row.recall = recl
    return row

def _compute_ap(row):
    p, r = row.precision, row.recall
    # Compute pinterp step-function using max. Needs to be run backwards due to
    # properties of precision.
    pinterp = np.maximum.accumulate(p[::-1])
    # K-point interpolation (over all thresholds) 
    row['ap'] = np.trapz(p, r)
    # This will be highly optimistic
    row['ap_interp'] = np.sum(np.diff(r[::]) * pinterp[::-1][:-1])
    return row
