import numpy as np
import pandas as pd
import logging
from .datatypes import Dataset
from .io import *
from .aggregators import *
from time import time
from joblib import Parallel, delayed
import sys

log = logging.getLogger(__name__)

def generate_zero_scores(labels):
    y = []
    if len(labels)>0:
        for i in labels:
            y.append( [ i, 0, [ 0.0, 0.0 ], [ 0.0, 0.0 ] ])
    else:
        log.error("No matching activities found in system output.")
        y.append( [ 'no_macthing_activity', 0, [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]) 
    return pd.DataFrame(y, columns=['activity_id', 'ap', 'precision', 'recall'])

#
# Wrappers
#
def compute_multiclass_pr(ds):
    y, activities = [], ds.ref.activity_id.unique()
    # Iterate over all activity-id treating them as a binary detection
    for act in activities:
        # Include only relevant TP/FP
        refs = ds.ref.loc[(ds.ref.activity_id == act)].reset_index(drop=True)
        hyps = ds.hyp.loc[(ds.hyp.activity_id == act)].reset_index(drop=True)        
        ap, prec, recl = compute_average_precision_ac(refs, hyps)
        y.append([ act, prec, recl, ap ])
    return pd.DataFrame(y, columns=['activity_id', 'precision', 'recall', 'ap' ])

def compute_multiclass_iou_pr(ds, iou_thresholds=np.linspace(0.5, 0.95, 10), nb_jobs=-1):
    """ Given dataframe of predictions compute P/R Metrics using
    'compute_average_precision_tad' at a specific IoU threshold.

    :param dataframe mpred: scoring-ready hypothesis data.
    :param float iou_threshold: Intersection Over Union threshold used to subset
        `mpred` for scoring.
        
    :returns dataframe: See output of #compute_precision_score
    """
    # Initialize
    scores = {}
    [ scores.setdefault(iout, []) for iout in iou_thresholds ]
    
    # Iterate over all activity-id treating them as a binary detection
    alist = ds.ref.loc[ds.ref.activity_id.str.contains('NO_SCORE_REGION')==False].activity_id.unique()        

    apScores = Parallel(n_jobs=nb_jobs)(delayed(compute_average_precision_tad)(
            ref=ds.ref.loc[(ds.ref.activity_id == act) | (ds.ref.activity_id == 'NO_SCORE_REGION')].reset_index(drop=True),                        
            hyp=ds.hyp.loc[(ds.hyp.activity_id == act)].reset_index(drop=True),
            iou_thresholds=iou_thresholds) for idx, act in enumerate(alist))

    for idx, act in enumerate(alist):
        for iout in iou_thresholds:            
            scores[iout].append([act, apScores[idx][iout][0], apScores[idx][iout][1], apScores[idx][iout][2]])        
        
    # Build results for all            
    pr_scores = {}
    for iout in iou_thresholds: 
        pr_scores[iout] = pd.DataFrame(scores[iout], columns = ['activity_id', 'ap', 'precision', 'recall'])

    return pr_scores

#
# Metrics
#        

def compute_average_precision_ac(ref, hyp):
    """ Compute average precision and precision-recall curve (AC Task / ranked
    retrieval) between ground truth and predictions data frames. If multiple
    predictions occur for the same predicted segment, only the one with highest
    score is matched as true positive. Parts of this code are inspired by Pascal
    VOC devkit/ActivityNET.
    
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances. Required fields:
        ['video_file_id', 'activity_id']
    prediction : df
        Data frame containing the prediction instances. Required fields:
        ['video_file_id', 'activity_id', 'confidence_score']

    Outputs
    -------
    ap : float
        Average precision score (using p_interp).
    precision : 1darray
        Precision values
    recall : 1darray
        Recall values
    """
    npos = len(ref)
    sys.stdout.write('.')  
    sys.stdout.flush()  
    
    # Compute video matches (1 row per hyp, MD activity_id will be NaN)
    mhyp = ref.merge(hyp, how='right', on="video_file_id", suffixes=['_ref', '_hyp'])
    
    if len(mhyp) == 0:
        prec = [0.0, 0.0]
        rec = [0.0, 1.0]                
    else:        
        mhyp[['tp', 'fp']] = [0, 0]        
        mhyp.loc[mhyp['activity_id_ref'] == mhyp['activity_id_hyp'], 'tp'] = 1
        mhyp.loc[mhyp['activity_id_ref'] != mhyp['activity_id_hyp'], 'fp'] = 1    
        # Double Sorting to reduce random noise @ same confidence levels.
        mhyp.sort_values(["confidence_score", "activity_id_hyp"], ascending=False, inplace=True)
        # Make sure only highest ref is matched against the video as TP
        nhyp = mhyp.duplicated(subset = ['video_file_id', 'tp'], keep='first')        
        mhyp.loc[mhyp.loc[nhyp == True].index, ['tp', 'fp']] = [ 0, 1 ]
        #print(mhyp)
        #pdb.set_trace()
        tp = np.cumsum(mhyp.tp).astype(float)
        fp = np.cumsum(mhyp.fp).astype(float)                      
        rec = (tp / npos).values
        prec = (tp / (tp + fp)).values    
    return ap_interp(prec, rec), prec, rec

def compute_ious(row, ref):
    refs = ref.loc[ ref['video_file_id'] == row.video_file_id ].copy()    
    # If there are no references for this hypothesis it's IoU is 0/FP
    if len(refs) == 0:
        return pd.DataFrame(data=[[row.activity_id, row.video_file_id, np.nan, np.nan, row.frame_start, row.frame_end, row.confidence_score, 0.0]],
            columns=['activity_id', 'video_file_id', 'frame_start_ref', 'frame_end_ref', 'frame_start_hyp', 'frame_end_hyp', 'confidence_score', 'IoU'])
    else:        
        refs['IoU'] = segment_iou(row.frame_start, row.frame_end, [refs.frame_start, refs.frame_end])
        rmax = refs.loc[refs.IoU == refs.IoU.max()]
        rout = rmax.rename(columns={'frame_start':'frame_start_ref', 'frame_end':'frame_end_ref'})
        rout[['frame_start_hyp', 'frame_end_hyp', 'confidence_score']] = row.frame_start, row.frame_end, row.confidence_score
        return rout

def compute_average_precision_tad(ref, hyp, iou_thresholds=np.linspace(0.5, 0.95, 10)):
    """ 
    Compute average precision and precision-recall curve at specific IoU
    thresholds between ground truth and predictions data frames. If multiple
    predictions occur for the same predicted segment, only the one with highest
    tIoU is matched as true positive. Activities which are missed in referece
    are treated as a no-score-region and excluded from computation. Parts of
    this code are inspired by Pascal VOC devkit/ActivityNET.
    
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances. Required fields:
        ['video_file_id', 'activity_id']
    prediction : df
        Data frame containing the prediction instances. Required fields:
        ['video_file_id', 'activity_id', 'confidence_score']
    iou_thresholds : 1darray, optional
        Temporal IoU Threshold (>=0)        

    Outputs
    -------
    ap : float
        Average precision score.
    precision : 1darray
        Precision values
    recall : 1darray
        Recall values
    """
        
    # REF has same amount of !score_regions for all runs, which need to be
    # excluded from overall REF count.
    npos = len(ref.loc[ref.activity_id.str.contains('NO_SCORE_REGION')==False])    
    output, out = {}, []

    # No activity found.
    if hyp.empty:
        for iout in iou_thresholds:
            output[iout] = 0.0, [0.0, 0.0], [0.0, 1.0]
        return output
 
    # Compute IoU for all hyps incl. NO_SCORE_REGION
    for idx, myhyp in hyp.iterrows():
        out.append(compute_ious(myhyp, ref))
    ihyp = pd.concat(out)

    # Exclude NO_SCORE_REGIONs but keep FP NA's
    ihyp = ihyp.loc[(ihyp.activity_id.str.contains('NO_SCORE_REGION') == False) | ihyp.frame_start_ref.isna()]        

    # Sort by confidence score
    ihyp.sort_values(["confidence_score"], ascending=False, inplace=True)        
    ihyp.reset_index(inplace=True, drop=True)        
        
    # Determine TP/FP @ IoU-Threshold
    for iout in iou_thresholds:        
        ihyp[['tp', 'fp']] = [ 0, 1 ]        
        ihyp.loc[~ihyp['frame_start_ref'].isna() & (ihyp['IoU'] >= iout), ['tp', 'fp']] = [ 1, 0 ]
        # Mark TP as FP for duplicate ref matches at lower CS
        nhyp = ihyp.duplicated(subset = ['video_file_id', 'frame_start_ref', 'frame_end_ref', 'tp'], keep='first')
        ihyp.loc[ihyp.loc[nhyp == True].index, ['tp', 'fp']] = [ 0, 1 ]        
        tp = np.cumsum(ihyp.tp).astype(float)
        fp = np.cumsum(ihyp.fp).astype(float)                      
        rec = (tp / npos).values
        prec = (tp / (tp + fp)).values
        output[iout] = ap_interp(prec, rec), prec, rec    
    return output

def segment_iou(ref_start, ref_end, tgts):
    """
    Compute __Temporal__ Intersection over Union (__IoU__) as defined in 
    [1], Eq.(1) given start and endpoints of intervals __g__ and __p__.    
    Vectorized impl. from ActivityNET/Pascal VOC devkit.

    Parameters
    ----------
    ref_start: float
        starting frame of source segement
    ref_end: float
        end frame of source segement        
    tgts : 2d array
        Temporal test segments containing [starting x N, ending X N] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(ref_start, tgts[0])
    tt2 = np.minimum(ref_end, tgts[1])    
    # Segment intersection including Non-negative overlap score
    inter = (tt2 - tt1).clip(0)    
    # Segment union.
    union = (tgts[1] - tgts[0]) + (ref_end - ref_start) - inter    
    tIoU = inter.astype(float) / union
    return tIoU
