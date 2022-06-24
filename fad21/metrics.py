import numpy as np
import pandas as pd
import logging
from .datatypes import Dataset
from .io import *
from .aggregators import *

log = logging.getLogger(__name__)

class MetricsValidationError(Exception):
    """Custom exception for error reporting."""
    pass

def generate_zero_scores(labels):
    y = []
    if len(labels)>0:
        for i in labels:
            y.append( [ i, 0, [ 0.0, 0.0 ], [ 0.0, 0.0 ] ])
    else:
        log.error("No matching activities found in system output.")
        y.append( [ 'no_macthing_activity', 0, [ 0.0, 0.0 ], [ 0.0, 0.0 ] ]) 
    return pd.DataFrame(y, columns=['activity_id', 'ap', 'precision', 'recall'])

def compute_multiclass_pr(ds):
    activities = ds.ref.activity_id.unique()
    y = []

    # Iterate over all activity-id isolating them into a binary detection
    for act in activities:
        # Include only relevant TP/FP (looking only at finding relevant retrievals)
        refs = ds.ref.loc[(ds.ref.activity_id == act)].reset_index(drop=True)
        hyps = ds.hyp.loc[(ds.hyp.activity_id == act)].reset_index(drop=True)
        ap, prec, recl = compute_average_precision_classification(refs, hyps)
        y.append([ act, prec, recl, ap ])

    return pd.DataFrame(y, columns=['activity_id', 'precision', 'recall', 'ap' ])

def ap_interp(prec, rec):
    """Interpolated AP - Based on VOCdevkit from VOC 2011.
    """
    mprec, mrec, idx = ap_interp_pr(prec, rec)
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def ap_interp_pr(prec, rec):
    """Return Interpolated P/R curve
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    return mprec, mrec, idx

def compute_average_precision_classification(ground_truth, prediction):
    """ Compute average precision (classification task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matched as true
    positive. This code is greatly inspired by Pascal VOC devkit and
    ActivityNET.

    Note: This method has ordering issues when confidence-scores are the same.

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
        Average precision score.
    precision : np.array[float]
        Precision values
    recall : np.array[float]
        Recall values
    """
    npos = float(len(ground_truth))
    lock_gt = np.ones(len(ground_truth)) * -1

    # Sort predictions by decreasing score order. This is problematic when
    # confidence scores are the same. Order will start to matter.
    sort_idx = prediction['confidence_score'].values.argsort()[::-1]
    #pdb.set_trace()
    
    # Fixes some issues w/ same confidence score and sort order.
    # sort_idx = np.lexsort((prediction['video_file_id'].values, prediction['confidence_score'].values))

    prediction = prediction.loc[sort_idx].reset_index(drop=True)
    #print(prediction)

    # Initialize true positive and false positive vectors.
    tp = np.zeros(len(prediction))
    fp = np.zeros(len(prediction))

    # Assigning TP / FP
    for idx in range(len(prediction)):
        this_pred = prediction.loc[idx]
        gt_idx = ground_truth['video_file_id'] == this_pred['video_file_id']

        # At least one video matching ground truth
        if not gt_idx.any():
            fp[idx] = 1
            continue
        this_gt = ground_truth.loc[gt_idx].reset_index()
        if lock_gt[this_gt['index']] >= 0:     
            fp[idx] = 1
        else:
            tp[idx] = 1
            lock_gt[this_gt['index']] = idx

    # Computing prec-rec
    tp = np.cumsum(tp).astype(np.float)
    fp = np.cumsum(fp).astype(np.float)
    rec = tp / npos
    prec = tp / (tp + fp)
    return ap_interp(prec, rec), prec, rec

def compute_temp_iou(gstart, gend, pstart, pend):
    """ Compute __Temporal__ Intersection over Union (__IoU__) as defined in [1], Eq.(1) given
    start and endpoints of intervals __g__ and __p__.    

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

def _rectify_pr_curve(row, at_start=False, at_end=False):
    prec,recl = row.precision.copy(), row.recall.copy()
    if not np.array(row.precision).any():
            prec, recl = [0.0,0.0], [0.0, 1.0]
    else:
        # When all thresholds are mapped to one point in P/R space, extend
        # recall range to edges.
        if (np.sum(np.diff(prec)) == 0) & (np.sum(np.diff(recl)) == 0):
            log.waring("only one point ?")
            recl[0] = 0.0
            recl[-1] = 1.0
        # When there support at start/end extend and recall points are
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