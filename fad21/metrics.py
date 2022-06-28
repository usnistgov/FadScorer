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

#
# Wrappers
#

def compute_multiclass_pr(ds):
    activities = ds.ref.activity_id.unique()
    y = []

    # Iterate over all activity-id isolating them into a binary detection
    for act in activities:
        # Include only relevant TP/FP (looking only at finding relevant retrievals)
        refs = ds.ref.loc[(ds.ref.activity_id == act)].reset_index(drop=True)
        hyps = ds.hyp.loc[(ds.hyp.activity_id == act)].reset_index(drop=True)
        #ap, prec, recl = _ref_compute_average_precision_classification(refs, hyps)        
        ap, prec, recl = compute_average_precision_ac(refs, hyps)
        y.append([ act, prec, recl, ap ])

    return pd.DataFrame(y, columns=['activity_id', 'precision', 'recall', 'ap' ])


#
# Metrics
#        

def compute_average_precision_ac(ref, hyp):
    """ Compute average precision (AC Task) between ground truth and
    predictions data frames. If multiple predictions occur for the same
    predicted segment, only the one with highest score is matched as true
    positive. This code is inspired by Pascal VOC devkit and ActivityNET but
    uses sets.

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
    npos = len(ref)
    # Remove bottom duplicate retrievals for same vid-aid
    hyp.drop_duplicates(subset=['video_file_id'], keep='first', inplace=True)
    # FP + MD will result in NaN for confidence score or activity_id pair, used
    # to mark them later
    subdata = ref.merge(hyp, how='outer', on="video_file_id", suffixes=['_ref', '_hyp'])
    subdata.loc[subdata['confidence_score'].isna(), 'confidence_score'] = -1 # Mark MD
    subdata[['tp', 'fp']] = [0, 0]
    subdata.loc[subdata['confidence_score'] == -1, 'fp'] = 1 # Account for MD
    subdata.loc[subdata['activity_id_ref'] == subdata['activity_id_hyp'], 'tp'] = 1 # Account for TP
    subdata.loc[subdata['activity_id_ref'] != subdata['activity_id_hyp'], 'fp'] = 1 # Account for FP
    
    if len(subdata) == 0:
        prec = [0.0, 0.0]
        rec = [0.0, 1.0]                
    else:
        # Rectify some random noise in extreme cases introduced when sorting
        # would occur only by confifence_score w/ identical values.
        subdata.sort_values(["activity_id_hyp", "confidence_score"], ascending=False, inplace=True)
        #subdata.sort_values(["confidence_score"], ascending=False, inplace=True)         
        tp = np.cumsum(subdata.tp).astype(float)
        fp = np.cumsum(subdata.fp).astype(float)                      
        rec = (tp / npos).values
        prec = (tp / (tp + fp)).values
    #print("act {}, tp {}, fp {}, p/r {}/{}".format(act, tp, fp, prec, rec))    
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