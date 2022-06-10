import numpy as np
import pandas as pd
import csv
from sklearn import metrics as skm
import json
import logging

from .datatypes import Dataset
from .io import *


# for aggregators, but FS/Format/FileIO stuff should not be here
import math
import h5py

log = logging.getLogger(__name__)

def compute_precision_recall(cm, labels):
    """ Given confusion-matrix and labels compute Precision & Recall of each class.

    :param matrix cm: Confusion Matrix
    :param list labels: List of Labels corresponding to __cm__
    :returns dataframe: __activity_id__, __precision__ (list), __recall__ (list)
    """
    sum_pred, sum_gt, precision, recall = cm.sum(axis=0), cm.sum(axis=1), [], []
    # CM is a square matrix
    for i in range(0, len(cm)):
        precision.append(cm[i,i]/sum_pred[i]) # TP/TP+FP
        recall.append(cm[i,i]/sum_gt[i]) # TP/TP+FN
    return(pd.DataFrame({'activity_id': labels, 'precision': precision, 'recall': recall}))

def score_pr(data):
    """ Precision/Recall from TP/FP/TN Counts across confusion matrix. 

    :param dataframe data: __activity_id_gt__ and __activity_id_pred__
    :returns list: __cm__ (matrix), __labels__ (list), __data__ (dataframe: _activity_id,precision,recall_)
    """
    # Compute confusion-matrix
    labels = data['activity_id_gt'].unique()
    cm = skm.confusion_matrix(data.activity_id_gt, data.activity_id_pred, labels=labels)
    # Compute Precision/Recall per class from confusion-matrix    
    spred = cm.sum(axis=0)
    sgt = cm.sum(axis=1)
    outp, outr = [], []    
    for i in range(0, len(cm)):
        outp.append(cm[i,i]/spred[i])
        outr.append(cm[i,i]/sgt[i])
    return cm, labels, pd.DataFrame({'activity_id': labels, 'precision':outp, 'recall':outr})

def score_sk(data):
    """Run SKLearn Classification Report Function

    :param dataframe data: __activity_id_gt__, __activity_id_pred__
    :returns object: sklearn report data
    """
    return skm.classification_report(data.activity_id_gt, data.activity_id_pred, output_dict=True)

def generate_zero_scores(labels):
    #import pdb
    #pdb.set_trace() 
    y = []
    if len(labels)>0:
        for i in labels:
            y.append([i, 0, 0, 0, 0, 0, [ 0.0, 0.0 ], [ 0.0, 0.0 ], [0.0], [] ])
    else:
        log.error("No matching activities found in system output.")
        y.append(['no_macthing_activity_placeholder', 0, 0, 0, 0, 0, [ 0.0, 0.0 ], [ 0.0, 0.0 ], [0.0], [] ])
    return pd.DataFrame(y, columns=['activity_id', 'ap', 'ap_interp', 'ap_11', 'ap_101', 'ap_auc', 'precision', 'recall', 'thresholds', 'ground_truth'])

# Rename to compute_pr_curve
def compute_precision_score(data, tad_mode = False):
    """ Multi-Class Precision/Recall Curves (computed across each class individually).

    > __Note__: prec,recall,thr are vectors with size varying on number of retrievals
    per refernce-class as non relevant classes have already been excluded.

    :param pd.dataframe data: __activity_id_gt__, __ground_truth__, __confidence_score__
    :returns pd.dataframe: Dataframe w/ following columns

    > - __activity_id__ : Activity Label 
    - __ap__ : Average precision score from sklearn (From docs: "This implementation is not
         interpolated and is different from computing the area under the
         precision-recall curve with the trapezoidal rule.")
    - __ap_interp__ : Interpolated AP over ALL thresholds
    - __ap_11__ : Interpolated AP over 11 equidistant thresholds
    - __ap_101__ : Interpolated AP over 101 equidistant thresholds
    - __ap_auc__ : Non-Interpolated AUC (integral)
    - __precision__ : [Array] w/ precision scores per activity
    - __recall__ : [Array] w/ recall scores per activity
    - __thresholds__ : [Array] w/ thresholds at which p/r is computed
    - __ground_thruth__ : [Array] w/ binary 1/0 labels from pred.

    """
    labels = data['activity_id_gt'].unique()    
    labels = np.append(labels, '__missed_detection__')
    #import pdb
    #pdb.set_trace()
    y = []
    for i in labels:
        if tad_mode == True:
            # Include only expected TP/FP
            subdata = data.loc[data.activity_id_gt == i]        
        else:        
            # Include expected TP/FP + hyp misdetection FP
            subdata = data.loc[(data.activity_id_gt == i) | (data.activity_id_pred == i)]
        y_gt = subdata.ground_truth
        y_pred = subdata.confidence_score
        #print(subdata)
        ap, ap_interp, ap_11, ap_101, ap_auc = 0, 0, 0, 0, 0
        # Handle no classes found (for robustness)
        if sum(y_gt) == 0:            
            precision, recall, thresholds = ([0., 0.], [0., 0], [0., 1.0])
        else:
            # NOTE: ap is mapped later to ap_avg !
            # TODO: rename ap to ap_weighted
            ap = skm.average_precision_score(y_gt, y_pred, average = "weighted")
            #print(y_gt)
            #print(y_pred)        
            precision, recall, thresholds = skm.precision_recall_curve(y_gt, y_pred)

            # Compute pinterp step-function using max
            pinterp = np.maximum.accumulate(precision)[::-1]

            # K-point interpolation (over all thresholds)
            #
            # skm adds a 1 at the end of the precision vector hence the [:-1] below
            # ap_interp = np.sum(np.diff(recall[::-1])[::-1] * pinterp[::-1][:-1])
            # For performance reaons let's use trapz though
            ap_interp = np.trapz(pinterp, recall[::-1])
            
            # N-Point interpolation (over N points interpolated @ thresholds)
            # Note: pinterp must be used here.
            ap_11 = np.mean(np.interp(np.linspace(0,1,11), recall[::-1], pinterp))
            ap_101 = np.mean(np.interp(np.linspace(0,1,101), recall[::-1], pinterp))

            # Using the max precision value per recall slot
            ap_auc = np.trapz(precision[::-1], recall[::-1])
        y.append([i, ap, ap_interp, ap_11, ap_101, ap_auc, precision, recall, thresholds, y_gt.values]) 
    return pd.DataFrame(y, columns=['activity_id', 'ap', 'ap_interp', 'ap_11', 'ap_101', 'ap_auc', 'precision', 'recall', 'thresholds', 'ground_truth'])

class MetricsValidationError(Exception):
    """Custom exception for error reporting."""
    pass

def compute_temp_iou(gstart, gend, pstart, pend):
    """ Compute __Temporal__ Intersection over Union (__IoU__) as defined in [1], Eq.(1) given start and endpoints of intervals __g__ and __p__.    

    :param float gstart: start first interval
    :param float gend: end first interval
    :param float pstart: start second interval
    :param float pend: end second interval
    :return float: IoU Value
    :raises MetricsValidationError: if input parameters are out of range
    """
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
    """ Given dataframe of predictions compute P/R Metrics using #compute_precision_score at a specific IoU threshold.

    :param dataframe mpred: scoring-ready hypothesis data.
    :param float iou_threshold: Intersection Over Union threshold used to subset `mpred` for scoring.
    :returns dataframe: See output of #compute_precision_score
    """
    if len(mpred) > 0:
        # MD is -1
        data = pd.DataFrame(mpred.loc[(mpred.alignment != '-1') & (mpred.IoU >= iou_threshold)])
        data.rename(columns={'alignment':'ground_truth', 'activity_id':'activity_id_gt'}, inplace=True)
        return compute_precision_score(data, tad_mode=True)
    else:
        return generate_zero_scores(['zero_score'])

def aggregate_xy(xy_list, method="average", average_resolution=500):
    """ Aggregate multiple xy arrays producing an y average incl. std-error.
        
    :param list xy_list: list of `[x,y]` arrays (x MUST be monotonically increasing !)
    :param str method: only 'average' method supported
    :param int average_resolution: number of interpolation points
    :returns list: Interpolated arrays of __precision__, __recall__, __stderr__.
    """
    if xy_list:
        # Filtering data with missing value
        is_valid = lambda dc: dc[0].size != 0 and dc[1].size != 0 and np.all(~np.isnan(dc[0])) and np.all(~np.isnan(dc[1]))        
        xy_list_filtered = [dc for dc in xy_list if is_valid(dc)]
        if xy_list_filtered:
            # Longest x axis
            max_fa_list = [max(dc[0]) for dc in xy_list_filtered]
            max_fa = max(max_fa_list)
            if method == "average":
                x = np.linspace(0, max_fa, average_resolution)
                ys = np.vstack([np.interp(x, data[0], data[1]) for data in xy_list_filtered])                
                stds = np.std(ys, axis=0, ddof=0)
                n = len(ys)
                stds = stds / math.sqrt(n)
                stds = 1.96 * stds
                # (0,1) (minpfa, 1)
                ys = [np.interp(x,
                                np.concatenate((np.array([0, data[0].min()]), data[0])),
                                np.concatenate((np.array([1, 1]),             data[1])))
                                for data in xy_list_filtered]
                aggregated_dc = [ x, (np.vstack(ys).sum(0) + len(xy_list) - len(xy_list_filtered)) / len(xy_list), stds ]
                return aggregated_dc
    log.error("Warning: No data remained after filtering, returning an empty array list")
    return [ [], [], [] ]

def aggregator(output_dir):
    """ Aggregate over all `ACTIVITY_*.h5` __files__ in output_dir using #aggregate_xy 
    :param str output_dir: Name of directory to look for h5-files.
    """
    files = get_activity_h5_files(output_dir)
    log.debug("Files to be aggregated: {}".format(files))
    f_list = [ h5py.File(f, 'r') for f in files ]    
    # dc['/prt'].attrs['activity_id'] 
    # [ [x[],y[]], ..., [x[],y[]] ]
    dlist = [ np.array([ dc['/prt/recall'][()], dc['/prt/precision'][()] ]) for dc in f_list ]
    aggregated_xy = aggregate_xy(dlist)
    output_fn = "{}/AGG_PRT_activities.h5".format(output_dir)
    write_aggregated_pr_as_hdf5(output_fn, aggregated_xy)    
    return None

def h5_aggregator(h5f):
    """ Aggregate over all activites within a h5 AC archive using #aggregate_xy
    Keep in mind that H5 Files have an access lock.    
    :param H5File h5f: H5 File Handle.
    :returns list: See output of #aggregate_xy
    """     
    # [ [x[],y[]], ..., [x[],y[]] ]
    activitiesG = h5f['activity']
    dlist = []
    for aName in activitiesG.keys():
        activityG = activitiesG[aName]        
        dlist.append(np.array([ activityG['prt/recall'][()], activityG['prt/precision'][()] ]))
    aggXYArr = aggregate_xy(dlist)    
    h5_add_aggregated_pr(h5f, aggXYArr)    
    return aggXYArr

def h5_iou_aggregator(h5f, iouThr):
    """ Aggregate over all IOU_ACTIVITY_*.h5 files in output_dir.

    :param H5File h5f: File handle
    :param float iouThr: IoU Threshold to find in the H5F archive. IoU entry must exist !    
    """
    # [ [x[],y[]], ..., [x[],y[]] ]
    activitiesG = h5f['activity']
    dlist = []
    for aName in activitiesG.keys():
        activityG = activitiesG[aName]
        if 'iou' in activityG.keys():
            gG = activityG['iou']
            if iouThr in gG.keys():
                iouG = gG[iouThr]
                dlist.append(np.array([ iouG['prt/recall'][()], iouG['prt/precision'][()] ]))
    if len(dlist):    
        aggXYArr = aggregate_xy(dlist)
        h5_add_aggregated_iou_pr(h5f, iouThr, aggXYArr)        
    return None      

def _sumup_ac_system_level_scores(metrics, pr_scores):
    """ Map internal to public representation. """
    co = []
    if 'map' in metrics:     co.append(['mAP',     round(np.mean(pr_scores.ap_interp), 3)]) # pinterp + AP
    if 'map_11' in metrics:  co.append(['mAP_11' , round(np.mean(pr_scores.ap_11), 3)])
    if 'map_101' in metrics: co.append(['mAP_101', round(np.mean(pr_scores.ap_101), 3)])
    if 'map_auc' in metrics: co.append(['mAP_auc', round(np.mean(pr_scores.ap_auc), 3)])
    if 'map_avg' in metrics: co.append(['mAP_avg', round(np.mean(pr_scores.ap), 3)])
    return co

def _sumup_ac_activity_level_scores(metrics, pr_scores):
    """ Map internal to public representation. """
    act = {}
    for index, row in pr_scores.iterrows():
        co = {}
        if 'map' in metrics:     co['ap'] =      round(row['ap_interp'], 3)
        if 'map_11' in metrics:  co['ap_11'] =   round(row['ap_11'], 3)
        if 'map_101' in metrics: co['ap_101'] =  round(row['ap_101'], 3)
        if 'map_auc' in metrics: co['ap_auc'] =  round(row['ap_auc'], 3)
        if 'map_avg' in metrics: co['ap_avg'] =  round(row['ap'], 3)
        act[row['activity_id']] = co
    return act

def _sumup_tad_system_level_scores(metrics, pr_iou_scores, iou_thresholds):
    """ Map internal to public representation. """
    ciou = {}
    for iout in iou_thresholds:
        pr_scores = pr_iou_scores[iout]
        co = {}
        if 'map'         in metrics: co['mAP'] = round(np.mean(pr_scores.ap), 3) # sklearn weighted AP method
        if 'map_11'      in metrics: co['mAP_11'] = round(np.mean(pr_scores.ap_11), 3)
        if 'map_101'     in metrics: co['mAP_101'] = round(np.mean(pr_scores.ap_101), 3)
        if 'map_auc'     in metrics: co['mAP_auc'] = round(np.mean(pr_scores.ap_auc), 3)
        if 'map_thr'     in metrics: co['mAP_thr'] = round(np.mean(pr_scores.ap_thr), 3)        
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
            if 'map'     in metrics: co[    "ap"] = round(row['ap'], 3) # sklearn weighted AP
            if 'map_11'  in metrics: co[ "ap_11"] = round(row['ap_11'], 3)
            if 'map_101' in metrics: co["ap_101"] = round(row['ap_101'], 3)
            if 'map_auc' in metrics: co["ap_auc"] = round(row['ap_auc'], 3)
            if 'map_thr' in metrics: co["ap_thr"] = round(row['ap_interp'], 3)
            activity = row['activity_id']
            if activity not in act.keys():
                act[activity] = {}
            act[activity][iout] = co
    return act
