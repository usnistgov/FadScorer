import numpy as np
import pandas as pd
import logging
from .datatypes import Dataset
from .io import *
from .aggregators import *
import math
import h5py

def ap_interp(prec, rec):
    """ Interpolated AP - Based on VOCdevkit from VOC 2011.

    Parameters
    ----------
    prec: 1d-array
        Precision Values
    rec: 1d-array
        Recall Values    
    """
    mprec, mrec, idx = ap_interp_pr(prec, rec)
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def ap_interp_pr(prec, rec):
    """ Return Interpolated P/R curve - Based on VOCdevkit from VOC 2011.

    Parameters
    ----------
    prec: 1d-array
        Precision Values
    rec: 1d-array
        Recall Values
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    return mprec, mrec, idx

def fix_pr_tail(precision, recall):
    """ Fix precision value at highest recall if it's a MD to be 0. This helps
    to plot the end of the P/R graph correctly (step instead of a ramp).

    Parameters
    ----------
    precision: 1d-array
        Precision Values
    recall: 1d-array
        Recall Values        
    """
    prec = precision.copy()
    lval = True if prec[::-1][0] == 0 else False
    if lval:
        # Walk plot backwards and mark first !0 to 0
        for ridx in range(len(recall)-1, -1, -1):
            if lval & (prec[ridx] != 0):
                lval = False;
                prec[ridx] = 0;
    return prec, recall

def aggregate_xy(xy_list, method="average", average_resolution=500):
    """ Aggregate multiple xy arrays producing an y average including std-error.

    Parameters
    ----------
    xy_list: 2d-array
        list of `[x,y]` arrays (x MUST be monotonically increasing !)
    method: str
        only 'average' method supported
    average_resolution: int
        number of interpolation points (x-axis)
    
    Returns
    -------
    2d-array
        Interpolated arrays of *precision*, *recall*, *stderr*.
    """
    #pdb.set_trace()
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

def pr_curve_aggregator(h5f, activities=[]):
    """ Aggregate over all activites within a h5 AC archive using #aggregate_xy
    method.

    Parameters
    ----------
    h5f: H5File
        HF5 Container handle.
    activities: 1d-array [str]
        Activities to include in the plot

    Returns
    -------
    2d-array
        Aggregated [*precision* , *reccall*, *stderr*] array, see aggregate_xy
    """         
    activitiesG = h5f['activity']
    dlist = []
    for aName in activitiesG.keys():
        if len(activities) > 0:
            if aName not in activities:
                continue 
        activityG = activitiesG[aName]
        recl = activityG['prt/recall'][()][::-1]
        prec = activityG['prt/precision'][()][::-1]
        # [0,0] / empty arrays does not work well w/ interp algo.        
        #if not np.any(prec):
        #    continue        
        prec, recl, _ = ap_interp_pr(prec, recl)
        prec, recl = fix_pr_tail(prec, recl)        
        dlist.append(np.array([ recl, prec ]))    
    return aggregate_xy(dlist) if len(dlist) else []

def compute_aggregate_pr(h5f, activities = []):
    """ 
    Compute aggregated pr for AC Task.

    Parameters
    ----------
    h5f: H5File
        HF5 Container handle.
    activities: 1d-array
        List of activities to include
    """
    # [ [x[],y[]], ..., [x[],y[]] ]
    activitiesG = h5f['activity']
    dlist = []
    aggPR = pr_curve_aggregator(h5f)            
    h5_add_aggregated_pr(h5f, aggPR)                
    return None

def iou_pr_curve_aggregator(h5f, iouThr, activities = []):
    """ 
    Compute aggregate pr over all activites for specific temp. iou threshold.    

    Parameters
    ----------
    h5f: H5File
        HF5 Container handle.
    iouThr: float 
        temp. IoU Threshold to find in the H5F archive.

    Returns
    -------
    2d-array
        Aggregated [prec , rec, stderr] array, see aggregate_xy
    """    
    # [ [x[],y[]], ..., [x[],y[]] ]
    activitiesG = h5f['activity']
    dlist = []
    for idx, aName in enumerate(activitiesG.keys()):
        # Filter by activity-list entries when provided
        if len(activities) > 0:
            if aName not in activities:
                continue        
        activityG = activitiesG[aName]        
        if 'iou' in activityG.keys():
            gG = activityG['iou']
            if iouThr in gG.keys():
                iouG = gG[iouThr] 
                recl = iouG['prt/recall'][()][::-1]
                prec = iouG['prt/precision'][()][::-1]                
                # [0,0] / empty arrays does not work well w/ interp algo.
                if not np.any(prec):
                    continue
                prec, recl = fix_pr_tail(prec, recl)
                dlist.append(np.array([ recl , prec ]))    
    return aggregate_xy(dlist) if len(dlist) else []
    
def compute_aggregate_iou_pr(h5f, activities = []):
    """ Compute aggregated pr for all temp. IoU

    Parameters
    ----------
    h5f: H5File
        HF5 Container handle.
    activities: [1d-array]
        List of activities to include
    """
    # [ [x[],y[]], ..., [x[],y[]] ]
    activitiesG = h5f['activity']
    dlist = []
    # Just access once to get all IoU levels from the first activity
    for aName in activitiesG.keys():
        activityG = activitiesG[aName]
        gG = activityG['iou']        
        for tiou in gG.keys():    
            aggXYArr = iou_pr_curve_aggregator(h5f, tiou, activities)
            h5_add_aggregated_iou_pr(h5f, tiou, aggXYArr)
        break      
    return None