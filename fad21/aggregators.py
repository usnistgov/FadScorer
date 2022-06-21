import numpy as np
import pandas as pd
import logging
from .datatypes import Dataset
from .io import *
from .aggregators import *
import math
import h5py

def aggregate_xy(xy_list, method="average", average_resolution=10):
    """ Aggregate multiple xy arrays producing an y average incl. std-error.
        
    :param list xy_list: list of `[x,y]` arrays (x MUST be monotonically increasing !)
    :param str method: only 'average' method supported
    :param int average_resolution: number of interpolation points
    :returns list: Interpolated arrays of __precision__, __recall__, __stderr__.
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