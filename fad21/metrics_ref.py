import numpy as np
import pandas as pd
import logging
from .datatypes import Dataset

log = logging.getLogger(__name__)
#_mdbg = False
_mdbg = True

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """             
    mprec = np.hstack([[0], prec, [0]])                                                                                                                        
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:                                                                                                                      
        mprec[i] = max(mprec[i], mprec[i + 1])                                                                                                                 
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1                                                                                                             
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])                                                                                                      
    return ap

def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as true
    positive. This code is greatly inspired by Pascal VOC devkit / ActivityNET.
    Does not handle 'no-score' regions (empty ref for segment).

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances. Required fields:
        ['video_file_id', 'frame_start', 'frame_end']
    prediction : df
        Data frame containing the prediction instances. Required fields:
        ['video_file_id, 'frame_start', 'frame_end', 'confidence_score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        ap, prec, recl = np.zeros(len(tiou_thresholds)), [], []
        for idx, iout in enumerate(tiou_thresholds):
            prec.append([0.0, 0.0])
            recl.append([0.0, 0.1])
        return ap, prec, recl        

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['confidence_score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    with np.printoptions(threshold=np.inf):
        ref = ground_truth
        hyp = prediction
        #print("REF: {}/{} (unique), HYP: {}/{} (unique)".format(len(ref), 
        #    len(ref.video_file_id.unique()), len(hyp), len(hyp.video_file_id.unique())))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video_file_id')
    plen = len(prediction)
    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():
        #if _mdbg:
        #    print("({}/{}) {}, {}".format(idx, plen, this_pred['video_file_id'], this_pred['activity_id']))
        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video_file_id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['frame_start', 'frame_end']].values,
                               this_gt[['frame_start', 'frame_end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        #print(this_gt.activity_id.unique(), tiou_arr)
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                # If first one is < thr
                if float(tiou_arr[jdx]) < float(tiou_thr):
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    if _mdbg:
        with np.printoptions(threshold=np.inf):
            for tidx, tiou_thr in enumerate(tiou_thresholds):
                
                hyp['tp'] = tp[tidx].astype(int)
                hyp['fp'] = fp[tidx].astype(int)
                print(hyp.to_string())
                
                print(hyp.loc[hyp.video_file_id == 'C059E6DE-F99E-460A-9361-39C9DC77CEBF'].to_string())
                print("REFIMPL: {} TP: {}/{}, FP: {}/{}".format(tiou_thr, np.sum(tp[tidx]), len(ref), np.sum(fp[tidx]), len(hyp)))
                print(hyp.tp.values)
                
    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)

    recall_cumsum = tp_cumsum / npos
    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    #if _mdbg:
    #    print("tp {}, fp {}, p/r {}/{}".format(tp_cumsum, fp_cumsum, precision_cumsum, recall_cumsum))    
    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap, precision_cumsum, recall_cumsum

def segment_iou(target_segment, candidate_segments):
    """    
    Compute the temporal intersection over union between a target segment and
    all the test segments. This code is greatly inspired by Pascal VOC devkit /
    ActivityNET.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU