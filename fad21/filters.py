# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import logging
from .datatypes import Dataset
from .io import *
from .metrics import *

import pdb

log = logging.getLogger(__name__)

def prep_ac_data(ds):
    """ Produce 'Dataset.register' entry w/ aligned and cleaned up ref-hypothesis matches

    Tidy-dataframe w/ VideoID, ref and hyp activity labels and Confidence score.

    Parameters
    ----------
    ds: DataSet object, containing GT and PRED
    """    
    
    # Subset hypdata only for matching activities with reference.activity_id
    prednum = len(ds.hyp.activity_id.unique())    
   

    # Remove out of bound activities (this should be catched before by validation step)
    fhyp = ds.hyp[ds.hyp.activity_id.isin(ds.activity_ids)]    
    log.info("[xform] {} matching activities (refXhyp) out of {} activities in hypothesis.".
        format(len(fhyp.activity_id.unique()), prednum))

    # Careful here w/ undefined extra columns in ref- or hyp-data
    # TP, FP, FN -> match of both: ref + hyp
    # MD -> hyp := NaN, confidence_scorer := NaN
    # TN -> filtered out
    mdata = ds.ref.merge(fhyp, how='outer', on='video_file_id', suffixes=('_ref', '_hyp'))
    # Make sure missed data is handled appropriately when using threshold of >0.0
    mdata.loc[mdata.activity_id_hyp.isna(), ['activity_id_hyp', 'confidence_score'] ] = ['0', -1.0 ]    
    ds.register = mdata
    
def tad_check_for_nan(ds):
    """ De-NaN GT and PRED (should ideally not have nan activity_id)
    """    
    # Check for NaN Activity Id and remove. Throw a warning now, might be an exception later.
    gtnan = ds.ref.isna().activity_id
    gtnanl = len(gtnan.loc[gtnan == True])
    if gtnanl > 0:
        log.warning("NaN activity_id in reference: Adding {} no-score regions.".format(gtnanl))
        gtnan = ds.ref.isna().activity_id
        ds.ref.loc[gtnan, 'activity_id']= "NO_SCORE_REGION"

    prednan = ds.hyp.isna().activity_id
    prednanl = len(prednan.loc[prednan == True])
    if prednanl > 0:
        log.warning("NaN activity_id in system-output detected. DROPPING {} NaN entries".format(prednanl))
        ds.hyp.dropna(inplace=True)
    # Check for NaN - END

def filter_by_activity(ds, activity_id_list):
    """ Filter REF to contain only activity-id in list
    Returns: filtered list
    """    
    ds.ref.loc[ds.ref.activity_id.isin(activity_id_list)]

def remove_out_of_scope_activities(ds):
    """ 
    If there are any activity-id which are out of scope or NA, whole entry is removed.    
    
    Side Effects:
    -------------
    - Modifies ds.hyp
    """    
    ds.hyp.drop(ds.hyp[~ds.hyp.activity_id.isin(ds.ref.activity_id.unique())].index, inplace = True)
    # Usecase: video_id,,,
    ds.hyp.drop(ds.hyp[ds.hyp.activity_id.isna()].index, inplace = True)

def append_missing_video_id(ds):
    """ 
    Create a new entry in the dataset based on missing video_file_id w/ a
    '_FN_' activity_id label and confidence_score of 1.0.

    Side Effects:
    -------------
    - Modifies ds.hyp

    :params DataSet ds: Dataset w/ ref and hyp data
    """
    ds.ref['video_file_id'] = pd.Categorical(ds.ref.video_file_id)
    ds.hyp['video_file_id'] = pd.Categorical(ds.hyp.video_file_id)
    ref_labels = ds.ref['video_file_id'].unique()
    hyp_labels = ds.hyp['video_file_id'].unique()
    label_distance = len(set(ref_labels) - set(hyp_labels))
    if label_distance > 0:                
        missing_vid = ds.ref[np.logical_not(ds.ref.video_file_id.isin(ds.hyp.video_file_id))]
        output = [ds.hyp]
        for index, entry in missing_vid.iterrows():
            log.warning("(FN) Appending missing: {}".format(entry.video_file_id))
            output.append(pd.DataFrame(data={
                'video_file_id': entry.video_file_id,
                'activity_id': '_FN_', 
                'confidence_score': 1.0
                }, index=[0]))
        ds.hyp = pd.concat(output)

def prep_tad_data(ds):
    """ 
    - Remove outt of band activity from pred (MD)
    - For all activity_id X video_filed_id compute IoU/MD/FP/TP of PRED vs. GT as described per eval-plan.
    Parameters
    ----------
    dataframe w/ results
    """    
    tad_check_for_nan(ds)
    #pdb.set_trace()

    # MD: Remove out-of-band activity-id from reference
    prednum = len(ds.hyp.activity_id.unique())
    ds.hyp.drop(ds.hyp[~ds.hyp.activity_id.isin(ds.ref.activity_id.unique())].index, inplace=True)
    log.info("[xform] removed {} activities from hyp".format(prednum - len(ds.hyp.activity_id.unique())))
    output = []    
    activities = ds.hyp.activity_id.unique()
    for activity in activities:
        mpred = []
        # include no-score-region represented as a separate class
        pRef = ds.ref.loc[(ds.ref.activity_id == activity) | (ds.ref.activity_id == 'NO_SCORE_REGION')]
        # only interested in activity occurences (rest is considered MD)
        pHyp = ds.hyp.loc[ds.hyp.activity_id == activity]
        for video_id in ds.hyp.video_file_id.unique():
            vRef = pRef.loc[pRef.video_file_id == video_id]
            vHyp = pHyp.loc[pHyp.video_file_id == video_id]
            vRef.index = pd.RangeIndex(len(vRef.index))
            vHyp.index = pd.RangeIndex(len(vHyp.index))            
            [mat, cVec, rVec, iVec] =  compute_alignment_matrix(vRef, vHyp, activity)
            df = pd.DataFrame(data={
                'video_file_id': video_id, 
                'activity_id': activity, 
                'IoU': np.around(iVec,3),
                'frame_start': np.around(vHyp['frame_start'],1),
                'frame_end': np.around(vHyp['frame_end'],1),
                'alignment': rVec,
                'confidence_score': np.around(cVec,3) })
            output.append(df)
    if len(output) == 0:
        log.error("No useable system output found!")
        return pd.DataFrame(columns = ['video_file_id', 'activity_id', 'IoU', 'frame_start', 'frame_end', 'alignment', 'confidence_score'])
    else:       
        return pd.concat(output)

def filter_by_top_k_confidence(data, k_value=0):
    """ Select top-k by using confidence score across video_file_id-groups.
    - data requires [activity_id, conf and video_id]    
    - if k_value = 0 only sort by confidence score, keeping all
    """    
    if k_value == 0:
        return data.sort_values(["activity_id", "confidence_score"], ascending=False)
    else:
        # using groupby + head is ~20x faster than using index + nlargest
        return data.sort_values(["activity_id", "confidence_score"], ascending=False).groupby('video_file_id').head(k_value)

def select_by_topk(ds, k_value=1):
    """ DEPRECATE """
    if k_value == 0:
        return ds.register.sort_values(["activity_id_gt", "confidence_score"], ascending=False)
    else:
        # using groupby + head is ~20x faster than using index + nlargest
        return ds.register.sort_values(["activity_id_gt", "confidence_score"], ascending=False).groupby('video_file_id').head(k_value)

def gen_md_data(vfid, activity, refdata):
    """
    Model Missed Detection Entry
    ----------------------------
    - NOTE: To make binary metrics computation work missing detection needs to
      be defined as 1.0 ! (we detect MD with 100% confidence)
    - However: missing detection is EXCLUDED from computing P-R lateron (see eval-spec.) so we set it to NaN here.
    - IoU of 0.0 ensures it's always flagged as not found.
    """
    return pd.DataFrame(index=[0], data={
        'video_file_id': vfid, 
        'activity_id': activity, 
        'frame_start': float("NaN"), 
        'frame_end': float("NaN"), 
        'IoU': 0.0,
        'ref_frame_start': refdata['frame_start'],
        'ref_frame_end': refdata['frame_end'],
        'alignment': 'MD',
        'confidence_score': float("NaN") })

def check_iou_overhang(cStart, cEnd, regionStart, regionEnd):
    """ Check if hyp data starts or ends outside of region. """
    return (cStart < regionStart) | (cEnd > regionEnd)

def compute_alignment_matrix(pRef, pHyp, activity):
    """
    Compute IoU and determine TP,FP,MD
    Results Vector Notation: (MD) -1, (FP) 0, (TP) 1
    Output IoU matrix of ref vs. hyp activities
    """

    lRef, lHyp = len(pRef), len(pHyp)
    iMat = np.zeros((lHyp, lRef)) # iou matrix    
    cVec = np.array(pHyp['confidence_score'])
    iVec = np.zeros(lHyp) # IoU result
    rVec = np.zeros(lHyp) # could set to -1 instead
    icVec = np.array(pRef.activity_id == activity) # in-class Vector    

    # Pass #0 - compute IoU across all pairs
    for hIdx, hRow in pHyp.iterrows():
        for rIdx, rRow in pRef.iterrows():            
            iMat[hIdx,rIdx] = compute_temp_iou(
                rRow['frame_start'], rRow['frame_end'],
                hRow['frame_start'], hRow['frame_end'])

    # Pass #1 - determine TP,FP,MD
    for hIdx, hRow in pHyp.iterrows():
        x = iMat[hIdx,:]
        if np.sum(x) == 0: # clear cut FP
            rVec[hIdx] = 0
            iVec[hIdx] = 1
        else:
            # covers both cases of: np.count_nonzero(x>0) > 1 and == 1
            rVec[hIdx] = 1 if icVec[np.argmax(x, axis=0)] else -1
            iVec[hIdx] = np.max(x) if icVec[np.argmax(x, axis=0)] else 0

    # Pass #2 - reduce multiple TP occurences over one-region by applying rules
    for hIdx, hCol in pRef.iterrows():        
        x = iMat[:, hIdx]        
        # - events missed by system: MD (unreported)
        # - multiple events chunked as one by system: 1TP, rest MD (unreported)
        if (np.count_nonzero(x>0) > 1) & icVec[hIdx]:
            maxes = np.flatnonzero(x == np.max(x))
            for iIdx in np.flatnonzero(x): # indices of nonzero col entry (IoU > 0)                
                if len(maxes) > 1: # multiple max IoU -> use bigger confidence_score
                    rVec[iIdx] = 1 if (cVec[iIdx] == np.max(cVec[maxes])) else 0
                    iVec[iIdx] = np.max(x) if (cVec[iIdx] == np.max(cVec[maxes])) else 1
                else: # one max IoU -> use bigger IoU
                    rVec[iIdx] = 1 if (x[iIdx] == np.max(x)) else 0
                    iVec[iIdx] = np.max(x) if (x[iIdx] == np.max(x)) else 1

    # Pass #3 - handle iou overhangs for all MD cases
    for rIdx in range(0, len(rVec)):
        if rVec[rIdx] < 0:            
            row = iMat[rIdx,:]
            rRow = np.argmax(row)
            #pdb.set_trace()
            rS = pRef.loc[rRow, 'frame_start']
            rE = pRef.loc[rRow, 'frame_end']
            hS = pHyp.loc[rIdx, 'frame_start']
            hE = pHyp.loc[rIdx, 'frame_end']
            if check_iou_overhang(rS,rE,hS,hE):
                # we check for INVERTED iou (the hanging out part)
                if row[rRow] < 0.2:
                    rVec[rIdx] = 0
                    iVec[rIdx] = 1
            
    #return [pRef, pHyp, iMat, cVec, rVec ]
    return [iMat, cVec, rVec, iVec]
