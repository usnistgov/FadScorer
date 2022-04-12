import pandas as pd
import json
import logging
import sys
import os
import glob
import h5py
import shlex

import pdb
from .validation import ValidationError

log = logging.getLogger(__name__)

def check_csv_header(file_name):
    with open(file_name) as f:
        line1 = f.readline()
        if line1.startswith('#'):            
            return [ l.strip() for l in line1[1:].strip().split(',') ]
    return None        

def load_mapping_file(file_name):
    """ Mapping file is a json file """
    f = open(file_name)    
    return json.load(f)

def _load_csv(file_name, **kwargs):
    """ Pandas Data-Loader, takes read_csv's args (see panda's docs) """
    log.info("Loading CSV '{}'".format(file_name))
    log.debug("  - header: {}".format(kwargs['names']))
    df = pd.read_csv(file_name, **kwargs)
    log.info("  - loaded {} entries.".format(len(df)))
    df.columns = df.columns.str.strip()
    return df

def _autoload_csv(fn, header, types = None):
    auto_header = check_csv_header(fn)
    if auto_header is None:
        log.debug("File is missing header in first line ! File: '{}'".format(fn))
        log.warning("Please use the following heders (order does not matter):")
        log.warning("--------------------------------------------------------")
        if 'frame_start' in types:
            log.warning("#video_file_id,activity_id,confidence_score,frame_start,frame_end")
        else:
            log.warning("#video_file_id,activity_id,confidence_score")        
        raise ValidationError("Missing header in first line of CSV file !")
        # automatically infer/match the header , EXCLUDE as this can lead to
        # hard to track down errors due to errorouns data/column assignments.
        # return _load_csv(fn, names=header, dtype=types)
    else:                
        if set(auto_header) <= set(header):
            return _load_csv(fn, names=auto_header, comment='#', dtype=types)
        else:
            missing_set = set(header) - set(auto_header)
            raise IOError("Header of '{}' missing fields: {}".format(fn, list(missing_set)))

def load_ref(fn):
    """ Load Reference data for Activity Classification Task
    - infers header as ["video_file_id", "activity_id"] if not provided
    - using header if provided in first line as comment '#col1,col2m .. colN'
    """
    return _autoload_csv(fn, ["video_file_id", "activity_id"], { 'video_file_id' : 'str', 'activity_id': 'str'})

def load_tad_ref(fn):
    """ Prelim. specific ground-truth loader for Activity DETection Task """
    return _autoload_csv(fn, ["video_file_id", "activity_id", "frame_start", "frame_end"], 
    { 'video_file_id' : 'str', 'activity_id': 'str', 'frame_start': 'float', 'frame_end': 'float'} )

def load_hyp(fn):
    """ Prelim. specific system output loader """
    return _autoload_csv(fn, ["video_file_id", "activity_id", "confidence_score"], 
    { 'video_file_id' : 'str', 'activity_id': 'str', 'confidence_score': 'float'})

def load_tad_hyp(fn):
    """ Prelim. specific system output loader """
    return _autoload_csv(fn, ["video_file_id", "activity_id", "confidence_score", "frame_start", "frame_end" ],
        { 'video_file_id' : 'str', 'activity_id': 'str', 'confidence_score': 'float', 
        'frame_start': 'float', 'frame_end': 'float'} )

def write_header_output(fh, df, columns):
    """ Write to file handle w/ custom header """
    fh.write("#{}\n".format(",".join(columns)))
    content = pd.DataFrame(df, columns = columns).to_csv(index=False, header=False)
    fh.write(content)        
    if fh is not sys.stdout:
        fh.close()
        log.info("Wrote '{}'".format(fh.name))

def write_output(fh, content):
    """ Write to file handle (file, sys.stdout etc.) """
    fh.write(content)        
    if fh is not sys.stdout:
        fh.close()
        log.info("Wrote '{}'".format(fh.name))

def ensure_output_dir(odir):
    if not os.path.exists(odir):
        os.makedirs(odir)

def wipe_scoring_file(fn):
    if os.path.exists(fn):
        os.remove(fn)

def load_list_file(fn):
    if os.path.exists(fn):
        fh = open(fn, "r")
        entries = fh.read()
        return (entries.split("\n"))
    else:
        raise IOError("File not found: '{}'".format(fn))

# ----------------------------------------------------------------------------

def h5_type_fetch(object, name, attr_val):
    """ Helper Method to either get or create a group path + attr
    Input:
    ------
    object:     hd5 Type Obj (assumed File or Group)
    name:       Tail of path to get/add on    
    attr_val:   attribute value
    """
    if name in object.keys():
        return(object[name])
    else:
        group = object.create_group(name)
        group.attrs['ftype'] = attr_val
        return(group)  

# ----------------------------------------------------------------------------

def h5_create_archive(fn, mode = 'a'):
    log.info("Creating scoring results file: '{}'".format(fn))
    fh = h5py.File(fn, mode)
    fh.attrs['scorer'] = "FAD21"
    fh.attrs['version'] = 20220223    
    return(fh)

def h5_add_info(h5f, argstr, scoring_mode):
    """ 
    Store run parameters in archive 
    """    
    h5f.attrs['scorer-args'] = argstr
    h5f.attrs['scorer-mode'] = scoring_mode

def h5_open_archive(fn, mode = 'r'):
    log.info("Opening scoring results file: '{}'".format(fn))
    fh = h5py.File(fn, mode)
    return(fh)    

# ----------------------------------------------------------------------------

def h5_add_system_scores(h5f, results):
    sysG = h5f.create_group('system')
    sysG.attrs['ftype'] = "LKey"
    for [metric, value] in results:
        sysG.create_dataset(metric, data=value)

def h5_add_activity_prt(h5f, pr_data):
    actG = h5_type_fetch(h5f, 'activity', "LKey")
    for _, row in pr_data.iterrows():
        activitySubG = h5_type_fetch(actG, row['activity_id'], "LVal")
        prtG = h5_type_fetch(activitySubG, 'prt', "MKey")
        prtG.create_dataset("recall", data=row['recall'][::-1])
        prtG.create_dataset("precision", data=row['precision'][::-1])
        prtG.create_dataset("thresholds", data=row['thresholds'])        

def h5_add_activity_scores(h5f, results):
    actG = h5_type_fetch(h5f, 'activity', "LKey")
    for activity, values in results.items():
        activitySubG = h5_type_fetch(actG, activity, "LVal")
        for metric, value in values.items():            
            activitySubG.create_dataset(metric, data=value)

def h5_sub_add_aggregated_pr(graphsG, aggregated_xy):
    prG = h5_type_fetch(graphsG, 'prs', "MKey")
    prG.create_dataset('precision', data=aggregated_xy[1])
    prG.create_dataset('recall', data=aggregated_xy[0])
    prG.create_dataset('stderror', data=aggregated_xy[2])                

def h5_add_aggregated_pr(h5f, aggregated_xy):
    log.debug("Writing aggregated XY PR-curves")
    graphsG = h5_type_fetch(h5f, 'system', "LKey")
    h5_sub_add_aggregated_pr(graphsG, aggregated_xy)
        
# ----------------------------------------------------------------------------
def h5_add_iou_system_scores(h5f, results):
    sysG = h5f.create_group('system')
    sysG.attrs['ftype'] = "LKey"
    paramKG = sysG.create_group('iou')
    paramKG.attrs['ftype'] = "PKey"
    for iout, metrics in results.items():   
        iouG = paramKG.create_group("{}".format(iout))
        iouG.attrs['ftype'] = "PVal"
        for metric, value in metrics.items():   
            iouG.create_dataset(metric, data=value)

def h5_add_iou_activity_scores(h5f, results):
    actG = h5_type_fetch(h5f, 'activity', "LKey")
    for activity, ious in results.items():
        activitySubG = h5_type_fetch(actG, activity, "LVal")
        paramKG = activitySubG.create_group('iou')
        paramKG.attrs['ftype'] = "PKey"
        for iout, metrics in ious.items():   
            iouG = paramKG.create_group("{}".format(iout))
            iouG.attrs['ftype'] = "PVal"        
            for metric, value in metrics.items():            
                iouG.create_dataset(metric, data=value)

def h5_add_iou_activity_prt(h5f, pr_iou_scores, iou_thresholds, activities):
    """ Dump all activity P/R scores, one HDF5 file per activity
    Input:
    ------
    h5f:              HDF5-File handle
    pr_iou_scores:    pd.dataframe w/ activity as rows, precision, recall and threshold as columns.
    iou_thresholds:   Array of thr.
    activities:       List of activities
    """ 
    actG = h5_type_fetch(h5f, 'activity', "LKey")
    
    for activity in activities:
        if not activity == "NO_SCORE_REGION":
            activitySubG = h5_type_fetch(actG, activity, "LVal") 
            for iout in iou_thresholds:
                prsi = pr_iou_scores[iout]
                target_prsi = prsi.loc[prsi.activity_id == activity]
                # REMOVED: Pad w/ empty data if IoU is excluding activity (due to
                # MD) This however needs to be detected when using this data to
                # aggregate to not distort the detection score (easier to detect miss but could include alignment status)
                # target_prsi = gen_empty_output(activity)            
                paramKG = h5_type_fetch(activitySubG, 'iou', "PKey")
                if not target_prsi.empty:                
                    for index, row in target_prsi.iterrows():                
                        #graphG = h5_fetch(activitySubG, 'graphs')                    
                        iouG = h5_type_fetch(paramKG, "{}".format(iout), "PVal")  
                        prtG = h5_type_fetch(iouG, 'prt', "MKey")
                        prtG.create_dataset("recall", data=row['recall'][::-1])
                        prtG.create_dataset("precision", data=row['precision'][::-1])
                        prtG.create_dataset("thresholds", data=row['thresholds'])

def h5_add_aggregated_iou_pr(h5f, iouStr, aggregated_xy):
    log.debug("Writing aggregated XY PR-curves for {}".format(iouStr))    
    iousG = h5f['system/iou']
    iouG = h5_type_fetch(iousG, iouStr, "PVal")
    h5_sub_add_aggregated_pr(iouG, aggregated_xy)

# ----------------------------------------------------------------------------

# Create array w/ system level scores
def h5_extract_system_scores(h5f):
    dataArr = []
    scoresG = h5f['/system']
    for scores in scoresG.keys():
        sG = scoresG[scores]
        if isinstance(sG, h5py.Dataset):
            dataArr.append([scores, sG[()]])
    return dataArr

def h5_extract_system_iou_scores(h5f):
    dataArr = []
    iousG = h5f['/system/iou']
    for iou in iousG.keys():
        iouG = iousG[iou]
        for metric in iouG.keys():
            sG = iouG[metric]
            if isinstance(sG, h5py.Dataset):
                # CHANGE THIS TO BE metric, iou_value, score csv (also need to change write_system_level_scores)
                dataArr.append(["{}_iou_{}".format(metric, iou), sG[()]])                
    return dataArr

# Create array w/ activity level scores
def h5_extract_activity_scores(h5f):
    activitiesG = h5f['activity']
    dataArr = []
    for activity in activitiesG.keys():
        scoresG = activitiesG[activity]        
        for metric in scoresG.keys():
            sG = scoresG[metric]
            if isinstance(sG, h5py.Dataset):
                dataArr.append([activity, metric, scoresG[metric][()]])
    return dataArr

def h5_extract_activity_iou_scores(h5f):
    activitiesG = h5f['activity']
    dataArr = []
    for activity in activitiesG.keys():
        activityG = activitiesG[activity]
        if 'iou' in activityG.keys():
            iousG = activityG["iou"]
            for iou in iousG.keys():
                iouG = iousG[iou]
                for metric in iouG.keys():
                    sG = iouG[metric]
                    if isinstance(sG, h5py.Dataset):
                        dataArr.append([activity, metric, sG[()]])
    return dataArr

# NOTE: OVERWRITES scoring results file ! Use first (or mess w/ appending as a)
def h5_add_alignment(h5f, alignment_df):
    alignment_df.to_hdf(h5f, 'alignments', mode='w', format='table', append=True)
    #write_output(fh, alignment_df.to_csv(index=False))

def h5_extract_alignment(hdf_fn):
    return pd.read_hdf(hdf_fn, 'alignments')    

# ----------------------------------------------------------------------------
def write_alignment_file(output_file, alignment_df):
    fh = open(output_file, "w")
    write_output(fh, alignment_df.to_csv(index=False))

# ----------------------------------------------------------------------------
# Extract as csv
def write_system_level_scores(output_file, results):
    co = []
    fh = open(output_file, "w")
    write_output(fh, pd.DataFrame(results, columns = ['metric', 'value']).to_csv(index=False))

# Extract as csv
def write_activity_level_scores(output_file, al_results):
    fh = open(output_file, "w")        
    write_output(fh, pd.DataFrame(al_results, columns = ['activity_id', 'metric', 'value']).to_csv(index=False))
    fh.close()

def gen_empty_output(activity):
    """     
    Generate Empty Graph Output in case of empty activities (NaN, MD) which are
    being serialized nevertheless. This is expected to happen for most IoU
    computations as TP might be missing due to low thresholds.
    """
    return(pd.DataFrame([[[0., 0.], [0., 0], [0., 1.0],activity]],
        columns=['precision', 'recall', 'thresholds', 'activity_id']))