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



def csv_has_no_header(file_name):
    """ Check if first line is a header (starts with '#') """
    with open(file_name) as f:
        line1 = f.readline()
        return not line1.startswith('#')

def sanitize_csv_header(file_name):    
    """ Given file_name read and return content with filtered spaces and LF's
    """
    with open(file_name) as f:        
        return [ l.strip() for l in f.readline()[1:].strip().split(',') ]

def _load_csv(file_name, **kwargs):    
    """ Generic CSV file loader based on pandas.read_csv used to selectively
    load reference, hypothesis or list-files.
    
    Parameters
    ----------
    file_name: str
        CSV file name
    **kwargs: kw-args
        List of special options for pd.read_csv.        
        names: 
            list named list of columns to load (others are ignored)
    
    Output
    ------
    df: pd.df
        Dataframe with dataset (ref or hyp) incl. resp. columns.
    """
    log.info("Loading CSV '{}'".format(file_name))
    log.debug("  - header: {}".format(kwargs['names']))
    df = pd.read_csv(file_name, **kwargs)
    log.info("  - loaded {} entries.".format(len(df)))
    df.columns = df.columns.str.strip()
    return df

def _autoload_csv(fn, header, types = None):
    """ Load CSV file, validating header information against file-header and
    enforcing strict column-type when loading data.

    Parameters
    ----------
    fn: str
        CSV file name
    header: list
        List of column names which must be present in the file.
    types: dict
        Dict with column name as key and type as value.

    Raises
    ------
    ValidationError
        If header is missing
    IOError
        Header is missing required columns.
    
    Output
    ------
    df: pd.df
        Dataframe with dataset (ref or hyp) incl. resp. columns.
    """    
    if csv_has_no_header(fn):
        log.warning("Can not parse header in first line of file '{}' ".format(fn))
        log.warning("Please use the following header (order of columns does not matter):")
        if 'frame_start' in types:
            log.warning("#video_file_id,activity_id,confidence_score,frame_start,frame_end")
        else:
            log.warning("#video_file_id,activity_id,confidence_score")        
        raise ValidationError("Cannot continue, please fix CSV file !")
    else:
        auto_header = sanitize_csv_header(fn)
        if set(header) <= set(auto_header):
            return _load_csv(fn, names=auto_header, comment='#', dtype=types)
        else:
            missing_set = set(header) - set(auto_header)
            raise IOError("Header of '{}' is missing following fields: {}".format(fn, list(missing_set)))

def load_ref(fn):
    """ Given fn load AC reference and return as dataframe """    
    return _autoload_csv(fn, ["video_file_id", "activity_id"], { 'video_file_id' : 'str', 'activity_id': 'str'})

def load_tad_ref(fn):
    """ Given fn load TAD reference and return as dataframe """
    return _autoload_csv(fn, ["video_file_id", "activity_id", "frame_start", "frame_end"], 
    { 'video_file_id' : 'str', 'activity_id': 'str', 'frame_start': 'float', 'frame_end': 'float'} )

def load_hyp(fn):    
    """ Given fn load AC system-output and return as dataframe """
    return _autoload_csv(fn, ["video_file_id", "activity_id", "confidence_score"], 
    { 'video_file_id' : 'str', 'activity_id': 'str', 'confidence_score': 'float'})

def load_tad_hyp(fn):
    """ Given fn load TAD system-output and return as dataframe """    
    return _autoload_csv(fn, ["video_file_id", "activity_id", "confidence_score", "frame_start", "frame_end" ],
        { 'video_file_id' : 'str', 'activity_id': 'str', 'confidence_score': 'float', 
        'frame_start': 'float', 'frame_end': 'float'} )

def load_index(fn):
    """ Given fn load Video-index list and rturn as dataframe """    
    return _autoload_csv(fn, ["video_file_id", "frame_rate"],
        { 'video_file_id' : 'str', 'frame_rate': 'float' } )

def load_mapping_file(fn):
    """ Given fn load JSON formatted mapping file and return as json.object """    
    f = open(fn)
    return json.load(f)

def write_header_output(fh, df, columns):
    """ Write CSV to file handle w/ custom header
    
    Parameters
    ----------
    fh: file
        File handle to an open file.
    df: dataframe
        Results dataframe
    columns: list
        List of columns to include in the CSV file.
    """
    fh.write("#{}\n".format(",".join(columns)))
    content = pd.DataFrame(df, columns = columns).to_csv(index=False, header=False)
    fh.write(content)        
    if fh is not sys.stdout:
        fh.close()
        log.info("Wrote '{}'".format(fh.name))

def write_output(fh, content):
    """ Genenric write using file handle (file, sys.stdout etc.)

    Parameters
    ----------
    fh: file
        File handle to an open file.
    content: object
        String or List to write to file.    
    """
    fh.write(content)        
    if fh is not sys.stdout:
        fh.close()
        log.info("Wrote '{}'".format(fh.name))

def ensure_output_dir(odir):
    """ Check if output directory exists and if not, create it. """
    if not os.path.exists(odir):
        os.makedirs(odir)

def wipe_scoring_file(fn):
    """ Delete file """
    if os.path.exists(fn):
        os.remove(fn)

def load_list_file(fn):
    """ Given fn load list from file using '\n' as separator. Raises if file is
    not found """
    if os.path.exists(fn):
        fh = open(fn, "r")
        entries = fh.read()
        return (entries.split("\n"))
    else:
        raise IOError("File not found: '{}'".format(fn))

# ----------------------------------------------------------------------------
def h5_type_fetch(object, name, attr_val):
    """ Helper Method to ensure get on path + attr. Creates group if it does not
    exist.

    Parameters
    ----------
    object: h5py.object
        File or Group
    name: str
        Tail of path to get/add on to        
    attr_val: object
        attribute value

    Output
    ------
    h5py.object:
        Group either retrieved or new.
    """
    if name in object.keys():
        return(object[name])
    else:
        group = object.create_group(name)
        group.attrs['ftype'] = attr_val
        return(group)  

# ----------------------------------------------------------------------------

def h5_create_archive(fn, mode = 'a'):
    """ Create H5 File. Used for scoring-results. """
    log.info("Creating scoring results file: '{}'".format(fn))
    fh = h5py.File(fn, mode)
    fh.attrs['scorer'] = "FAD21"
    fh.attrs['version'] = 20220223    
    return(fh)

def h5_add_info(h5f, argstr, scoring_mode):
    """ Append CLI runtime parameters in H5 archive """    
    h5f.attrs['scorer-args'] = argstr
    h5f.attrs['scorer-mode'] = scoring_mode

def h5_open_archive(fn, mode = 'r'):
    """ Open H5 Archive for access """
    log.info("Opening scoring results file: '{}'".format(fn))
    fh = h5py.File(fn, mode)
    return(fh)    

# ----------------------------------------------------------------------------

def h5_add_system_scores(h5f, results):
    """ Append system-scores to h5f file

    Parameters
    ----------
    h5f: H5 handle
        Opened archive handle.
    results: dict
        System Scores
    """
    sysG = h5f.create_group('system')
    sysG.attrs['ftype'] = "LKey"
    for [metric, value] in results:
        sysG.create_dataset(metric, data=value)

def h5_add_activity_prt(h5f, pr_data):
    """ Append per-activity P/R cuvrve data H5 File

    Parameters
    ----------
    h5f: H5 handle
        Opened archive handle.
    pr_data: pd.Dataframe
        DataFrame with [activity_id, precision, recall] columns
    """    
    actG = h5_type_fetch(h5f, 'activity', "LKey")
    for _, row in pr_data.iterrows():
        activitySubG = h5_type_fetch(actG, row['activity_id'], "LVal")
        prtG = h5_type_fetch(activitySubG, 'prt', "MKey")
        prtG.create_dataset("recall", data=row['recall'][::-1])
        prtG.create_dataset("precision", data=row['precision'][::-1])
        #prtG.create_dataset("thresholds", data=row['thresholds'])        

def h5_add_activity_scores(h5f, results):
    """ Append activity-scores to h5f file

    Parameters
    ----------
    h5f: H5 handle
        Opened archive handle.
    results: dict
        Activity Scores
    """
    actG = h5_type_fetch(h5f, 'activity', "LKey")
    for activity, values in results.items():
        activitySubG = h5_type_fetch(actG, activity, "LVal")
        for metric, value in values.items():            
            activitySubG.create_dataset(metric, data=value)

# TODO deprecate interp flag
def h5_sub_add_aggregated_pr(graphsG, aggregated_xy, interp=False):
    """ Append aggregated P/R cuvrve data to H5 Sub-Group.

    Parameters
    ----------
    graphsG: H5 Group handle
        Pointing to sub-group.
    aggregated_xy: 2d-array
        1d-array[float] of Precision, Recall and Standard-Error
    """        
    if interp:
        prG = h5_type_fetch(graphsG, 'prs_interp', "MKey")
    else:
        prG = h5_type_fetch(graphsG, 'prs', "MKey")
    if 'precision' in prG.keys():
        del prG['precision']
        del prG['recall']
        del prG['stderror']
    prG.create_dataset('precision', data=aggregated_xy[1])
    prG.create_dataset('recall', data=aggregated_xy[0])
    prG.create_dataset('stderror', data=aggregated_xy[2])                

# TODO deprecate interp flag
def h5_add_aggregated_pr(h5f, aggregated_xy, interp=False):
    """ Append aggregated P/R cuvrve data to H5 File under /system

    Parameters
    ----------
    h5f: H5 handle
        Opened archive handle.
    aggregated_xy: 2d-array
        1d-array[float] of Precision, Recall and Standard-Error    
    """            
    log.debug("Writing aggregated XY PR-curves")
    graphsG = h5_type_fetch(h5f, 'system', "LKey")
    h5_sub_add_aggregated_pr(graphsG, aggregated_xy, interp)
        
# ----------------------------------------------------------------------------
def h5_add_iou_system_scores(h5f, results):
    """ Append IoU system-scores to H5 file

    Parameters
    ----------
    h5f: H5 handle
        Opened archive handle.
    results: dict
        System Scores
    """    
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
    """ Append IoU activity-scores to H5 file.

    Parameters
    ----------
    h5f: H5 handle
        Opened archive handle.
    results: dict
        System Scores
    """    
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
    """ Store activity P/R curves for all tIoU thresholds in a HDF5 file.

    Parameters
    ----------
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
                        #prtG.create_dataset("thresholds", data=row['thresholds'])

def h5_add_aggregated_iou_pr(h5f, tiou, aggregated_xy):
    """ Append aggregated P/R cuvrve data for specific tIoU to a H5 File under
    /systems/iou/
    
    Parameters
    ----------
    h5f: H5 handle
        Opened archive handle.
    tiou: str
        tIoU Threshold of graph for Group-Name
    aggregated_xy: 2d-array
        1d-array[float] of Precision, Recall and Standard-Error    
    """            
    log.info("Writing aggregated XY PR-curves for {}".format(tiou))    
    iousG = h5f['system/iou']
    iouG = h5_type_fetch(iousG, tiou, "PVal")
    h5_sub_add_aggregated_pr(iouG, aggregated_xy)

# ----------------------------------------------------------------------------

# Create array w/ system level scores
def h5_extract_system_scores(h5f):
    """ Get System scores from H5 file.

    Parameters
    ----------
    h5f: H5 handle
        Opened archive handle.

    Output
    ------
    1d-array:
        with tuples of metric and scores.
    """    
    dataArr = []
    scoresG = h5f['/system']
    for scores in scoresG.keys():
        sG = scoresG[scores]
        if isinstance(sG, h5py.Dataset):
            dataArr.append([scores, sG[()]])
    return dataArr

def h5_extract_system_iou_scores(h5f):
    """ Get System IoU scores from H5 file.

    Parameters
    ----------
    h5f: H5 handle
        Opened archive handle.

    Output
    ------
    1d-array:
        with tuples of metric and scores.
    """        
    dataArr = []
    iousG = h5f['/system/iou']
    for iou in iousG.keys():
        iouG = iousG[iou]
        for metric in iouG.keys():
            sG = iouG[metric]
            if isinstance(sG, h5py.Dataset):
                # CHANGE THIS TO BE metric, iou_value, score csv (also need to change write_system_level_scores)
                dataArr.append(["{}_iou_{:.2f}".format(metric, float(iou)), sG[()]])                
    return dataArr

# Create array w/ activity level scores
def h5_extract_activity_scores(h5f):
    """ Get Activity scores from H5 file.

    Parameters
    ----------
    h5f: H5 handle
        Opened archive handle.

    Output
    ------
    1d-array:
        with tuples of activity, metric and scores.
    """    
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
    """ Get Activity IoU scores from H5 file.

    Parameters
    ----------
    h5f: H5 handle
        Opened archive handle.

    Output
    ------
    1d-array:
        with tuples of activity, metric and scores.
    """    
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
                        dataArr.append([activity, "{}_iou_{:.2f}".format(metric, float(iou)), sG[()]])
    return dataArr

# ----------------------------------------------------------------------------
# Extract as csv
def write_system_level_scores(output_file, results):
    """ Write system level scores to output file.

    Parameter
    ---------
    output_file: str
        Full path of output file. Will overwrite content or create file on demand.
    results: 1darray of df
        System scores results array
    """    
    co = []
    fh = open(output_file, "w")
    write_output(fh, pd.DataFrame(results, columns = ['metric', 'value']).to_csv(index=False))

# Extract as csv
def write_activity_level_scores(output_file, al_results):
    """ Write activity level scores to output file.

    Parameter
    ---------
    output_file: str
        Full path of output file. Will overwrite content or create file on demand.
    al_results: 1darray of df
        Activity results array
    """
    fh = open(output_file, "w")        
    write_output(fh, pd.DataFrame(al_results, columns = ['activity_id', 'metric', 'value']).to_csv(index=False))
    fh.close()

def gen_empty_output(activity):
    """ Generate Empty P/R plot data in case of empty activities (NaN, MD) which
    are being serialized nevertheless.

    Parameters
    ----------
    activity: str
        Activity-id to use as designated label.
    """
    return(pd.DataFrame([[[0., 0.], [0., 0], [0., 1.0],activity]],
        columns=['precision', 'recall', 'thresholds', 'activity_id']))

def eprint(*args, **popts):
    """ Print to stderr instead of stdout 
    
    Parameters
    ----------
    *args: non keyword args
        output to print 
    **popts: kw-args
        special print options
    """
    print(*args, file=sys.stderr, **popts)