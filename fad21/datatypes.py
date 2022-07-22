import numpy as np
import pandas as pd
import csv
from sklearn import metrics as skm
import json
import logging
from collections import OrderedDict
import dill
import sys

log = logging.getLogger(__name__)

def err_quit(msg):
    """ Print an error message (msg) and exit(1). """
    log.critical(msg)
    sys.exit(1)

def get(list, index, default=None):
    """ Similar to dict.get, but for list objects. Returns value at index or
    default if not existent.
    """
    try:
        return list[index]
    except IndexError:
        return default

class Dataset(object):
    """ Dataset State

    Data Spec
    ---------
    # index_df      = { video_file_id, tbd } // NOT IN USE
    
    reference    = { video_file_id, start_frame, end_frame,                    activity_id, instance_id }
    hypothesis   = { video_file_id, start_frame, end_frame, processing_status, activity_id, confidence_score }        
    register     = { video_file_id, activity_id_hyp, activtiy_id_ref, confidence_score }

    # For Advanced Usage
    activity_ids = List of activity-ids to use (default: unique(REF.activity_id))
    video_ids    = List of video-ids to use (default: unique(REF.video_file_id))
    """

    def __init__(self, ref=None, hyp=None, register=None):
        """ Create Object

        Parameters
        ----------
        ref: pd.DataFrame
            Reference Dataframe
        hyp: pd.DataFrame
            Hypothesis Dataframe
        register: pd.DataFrame
            Dataframe for internal use.
        """
        self.ref, self.hyp, self.register = ref, hyp, register

        # Default to use all activity Id. 
        if 'activity_id' in self.ref:
            self.activity_ids = self.ref['activity_id'].unique()
        # Default to use all video Id. 
        self.video_ids = self.ref['video_file_id'].unique()

    def __repr__(self):
        """ String representation """        
        ref_str = "REF(None)"
        hyp_str = "HYP(None)"
        register_str = "REGISTER(None)"

        if (self.ref is not None):
            ref_str = "REF({}|{})".format(
                len(self.ref), len(self.ref['activity_id'].unique()))
        
        if (self.hyp is not None):
            hyp_str = "HYP({}|{})".format(                
                len(self.hyp), len(self.hyp['activity_id'].unique()))                

        return "[dataset] (#|ActivityID): {}, {}".format(ref_str, hyp_str)