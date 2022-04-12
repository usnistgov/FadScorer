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

def dedup_gt_labels(ds):
    """ Cleanup method. Kept for reference. """
    ds.ref['ground_truth'] = 1
    refnum = len(ds.ref)
    ds.ref.drop_duplicates(inplace=True)
    log.info("GT activity-id uniqueness: {}".format(len(ds.ref)/refnum))


def err_quit(msg):
    """ Print an error and exit """
    log.critical(msg)
    sys.exit(1)


def get(list, index, default=None):
    """ similar to dict.get, but for list objects """
    try:
        return list[index]
    except IndexError:
        return default

"""
Dataset State Struct
    Data Spec:
        # index_df      = { video_file_id, tbd } // NOT IN USE
        
        ground_truth = { video_file_id, start_frame, end_frame,                    activity_id, instance_id }
        prediction   = { video_file_id, start_frame, end_frame, processing_status, activity_id, confidence_score }        
        register     = { video_file_id, activity_id_hyp, activtiy_id_ref, ground_truth, confidence_score }
        activity_ids = List of activity-ids to use (default: unique(GT.activity_id))
        video_ids    = List of video-ids to use (default: unique(GT.video_file_id))
"""
class Dataset(object):
    def __init__(self, gt=None, pred=None, register=None):
        self.ref = gt
        self.hyp = pred
        self.register = register
        self.activity_ids = self.ref['activity_id'].unique()
        self.video_ids = self.ref['video_file_id'].unique()

    def __repr__(self):
        gt_str = "GT(None)"
        pred_str = "PRED(None)"
        register_str = "REGISTER(None)"

        if (self.ref is not None):
            gt_str = "GT({}|{})".format(
                len(self.ref), len(self.ref['activity_id'].unique()))
        
        if (self.hyp is not None):
            pred_str = "PRED({}|{})".format(                
                len(self.hyp), len(self.hyp['activity_id'].unique()))                

        if (self.register is not None):
            register_str = "REGISTER({}| pred {}, gt {})".format(
                len(self.register), len(self.register['activity_id_pred'].unique()), len(self.register['activity_id_gt'].unique()))

        return "[dataset] (#|ActivityID):  {}, {}, {}".format(gt_str, pred_str, register_str)