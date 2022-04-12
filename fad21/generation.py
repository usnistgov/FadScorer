import numpy as np
import pandas as pd
import csv
import logging
import sys
import random

# Use this to abort silently on broken pipes (interact w/ standard UNIX tools like head)
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

log = logging.getLogger(__name__)

from .datatypes import Dataset
from .io import *

class GenerationError(Exception):
    """Exception for error reporting."""
    pass

class ACGenerator(object):

    def add_confidence(self):
        """Create a 'confidence_score' column in system output ds."""        
        dim = self.ds.hyp.shape[0]
        #self.ds.hyp = pd.DataFrame(self.ds.ref)
        self.ds.hyp['confidence_score'] = np.random.uniform(0, 1, dim)
    
    def gen_perfect_match(self):        
        """
        Generate a 1-1 match system output using reference activity-id/video-id paris.
        """
        log.info("Generating perfect match DS from Reference")
        self.add_confidence()
        self.ds.hyp = self.ds.hyp.sample(frac=1).reset_index(drop=True)
        return (self.ds.hyp)
        
    def gen_randomized_topk_match_output(self, match_factor=0.5, topk=1):
        """
        Randomize n% of activity-id matches using reference reference activity-id/video-id paris.
        - 0.0 = 100% random match
        - 0.5 = 50% random, 50% matching
        - 1.0 = 0% random match (perfect match)
        """
        log.info("Random System-Ouptut Generator. Match factor: {}%".format(match_factor * 100))        
        sdf = pd.DataFrame(self.ds.ref)        
        md_match =  pd.DataFrame(sdf.sample(frac=match_factor))
        md_random = sdf.drop(md_match.index)
        log.info("match: {}, random: {}".format(len(md_match), len(md_random)))

        outary = []
        for i in range(0,topk):
            log.info("Round {}".format(i))
            outary.append(pd.DataFrame(md_match))
            ract = pd.DataFrame(md_random)
            ract['activity_id'] = np.random.permutation(ract['activity_id'].values)
            outary.append(pd.DataFrame(ract))
        
        self.ds.hyp = pd.concat(outary, sort=False)
        self.add_confidence()
        return(self.ds.hyp)

    def gen_randomized_match_output(self, match_factor=0.5):
        """
        Randomize n% of activity-id matches using reference reference activity-id/video-id paris.
        - 0.0 = 100% random match
        - 0.5 = 50% random, 50% matching
        - 1.0 = 0% random match (perfect match)
        """
        log.info("Generating {}% matching, {}% random output.".format(match_factor * 100, (1.0-match_factor)*100))
        l_match = len(self.ds.ref)*match_factor
        l_random = len(self.ds.ref)*(1.0-match_factor)
        log.info("  #{} matching, #{} random".format(l_match, l_random))
        self.add_confidence()

        sdf = pd.DataFrame(self.ds.ref)
        sdf.sample(frac=1).reset_index(drop=True)                

        md_match =  sdf.loc[:l_match-1]
        md_random = sdf.loc[l_match:l_random+l_match]
        out_random = pd.DataFrame(md_random)
        out_random['activity_id'] = np.random.permutation(md_random['activity_id'].values)
        log.info(len(md_random))
        self.ds.hyp = pd.concat([md_match, out_random], sort=False)
        return(self.ds.hyp)        
    
    def __init__(self, ground_truth_fn=None):
        #np.random.seed(404)
        if not ground_truth_fn:
            raise IOError('Missing ground_truth file.')            
        self.ds = Dataset(load_ref(ground_truth_fn))
        log.info(self.ds)

    def generate(self, method, match_factor):
        if method == 'random':
            #content = self.gen_randomized_match_output(match_factor)
            content = self.gen_randomized_topk_match_output(match_factor, topk=3)
        elif method == 'match':
            content = self.gen_perfect_match()
        else:
            raise GeneratorError("Need to provide a valid method for generation !")
        return(content)

def random_offset_start_end_frames(datarow):
        delta = round((datarow['frame_end'] - datarow['frame_start'])*0.7, 0)
        datarow['frame_start'] += max(0, random.randint(-delta,delta-1))
        datarow['frame_end'] += min(datarow['frame_start']+1, random.randint(delta,delta))        
        return datarow

class TADGenerator(object):

    def randomomize_confidence_scores(self):
        """
        Create a 'confidence_score' column in system output ds.
        """
        dim = self.ds.ref.shape[0]
        self.ds.hyp = pd.DataFrame(self.ds.ref)
        self.ds.hyp['confidence_score'] = np.random.uniform(0, 1, dim)
 
    def gen_perfect_match(self):        
        """
        Generate a 1-1 match system output using reference activity-id/video-id paris.
        """
        log.info("Generating perfect match DS from Reference w/ random confidence scores.")
        self.randomomize_confidence_scores()
        self.ds.hyp = self.ds.hyp.sample(frac=1).reset_index(drop=True)
        return (self.ds.hyp)

         
    def gen_randomized_match_output(self, match_factor=0.5):
        """
        Randomize n% of activity-id matches using reference reference activity-id/video-id paris.
        - 0.0 = 100% random match
        - 0.5 = 50% random, 50% matching
        - 1.0 = 0% random match (perfect match)
        """
        log.info("Generating {}% matching, {}% random class output.".format(match_factor * 100, (1.0-match_factor)*100))
        l_match = len(self.ds.ref)*match_factor
        l_random = len(self.ds.ref)*(1.0-match_factor)
        log.info("  #{} matching, #{} random".format(l_match, l_random))
        self.randomomize_confidence_scores()

        md_match = self.ds.ref.loc[:l_match-1]
        md_random = self.ds.ref.loc[l_match:l_random+l_match]
        out_random = pd.DataFrame(md_random)
        out_random['activity_id'] = np.random.permutation(md_random['activity_id'].values)
        log.info(len(md_random))
        self.ds.hyp = pd.concat([md_match, out_random])
        self.ds.hyp = self.ds.hyp.apply(random_offset_start_end_frames, axis=1)
        return(self.ds.hyp)
    
    def __init__(self, ds=None):
        #np.random.seed(404)
        self.ds = ds
        log.info(self.ds)

    def generate(self, method, match_factor):
        if method == 'random':            
            content = self.gen_randomized_match_output(match_factor)
        elif method == 'match':
            content = self.gen_perfect_match()
        else:
            raise GeneratorError("Need to provide a valid method for generation !")
        return(content)