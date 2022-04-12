import fad21
from fad21.scoring import TADScorer
from pathlib import Path
import pytest
import io

def scoring_run(gt_file, pred_file, topk, of):
    scorer = TADScorer(gt_file, pred_file)    
    scorer.score(['map', 'map_11'], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], of)

def test_tad_match_3(tmpdir):
    output = scoring_run('testdata/tad_gt_2x3.csv', 'testdata/tad_pred_2x3_match.csv', 1, tmpdir)
    assert(list(io.open(tmpdir + "/system_tad_scores.csv")) == 
        list(io.open("testrefs/tad_match_3/system_tad_scores.csv")))
    assert(list(io.open(tmpdir + "/activity_tad_scores.csv")) ==
        list(io.open("testrefs/tad_match_3/activity_tad_scores.csv")))        
