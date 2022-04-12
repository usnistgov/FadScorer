import fad21
from fad21.scoring import score_ac
from fad21.validation import validate_ac
from fad21.datatypes import Dataset
from fad21.io import *
from pathlib import Path
import pytest
import io

# Tests: scoring-execution, hd5 gen, csv-gen, map-score

def scoring_run(refFile, hypFile, topk, outDir):
    ds = Dataset(load_ref(refFile), load_hyp(hypFile))    
    score_ac(ds, ['map'], topk, outDir)
    h5f = h5_open_archive(os.path.join(outDir, 'scoring_results.h5'))
    data = h5_extract_system_scores(h5f)          
    aData = h5_extract_activity_scores(h5f)
    ensure_output_dir(outDir)    
    write_system_level_scores(os.path.join(outDir, 'system_ac_scores.csv'), data)
    write_activity_level_scores(os.path.join(outDir, 'activity_ac_scores.csv'), aData)


def test_scoring_match_3(tmpdir):
    scoring_run('testdata/ac_ref_2x3.csv', 'testdata/ac_hyp_2x3_perf.csv', 1, tmpdir)
    assert(list(io.open(tmpdir + "/system_ac_scores.csv")) == 
        list(io.open("testrefs/test_scoring_match_3/system_ac_scores.csv")))
    assert(list(io.open(tmpdir + "/activity_ac_scores.csv")) ==
        list(io.open("testrefs/test_scoring_match_3/activity_ac_scores.csv")))        

def test_scoring_1fp_3(tmpdir):
    scoring_run('testdata/ac_ref_2x3.csv', 'testdata/ac_hyp_2x3_1fp.csv', 1, tmpdir)
    assert(list(io.open(tmpdir + "/system_ac_scores.csv")) ==
        list(io.open("testrefs/test_scoring_1fp_3/system_ac_scores.csv")))
    assert(list(io.open(tmpdir + "/activity_ac_scores.csv")) ==
        list(io.open("testrefs/test_scoring_1fp_3/activity_ac_scores.csv")))        

def test_scoring_2fp_3(tmpdir):
    scoring_run('testdata/ac_ref_2x3.csv', 'testdata/ac_hyp_2x3_2fp.csv', 1, tmpdir)
    assert(list(io.open(tmpdir + "/system_ac_scores.csv")) ==
        list(io.open("testrefs/test_scoring_2fp_3/system_ac_scores.csv")))
    assert(list(io.open(tmpdir + "/activity_ac_scores.csv")) ==
        list(io.open("testrefs/test_scoring_2fp_3/activity_ac_scores.csv")))            

def test_scoring_const(tmpdir):
    scoring_run('testdata/ac_ref_2x3.csv', 'testdata/ac_hyp_2x3_const.csv', 1, tmpdir)
    assert(list(io.open(tmpdir + "/system_ac_scores.csv")) ==
        list(io.open("testrefs/test_scoring_const/system_scores.csv")))
    assert(list(io.open(tmpdir + "/activity_ac_scores.csv")) ==
        list(io.open("testrefs/test_scoring_const/activity_scores.csv")))            

def test_scoring_match_19(tmpdir):
    scoring_run('testdata/ac_ref_19x840.csv', 'testdata/ac_hyp_19x840_perf.csv', 1, tmpdir)
    assert(list(io.open(tmpdir + "/system_ac_scores.csv")) ==
        list(io.open("testrefs/test_scoring_match_19/system_ac_scores.csv")))
    assert(list(io.open(tmpdir + "/activity_ac_scores.csv")) ==
        list(io.open("testrefs/test_scoring_match_19/activity_ac_scores.csv")))
