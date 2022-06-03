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
    validate_ac(ds)   
    score_ac(ds, ['map'], topk, outDir)
    h5f = h5_open_archive(os.path.join(outDir, 'scoring_results.h5'))
    data = h5_extract_system_scores(h5f)          
    aData = h5_extract_activity_scores(h5f)
    ensure_output_dir(outDir)    
    write_system_level_scores(os.path.join(outDir, 'system_ac_scores.csv'), data)
    write_activity_level_scores(os.path.join(outDir, 'activity_ac_scores.csv'), aData)

def test_ac_missing_video_in_hyp(tmpdir):
    with pytest.raises(fad21.validation.ValidationError):        
        scoring_run('testdata/ac_ref_2x3.csv', 'testdata/ac_hyp_2x3_missing_video_id.csv', 1, tmpdir)

def test_ac_no_matching_activity_in_hyp(tmpdir):
    scoring_run('testdata/ac_ref_2x3.csv', 'testdata/ac_hyp_2x3_no_matching_activity_id.csv', 1, tmpdir)
    assert(list(io.open(tmpdir + "/system_ac_scores.csv")) == 
        list(io.open("testrefs/ac_edgecases/sys_0_activities.csv")))
    assert(list(io.open(tmpdir + "/activity_ac_scores.csv")) ==
        list(io.open("testrefs/ac_edgecases/act_0_activities.csv")))
        