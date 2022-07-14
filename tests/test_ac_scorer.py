import fad21
from fad21.scoring import score_ac
from fad21.validation import validate_ac
from fad21.datatypes import Dataset
from fad21.io import *
from pathlib import Path
import pytest
import io

# Tests: 
#
# - scoring-execution
# - output file generation (hd5),
# - compares map-scores for top confidence score selection
# - check csv from hd5 extraction

def get_root():
    return Path(__file__).parent

# top_n_scores: 0 == all
def scoring_run(refFile, hypFile, top_n_scores, outDir):
    ds = Dataset(load_ref(get_root() / refFile), load_hyp(get_root() / hypFile))    
    score_ac(ds, ['map'], top_n_scores, outDir)
    h5f = h5_open_archive(os.path.join(outDir, 'scoring_results.h5'))
    data = h5_extract_system_scores(h5f)          
    aData = h5_extract_activity_scores(h5f)
    ensure_output_dir(outDir)    
    write_system_level_scores(os.path.join(outDir, 'system_ac_scores.csv'), data)
    write_activity_level_scores(os.path.join(outDir, 'activity_ac_scores.csv'), aData)

def test_scoring_edge_cases_1(tmpdir):
    scoring_run('testdata/ac_ref_ec_1.csv', 'testdata/ac_hyp_ec_1.csv', 0, tmpdir)    
    assert(list(io.open(tmpdir + "/system_ac_scores.csv")) == 
        list(io.open(get_root() / "testrefs/test_ec1/system_ac_scores.csv")))
    assert(list(io.open(tmpdir + "/activity_ac_scores.csv")) ==
        list(io.open(get_root() / "testrefs/test_ec1/activity_ac_scores.csv")))