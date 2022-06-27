import fad21
from fad21.scoring import score_ac
from fad21.validation import validate_ac
from fad21.datatypes import Dataset
from fad21.io import *
from pathlib import Path
import pytest
import io

# Validates scorer-implementation w/ specific classification use-cases.

def get_root():
    return Path(__file__).parent

def scoring_run(refFile, hypFile, topk, outDir):
    """ Loader Func """
    ds = Dataset(load_ref(get_root()/refFile), load_hyp(get_root()/hypFile))    
    score_ac(ds, ['map'], topk, outDir)
    h5f = h5_open_archive(os.path.join(outDir, 'scoring_results.h5'))
    data = h5_extract_system_scores(h5f)          
    aData = h5_extract_activity_scores(h5f)
    return [data, aData]

 
def test_md_50p_cases(tmpdir):
    """ USE-CASES
    Hypothesis has four usecases:
     
     - 50_p_missing:      : 100 vid, 1 class, 50% missing, rest retrieved at 1.0 confidence
     - 50_p_missing_noise : 100 vid, 1 class, 50% missing, rest retrieved at 0.990n confidence (n := random [0..9])
     - 50_p_correct       : 100 vid, 1 class, 50% retrieved, 50% not-retrieved (50_p_incorrect) at 1.0 confidence
     - 50_p_incorrect     : 100 vid, 1 class, 50% retrieved, 50% not-retrieved (50_p_correct) at 1.0 confidence     
     - 100_p_correct      : 10 vid, 1 class, 100% retrieved (class B), at 1.0 confidence

    Expected output:
        50_p_missing       : aP: 0.5
        50_p_missing_noise : aP: 0.5
        50_p_correct       : ap: 0.5
        50_p_correct       : ap: 0.5
        100_p_correct      : ap: 1.0
    """
    data, aData = scoring_run('testdata/ref_ec_1.csv', 'testdata/hyp_ec_1.csv', 0, tmpdir)
    hData = {}
    for entry in aData:
        hData[entry[0]] = entry[2]
    assert(len(hData) == 5)
    # resulting aP == P/R values here
    assert(hData['50_p_missing'] == pytest.approx(0.5, 0.1))
    assert(hData['50_p_missing_noise'] == pytest.approx(0.5, 0.1))
    assert(hData['50_p_correct'] == pytest.approx(0.5, 0.1))
    assert(hData['50_p_incorrect'] == pytest.approx(0.5, 0.1))
    assert(hData['100_p_correct'] == pytest.approx(1.0, 0.1))
