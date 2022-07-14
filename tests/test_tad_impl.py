import fad21
from fad21.scoring import score_tad
from fad21.validation import validate_tad
from fad21.datatypes import Dataset
from fad21.io import *
from pathlib import Path
import numpy as np
import pytest
import io
import collections

# Validate TAD scorer-implementation w/ edge-cases

def get_root():
    return Path(__file__).parent

def tad_scoring_run(refFile, hypFile, outDir):
    """ Loader Func """
    ds = Dataset(load_tad_ref(get_root()/refFile), load_tad_hyp(get_root()/hypFile))
    validate_tad(ds)
    # Round linspace due to numerical precision issues(0.6.....01) ofsetting IoU levels at critical thrs.
    score_tad(ds, np.linspace(0, 1, 11).round(1), ['map'], outDir)    
    h5f = h5_open_archive(os.path.join(outDir, 'scoring_results.h5'))
    data = h5_extract_system_iou_scores(h5f)          
    aData = h5_extract_activity_iou_scores(h5f)
    return [data, aData]

def test_pr_levels(tmpdir):
    data, aData = tad_scoring_run('testdata/tad_ref_smoothcurve.csv', 'testdata/tad_hyp_smoothcurve.csv', tmpdir)    
    hData = { entry[0]:{} for entry in aData }
    for entry in aData:
        hData[entry[0]][entry[1]] = entry[2]

    assert(len(hData) == 2)
        
    assert(hData['Closing']['ap_@iou_0.00'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.10'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.20'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.30'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.40'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.50'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.60'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.70'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.80'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.90'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_1.00'] == pytest.approx(1.0, 0.1))

    assert(hData['Closing-Temp-IOU']['ap_@iou_0.00'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.10'] == pytest.approx(0.75, 0.2))
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.20'] == pytest.approx(0.625, 0.3))
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.30'] == pytest.approx(0.5, 0.1))
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.40'] == pytest.approx(0.5, 0.1))
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.50'] == pytest.approx(0.375, 0.3))
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.60'] == pytest.approx(0.375, 0.3))    
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.70'] == pytest.approx(0.25, 0.2))
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.80'] == pytest.approx(0.25, 0.2))
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.90'] == pytest.approx(0.125, 0.3))
    assert(hData['Closing-Temp-IOU']['ap_@iou_1.00'] == pytest.approx(0.125, 0.3))

def test_no_score_region_exclusion(tmpdir):
    """
    tad_ref_sc_nsr has 3 NSR: 
    - 1 for Closing-Temp-IOU: 8 REF - 1 NSR = 7 REF final
    - 2 for Closing, all are matches
    - Closing has iou of 1.0 for all
    - Temp-IOU HYP has variable IoU: 1.0, 0.8, 0.4, 0.2, 0.1, 0.05, 0.01
    """
    data, aData = tad_scoring_run('testdata/tad_ref_sc_nsr.csv', 'testdata/tad_hyp_smoothcurve.csv', tmpdir)    
    hData = { entry[0]:{} for entry in aData }
    for entry in aData:
        hData[entry[0]][entry[1]] = entry[2]

    assert(len(hData) == 2)

    assert(hData['Closing']['ap_@iou_0.00'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.10'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.20'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.30'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.40'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.50'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.60'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.70'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.80'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_0.90'] == pytest.approx(1.0, 0.1))
    assert(hData['Closing']['ap_@iou_1.00'] == pytest.approx(1.0, 0.1))

    assert(hData['Closing-Temp-IOU']['ap_@iou_0.00'] == pytest.approx(1.0, 0.1))   # 7/7
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.10'] == pytest.approx(0.714, 0.3)) # 5/7
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.20'] == pytest.approx(0.571, 0.3)) # 4/7
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.30'] == pytest.approx(0.571, 0.3)) 
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.40'] == pytest.approx(0.571, 0.3)) 
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.50'] == pytest.approx(0.429, 0.3)) # 3/7
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.60'] == pytest.approx(0.429, 0.3)) 
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.70'] == pytest.approx(0.286, 0.3)) # 2/7
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.80'] == pytest.approx(0.286, 0.3)) 
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.90'] == pytest.approx(0.143, 0.3)) # 1/7
    assert(hData['Closing-Temp-IOU']['ap_@iou_1.00'] == pytest.approx(0.143, 0.3))

def test_missing_data(tmpdir):
    data, aData = tad_scoring_run('testdata/tad_ref_smoothcurve.csv', 'testdata/tad_hyp_sc_md.csv', tmpdir)    
    hData = { entry[0]:{} for entry in aData }
    for entry in aData:
        hData[entry[0]][entry[1]] = entry[2]

    assert(len(hData) == 2)

    # 7/8 = 0.875
    assert(hData['Closing']['ap_@iou_0.00'] == pytest.approx(0.875, 0.3))
    assert(hData['Closing']['ap_@iou_0.10'] == pytest.approx(0.875, 0.3))
    assert(hData['Closing']['ap_@iou_0.20'] == pytest.approx(0.875, 0.3))
    assert(hData['Closing']['ap_@iou_0.30'] == pytest.approx(0.875, 0.3))
    assert(hData['Closing']['ap_@iou_0.40'] == pytest.approx(0.875, 0.3))
    assert(hData['Closing']['ap_@iou_0.50'] == pytest.approx(0.875, 0.3))
    assert(hData['Closing']['ap_@iou_0.60'] == pytest.approx(0.875, 0.3))
    assert(hData['Closing']['ap_@iou_0.70'] == pytest.approx(0.875, 0.3))
    assert(hData['Closing']['ap_@iou_0.80'] == pytest.approx(0.875, 0.3))
    assert(hData['Closing']['ap_@iou_0.90'] == pytest.approx(0.875, 0.3))
    assert(hData['Closing']['ap_@iou_1.00'] == pytest.approx(0.875, 0.3))

    assert(hData['Closing-Temp-IOU']['ap_@iou_0.00'] == pytest.approx(0.875, 0.3)) # 7/8
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.10'] == pytest.approx(0.75, 0.2)) # 6/8
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.20'] == pytest.approx(0.604, 0.3)) # off based on CS
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.30'] == pytest.approx(0.458, 0.3)) 
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.40'] == pytest.approx(0.458, 0.3)) 
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.50'] == pytest.approx(0.312, 0.3)) 
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.60'] == pytest.approx(0.312, 0.3)) 
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.70'] == pytest.approx(0.167, 0.3)) 
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.80'] == pytest.approx(0.167, 0.3)) 
    assert(hData['Closing-Temp-IOU']['ap_@iou_0.90'] == pytest.approx(0.167, 0.3)) 
    assert(hData['Closing-Temp-IOU']['ap_@iou_1.00'] == pytest.approx(0.125, 0.3)) #1/8