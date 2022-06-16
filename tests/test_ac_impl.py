import fad21
from fad21.scoring import score_ac
from fad21.validation import validate_ac
from fad21.datatypes import Dataset
from fad21.io import *
from pathlib import Path
import pytest
import io

# Validates scorer-implementation w/ specific classification use-cases.

def scoring_run(refFile, hypFile, topk, outDir):
    """ Loader Func """
    ds = Dataset(load_ref(refFile), load_hyp(hypFile))    
    score_ac(ds, ['map'], topk, outDir)
    h5f = h5_open_archive(os.path.join(outDir, 'scoring_results.h5'))
    data = h5_extract_system_scores(h5f)          
    aData = h5_extract_activity_scores(h5f)
    return [data, aData]

 
def test_3ref_single_1fp(tmpdir):
    """ USE-CASE: 3 classes, 1fp @ top confidence

    video_file_id  activity_id_pred  gt    conf  activity_id_gt
    1711_1         person_rubs_eyes  0.0   0.7   person_twirls
    1711_2         person_twirls     1.0   0.6   person_twirls
    1711_0         person_twirls     1.0   0.4   person_twirls
    7132_2         person_rubs_eyes  1.0   0.6   person_rubs_eyes
    7132_1         person_rubs_eyes  1.0   0.5   person_rubs_eyes
    7132_0         person_rubs_eyes  1.0   0.4   person_rubs_eyes
    8665_2         person_jumps      1.0   0.6   person_jumps
    8665_1         person_jumps      1.0   0.5   person_jumps
    8665_0         person_jumps      1.0   0.4   person_jumps

    Results in following Confusion Matrix & P/R

    [2, 1, 0],
    [0, 3, 0],
    [0, 0, 3]
    ['person_twirls', 'person_rubs_eyes', 'person_jumps']

            activity_id  precision    recall
          person_twirls       1.00  0.666667
       person_rubs_eyes       0.75  1.000000
           person_jumps       1.00  1.000000)

    > AP Breakdown: (non-interp. aP) := sum_n (r_n-r_n-1)*p_n 
    >   python:= -np.sum(np.diff(recall) * np.array(precision)[:-1])

 	precision                    recall                         activity_id 	 thresholds
 	[0.667, 0.5, 0.0, 1.0]       [1.0, 0.5, 0.0, 0.0]           person_twirls 	 [0.4, 0.6, 0.7]
 	[0.75, 0.667, 0.5, 0.0, 1.0] [1.0, 0.667, 0.333, 0.0, 0.0]  person_rubs_eyes [0.4, 0.5, 0.6, 0.7]
 	[1.0, 1.0, 1.0, 1.0]         [1.0, 0.667, 0.333, 0.0]       person_jumps     [0.4, 0.5, 0.6]

    > person_twirls   : 0.667*0.5 + 0.5+0.5 = 0.3333+ 0.25 + 0 + 0 ~= 0.583
    > person_rubs_eyes: 
    """
    data, aData = scoring_run('testdata/ac_3ref.csv', 'testdata/ac_3hyp1fp.csv', 0, tmpdir)
    hData = {}
    for entry in aData:
        hData[entry[0]] = entry[2]
    assert(len(hData) == 3)
    # resulting aP == P/R values here
    assert(hData['person_twirls'] == pytest.approx(0.583, 0.01))
    assert(hData['person_rubs_eyes'] == pytest.approx(0.639, 0.1))
    assert(hData['person_jumps'] == pytest.approx(1.0, 0.1))


def test_3ref_single_2fp(tmpdir):
    """ USE-CASE: 3 classes, 2fp @ top confidence

    video_file_id  activity_id_pred  gt    conf  activity_id_gt
    1711_1         person_rubs_eyes  0.0   0.7   person_twirls
    1711_2         person_twirls     1.0   0.6   person_twirls
    1711_0         person_twirls     1.0   0.4   person_twirls
    7132_2         person_rubs_eyes  1.0   0.6   person_rubs_eyes
    7132_1         person_rubs_eyes  1.0   0.5   person_rubs_eyes
    7132_0         person_rubs_eyes  1.0   0.4   person_rubs_eyes
    8665_2         person_twirls     0.0   0.6   person_jumps
    8665_1         person_jumps      1.0   0.5   person_jumps
    8665_0         person_jumps      1.0   0.4   person_jumps

    Results in following Confusion Matrix & P/R

    [2, 1, 0],
    [0, 3, 0],
    [1, 0, 2],
    ['person_twirls', 'person_rubs_eyes', 'person_jumps']

             activity_id  precision    recall
           person_twirls   0.666667  0.666667 <- ! aP(interp) = 0.5
        person_rubs_eyes   0.750000  1.000000
            person_jumps   1.000000  0.666667

    [person_twirls]
    precision                           recall               thr             gt
    [0.5, 0.3333333333333333, 0.0, 1.0] [1.0, 0.5, 0.0, 0.0] [0.4, 0.6, 0.7] [0.0, 1.0, 1.0, 0.0]
    """
    data, aData = scoring_run('testdata/ac_3ref.csv', 'testdata/ac_3hyp2fp.csv', 0, tmpdir)
    hData = {}
    for entry in aData:
        hData[entry[0]] = entry[2]
    assert(len(hData) == 3)
    # resulting aP == P/R values here
    assert(hData['person_twirls'] == pytest.approx(0.417, 0.01))
    assert(hData['person_rubs_eyes'] == pytest.approx(0.639, 0.01))
    assert(hData['person_jumps'] == pytest.approx(0.583, 0.01))

def test_3ref_single_1fp1md(tmpdir):
    """ USE-CASE: 3 classes, 1fp @ top confidence + 1 missing video-id

    video_file_id  activity_id_pred     gt    conf  activity_id_gt
    1711_1 	     person_rubs_eyes     0.0   0.7   person_twirls
    1711_2 	     person_twirls        1.0   0.6   person_twirls
    1711_0 	     person_twirls        1.0   0.4   person_twirls
    7132_2 	     person_rubs_eyes     1.0   0.6   person_rubs_eyes
    7132_1 	     person_rubs_eyes     1.0   0.5   person_rubs_eyes
    7132_0 	     person_rubs_eyes     1.0   0.4   person_rubs_eyes
    8665_1 	     __missed_detection__ 0.0   1.0   person_jumps
    8665_2 	     person_jumps         1.0   0.6   person_jumps
    8665_0 	     person_jumps         1.0   0.4   person_jumps

    Results in following Confusion Matrix & P/R
    [2, 1, 0, 0],
    [0, 3, 0, 0],
    [0, 0, 2, 1],
    [0, 0, 0, 0] 
    ['person_twirls', 'person_rubs_eyes', 'person_jumps', '__missed_detection__']

             activity_id  precision    recall
           person_twirls       1.00  0.666667
        person_rubs_eyes       0.75  1.000000
            person_jumps       1.00  0.666667
    __missed_detection__       0.00  0.000000)
    """
    data, aData = scoring_run('testdata/ac_3ref.csv', 'testdata/ac_3hyp1fp1md.csv', 0, tmpdir)
    hData = {}
    for entry in aData:
        hData[entry[0]] = entry[2]
    assert(len(hData) == 3)
    # resulting aP == P/R values here
    assert(hData['person_twirls'] == pytest.approx(0.583, 0.01))
    assert(hData['person_rubs_eyes'] == pytest.approx(0.639, 0.1))
    assert(hData['person_jumps'] == pytest.approx(0.583, 0.01))

def test_3ref_multihyp_1fp_multi(tmpdir):
    """ USE-CASE: 3 Classes w/ 3 hyp each w/ 1fp @ top confidence
    
    video_file_id  activity_id_pred 	gt    conf  activity_id_gt
    1711_2         person_rubs_eyes   0.0   0.9   person_twirls
    1711_2         person_twirls      1.0   0.8   person_twirls
    1711_2         person_twirls      1.0   0.7   person_twirls
    1711_1         person_twirls      1.0   0.6   person_twirls
    1711_1         person_twirls      1.0   0.5   person_twirls
    1711_1         person_twirls      1.0   0.4   person_twirls
    1711_0         person_twirls      1.0   0.3   person_twirls
    1711_0         person_twirls      1.0   0.2   person_twirls
    1711_0         person_twirls      1.0   0.1   person_twirls
    7132_2         person_rubs_eyes   1.0   0.9   person_rubs_eyes
    7132_2         person_rubs_eyes   1.0   0.8   person_rubs_eyes
    7132_2         person_rubs_eyes   1.0   0.7   person_rubs_eyes
    7132_1         person_rubs_eyes   1.0   0.6   person_rubs_eyes
    7132_1         person_rubs_eyes   1.0   0.5   person_rubs_eyes
    7132_1         person_rubs_eyes   1.0   0.4   person_rubs_eyes
    7132_0         person_rubs_eyes   1.0   0.3   person_rubs_eyes
    7132_0         person_rubs_eyes   1.0   0.2   person_rubs_eyes
    7132_0         person_rubs_eyes   1.0   0.1   person_rubs_eyes
    8665_2         person_jumps       1.0   0.9   person_jumps
    8665_2         person_jumps       1.0   0.8   person_jumps
    8665_2         person_jumps       1.0   0.7   person_jumps
    8665_1         person_jumps       1.0   0.6   person_jumps
    8665_1         person_jumps       1.0   0.5   person_jumps
    8665_1         person_jumps       1.0   0.4   person_jumps
    8665_0         person_jumps       1.0   0.3   person_jumps
    8665_0         person_jumps       1.0   0.2   person_jumps
    8665_0         person_jumps       1.0   0.1   person_jumps

    Results in following Confusion Matrix & P/R
    [8, 1, 0],
    [0, 9, 0],
    [0, 0, 9]
    ['person_twirls', 'person_rubs_eyes', 'person_jumps']

         activity_id  precision    recall
       person_twirls        1.0  0.888889
    person_rubs_eyes        0.9  1.000000
        person_jumps        1.0  1.000000
    """
    data, aData = scoring_run('testdata/ac_3ref.csv', 'testdata/ac_3hyp1fp_multi.csv', 0, tmpdir)
    hData = {}
    for entry in aData:
        hData[entry[0]] = entry[2]
    assert(len(hData) == 3)
    # resulting aP == P/R values here
    assert(hData['person_twirls'] == pytest.approx(0.771, 0.01))
    assert(hData['person_rubs_eyes'] == pytest.approx(0.786, 0.01))
    assert(hData['person_jumps'] == pytest.approx(1.0, 0.1))


def test_3ref_single_1ma(tmpdir):
    """ USE-CASE: 3 classes, 1 missed activity

    video_file_id  activity_id_pred  gt    conf  activity_id_gt    
    1711_0         person_twirls     1.0   0.4   person_twirls
    1711_1         na                0.0   na    person_twirls
    1711_2         person_twirls     1.0   0.6   person_twirls
    7132_2         person_rubs_eyes  1.0   0.6   person_rubs_eyes
    7132_1         person_rubs_eyes  1.0   0.5   person_rubs_eyes
    7132_0         person_rubs_eyes  1.0   0.4   person_rubs_eyes    
    8665_1         person_jumps      1.0   0.5   person_jumps
    8665_0         person_jumps      1.0   0.4   person_jumps
    8665_2         person_jumps      1.0   0.6   person_jumps
    8665_2         na                0.0   na    person_jumps

    Results in following Confusion Matrix & P/R

    [2, 1, 0],
    [0, 3, 0],
    [0, 0, 3]
    ['person_twirls', 'person_rubs_eyes', 'person_jumps']

            activity_id  precision    recall
          person_twirls       1.00  0.666667
       person_rubs_eyes       0.75  1.000000
           person_jumps       1.00  1.000000)    
    """
    data, aData = scoring_run('testdata/ac_3ref.csv', 'testdata/ac_3hyp1ma.csv', 0, tmpdir)
    hData = {}
    for entry in aData:
        hData[entry[0]] = entry[2]
    assert(len(hData) == 3)
    # resulting aP == P/R values here
    assert(hData['person_twirls'] == pytest.approx(0.583, 0.01))
    assert(hData['person_rubs_eyes'] == pytest.approx(1.0, 0.1))
    assert(hData['person_jumps'] == pytest.approx(1.0, 0.1))