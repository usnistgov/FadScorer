import fad21
from fad21.scoring import score_tad
from fad21.datatypes import Dataset
from fad21.io import *
from pathlib import Path
import pytest, io, logging

# Tests: scoring-execution, hd5 gen, csv-gen, map-score
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def tad_scoring_run(refFile, hypFile, iouThr, outDir):
    ds = Dataset(load_tad_ref(refFile), load_tad_hyp(hypFile))
    score_tad(ds, ['map'], iouThr, outDir)
    fn = os.path.join(outDir, 'scoring_results.h5')
    h5f = h5_open_archive(fn)
    data = h5_extract_system_iou_scores(h5f)
    aData = h5_extract_activity_iou_scores(h5f)
    alignData = h5_extract_alignment(fn)
    ensure_output_dir(outDir)
#    write_system_level_scores(os.path.join(outDir, 'system_scores.csv'), data)
#    write_activity_level_scores(os.path.join(outDir, 'activity_scores.csv'), aData)                        
    write_alignment_file(os.path.join(outDir, 'alignments.csv'), alignData)

def test_scoring_smoothcurve(tmpdir):
    
    tad_scoring_run(
        'testdata/tad_ref_usecases.csv', 
        'testdata/tad_hyp_usecases.csv',
        [0.5], 
        tmpdir)
    assert(
        list(io.open(tmpdir + "/alignments.csv")) == 
        list(io.open("testrefs/tad_alignment/alignments.csv")))