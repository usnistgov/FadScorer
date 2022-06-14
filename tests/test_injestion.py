import fad21
import pytest
from fad21.io import load_ref, load_hyp

def load_tiny():
    return fad21.load_ref("testdata/ac_ref_2x3.csv")

def test_load_1():
    df = load_tiny()
    assert len(df) == 6


def test_faulty_header():
    with pytest.raises(OSError):
        load_ref("testdata/faulty_header.csv")    

def test_header_ignoring_extra_columns():
    ref = fad21.load_ref("testdata/ac_ref_header_extra.csv")
    assert 'video_file_id' in ref
    assert 'activity_id' in ref
    hyp = fad21.load_hyp("testdata/ac_hyp_header_extra.csv")
    assert 'video_file_id' in hyp
    assert 'activity_id' in hyp
    assert 'confidence_score' in hyp

def test_header_ignoring_spaces():    
    ref = fad21.load_ref("testdata/ac_ref_header_spaces.csv")
    assert 'video_file_id' in ref
    assert 'activity_id' in ref