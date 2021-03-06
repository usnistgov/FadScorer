import fad21
import pytest
from fad21.io import load_ref, load_hyp
from pathlib import Path

def get_root():
    return Path(__file__).parent

def load_tiny():
    return fad21.load_ref(get_root() / "testdata/ac_ref_ec_1.csv")

def test_load_1():    
    df = load_tiny()
    assert len(df) == 360

def test_load_1():    
    df = load_tiny()
    assert len(df) == 360

def test_faulty_header():
    with pytest.raises(OSError):
        load_ref(get_root() / "testdata/faulty_header.csv")    

def test_tad_ref_load():
    ref = fad21.load_ref(get_root() / "testdata/tad_ref_smoothcurve.csv")
    assert 'video_file_id' in ref
    assert 'activity_id' in ref
    assert 'frame_start' in ref
    assert 'frame_end' in ref

def test_tad_hyp_load():
    hyp = fad21.load_hyp(get_root() / "testdata/tad_hyp_smoothcurve.csv")
    assert 'video_file_id' in hyp
    assert 'activity_id' in hyp
    assert 'confidence_score' in hyp
    assert 'frame_start' in hyp
    assert 'frame_end' in hyp

def test_header_ignoring_extra_columns():
    ref = fad21.load_ref(get_root() / "testdata/ac_ref_header_extra.csv")
    assert 'video_file_id' in ref
    assert 'activity_id' in ref
    hyp = fad21.load_hyp(get_root() / "testdata/ac_hyp_header_extra.csv")
    assert 'video_file_id' in hyp
    assert 'activity_id' in hyp
    assert 'confidence_score' in hyp

def test_header_ignoring_spaces():    
    ref = fad21.load_ref(get_root() / "testdata/ac_ref_header_spaces.csv")
    assert 'video_file_id' in ref
    assert 'activity_id' in ref

