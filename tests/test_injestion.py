import fad21

def load_tiny():
    return fad21.load_ref("testdata/ac_ref_2x3.csv")

def test_load_1():
    df = load_tiny()
    assert len(df) == 6
