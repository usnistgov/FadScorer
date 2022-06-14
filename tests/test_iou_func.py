import fad21
from fad21.metrics import *
import pytest

import logging
log = logging.getLogger(__name__)

def test_temp_iou_100():
    """ EDGE-CASE: start/end """
    x = compute_temp_iou(10,20,10,20)
    assert x == 1.0

# There are 6 overlap scenarios
def test_temp_iou_outside_before_start():
    x = compute_temp_iou(10,20,5,6)
    assert x == 0

def test_temp_iou_outside_before_start_at_start_edge():    
    x = compute_temp_iou(10,20,9,10)
    assert x == 0

def test_temp_iou_outside_after_end():
    x = compute_temp_iou(10,20,25,26)
    assert x == 0

def test_temp_iou_outside_after_end_at_end_edge():
    x = compute_temp_iou(10,20,20,22)
    assert x == 0

def test_temp_iou_before_at_33_percent():    
    x = compute_temp_iou(10,20,5,15)
    assert x == pytest.approx(0.33, 0.1)


def test_temp_iou_case_within_edge_start():    
    x = compute_temp_iou(10,20,5,15)
    assert x == 0.25


def test_temp_iou_case_within():    
    x = compute_temp_iou(10,20,15,17.5)
    assert x == 0.25


def test_temp_iou_case_within_edge_start():    
    x = compute_temp_iou(10,20,17.5, 20)
    assert x == 0.25


def test_temp_iou_case_after_at_33_percent():
    x = compute_temp_iou(10,20,15,25)
    assert x == pytest.approx(0.33, 0.1)