import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('refinement',Path(__file__).resolve().parents[1]/'source/path_policy_refinement.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def test_conditioning_leaves_inactive_features_unchanged():
    old=np.ones(8);squares=np.array([.01,1,100,0,1,1,1,1])*100
    factor,rms=p.direction_scale(old,squares,100)
    assert np.allclose(factor[:4],[4,1,.25,0])
    before=np.arange(8,dtype=float)
    directions=p.conditioned_directions(np.random.default_rng(42),factor)
    for sign in [-1,1]:
        assert np.all((before+sign*.01*directions)[:,3]==before[3])


def test_matched_arms_receive_identical_perturbations():
    factor=np.array([4,3,1,1,1,0,1,1])
    a=p.conditioned_directions(np.random.default_rng(9101),factor)
    b=p.conditioned_directions(np.random.default_rng(9101),factor)
    assert np.array_equal(a,b)


def test_reject_nonfinite_statistics():
    try:p.direction_scale(np.ones(8),np.full(8,np.nan),100)
    except ValueError:return
    raise AssertionError('Nonfinite observations must not set exploration scales')
