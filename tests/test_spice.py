import os

import numpy as np
import pytest

import irfuplanets.spice as sp


def test_check_update_lsk_kernel():
    lsk_filename = sp.check_update_lsk_kernel()

    assert os.path.exists(lsk_filename), "No LSK file made/found"


@pytest.mark.skip("Tested at s/c module level instead")
def test_update_all_kernels():
    sp.update_all_kernels(test=True)


def test_spice_wrapper():
    def f(t):
        return np.array(
            (
                t,
                t,
                t,
            )
        )

    ff = sp.spice_wrapper(f)
    time = np.arange(10.0)
    res = ff(time)
    assert res.shape == (3, 10), f"Bad shape: {res.shape}"


def test_loaded_kernels():
    sp.describe_loaded_kernels
