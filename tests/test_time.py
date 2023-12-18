import numpy as np
import pytest

import irfuplanets.time as itime
import irfuplanets.time.time
from irfuplanets.time.time import VALID_FORMATS


def test_import():
    assert isinstance(itime.now(), float)


dt = [
    1.0,
    10.0,
    100.0,
    1000.0,
    3600.0,
    86400.0,
    86400.0 * 10.0,
    86400.0 * 100,
    86400.0 * 365,
    86400.0 * 365 * 10.0,
]

t0s = [
    0.0,
    1.0,
    2.0,
    59.0,
    60.0,
    100.0,
    3600.0,
    3600.0 * 2.0,
    3600.0 * 24.0 + 1.0,
    86400.0 * (365 - 1),
    86400.0 * (365 + 0),
    86400.0 * (365 + 1),
    86400.0 * (365) * 10,
    86400.0 * (365) * 100,
]


@pytest.mark.parametrize("duration", dt)
def test_setup_time_axis(duration):
    import matplotlib.pyplot as plt

    t0 = itime.spiceet("2016-07-07T05:12")
    t1 = t0 + duration

    x = [0.0, 1.0]
    plt.plot((t0, t1), x)
    itime.setup_time_axis()


fmts = [f.lower() for f in VALID_FORMATS]


@pytest.mark.parametrize("dt", dt)
@pytest.mark.parametrize("fmt", fmts)
def test_time_convert(fmt, dt):
    t = 0.0 + dt

    if fmt == "auto":
        pytest.skip(f"Not needed for {fmt}")

    if fmt == "spiceet":
        pytest.skip(f"Not needed for {fmt}")

    func = getattr(irfuplanets.time.time, f"spiceet_to_{fmt}")
    ifunc = getattr(irfuplanets.time.time, f"{fmt}_to_spiceet")

    assert hasattr(func, "__call__"), "Not callable"
    assert hasattr(ifunc, "__call__"), "Not callable"

    f = func(t)
    tf = ifunc(f)

    precision = 0.5 * np.abs(tf - t) / (tf + t)
    assert precision < 1e-6, f"Precision {precision} exceeds tolerance"
