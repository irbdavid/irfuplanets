import os

import pytest

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

pytest.importorskip("irfuplanets.maven")


def test_spice_update():
    from irfuplanets.spice import update_all_kernels

    # update_all_kernels("MAVEN", quota="5m")
    update_all_kernels("MAVEN", test=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="No data.")
@pytest.mark.skip(reason="No access to most recent data")
def test_lpw_recent_data():
    from irfuplanets.maven.lpw import lpw_l2_load
    from irfuplanets.time import now

    t0 = now()
    dt = 86400.0 * 10.0
    d = lpw_l2_load(t0 - dt, t0 - 2 * dt)

    assert d is not None


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="No data.")
def test_lpw_old_data():
    from irfuplanets.maven.lpw import lpw_l2_load
    from irfuplanets.time import spiceet

    t0 = spiceet("2015-01-01T00:00")
    dt = 86400.0 * 10.0
    d = lpw_l2_load(t0, t0 + 2 * dt)

    assert d is not None
