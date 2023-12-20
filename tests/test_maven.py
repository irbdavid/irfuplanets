import os

import pytest

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# pytest.importorskip("irfuplanets.maven")

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


# @pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="No data.")
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


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="No data.")
def test_swia_old_data():
    import matplotlib.pyplot as plt

    from irfuplanets.maven.swia import (
        load_swia_l2_summary,
        plot_swia_l2_summary,
    )
    from irfuplanets.time import spiceet

    t0 = spiceet("2015-01-01T00:00")
    dt = 86400.0 * 10.0
    d = load_swia_l2_summary(t0, t0 + 2 * dt, kind="onboardsvyspec")

    assert d is not None, "Nothing returned."

    plt.figure()
    img = plot_swia_l2_summary(d)

    assert img is not None, "Nothing plotted."


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="No data.")
def test_swea_old_data():
    import matplotlib.pyplot as plt

    from irfuplanets.maven.swea import (
        load_swea_l2_summary,
        plot_swea_l2_summary,
    )
    from irfuplanets.time import spiceet

    t0 = spiceet("2015-01-01T00:00")
    dt = 86400.0 * 10.0
    d = load_swea_l2_summary(t0, t0 + 2 * dt)

    assert d is not None, "Nothing returned."

    plt.figure()
    img = plot_swea_l2_summary(d)

    assert img is not None, "Nothing plotted."
