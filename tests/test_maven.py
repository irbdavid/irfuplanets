import pytest

# def test_import():
#     import irfuplanets.maven
#     assert irfuplanets.maven.maven_http_manager is not None

pytest.importorskip("irfuplanets.maven")


def test_lpw_recent_data():
    from irfuplanets.maven.lpw import lpw_l2_load
    from irfuplanets.time import now

    t0 = now()
    dt = 86400.0 * 10.0
    d = lpw_l2_load(t0 - dt, t0 - 2 * dt)

    assert d is not None


def test_lpw_old_data():
    from irfuplanets.maven.lpw import lpw_l2_load
    from irfuplanets.time import spiceet

    t0 = spiceet("2015-01-01T00:00")
    dt = 86400.0 * 10.0
    d = lpw_l2_load(t0, t0 + 2 * dt)

    assert d is not None

    assert d is not None
