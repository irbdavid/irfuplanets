import os

import pytest

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# pytest.importorskip("irfuplanets.mex")


def test_spice_update():
    from irfuplanets.spice import update_all_kernels

    # update_all_kernels("MAVEN", quota="5m")
    update_all_kernels("MEX", test=True)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="No data.")
def test_aisreview():
    from irfuplanets.mex.ais import AISReview

    a = AISReview(8020)
    a.main()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="No data.")
def test_digitizations():
    import irfuplanets.mex.ais as ais

    result = ais.compute_all_digitizations(8020, filename="tmp.dat")
    assert "failed" not in result.lower(), f"Failed: {result}"
