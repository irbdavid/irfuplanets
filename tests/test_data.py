import numpy as np

import irfuplanets.data as data

NPTS = 64

angles_deg = np.random.randn(NPTS) * 180.0
angles_rad = angles_deg * np.pi / 180
randints = np.random.randint(-5, 5, NPTS)

interp_t = np.linspace(-1.0, 1.0, NPTS)
interp_t_new = np.linspace(-1.0, 1.0, NPTS * 2)
interp_t_new2 = np.linspace(-1.0, 1.0, NPTS // 2)

# interp_t_new = np.linspace(-1.5, 1.5, NPTS * 2)
# interp_t_new2 = np.linspace(-1.5, 1.5, NPTS // 2)
interp_y = np.sin(interp_t * 2.0 * np.pi)

# Big gap 0.5, little gap 0.05
interp_gaps = (interp_t > 0.0) & (interp_t < 0.5)
interp_gaps = interp_gaps | (interp_t > 0.71) & (interp_t < 0.76)
interp_gaps[0] = True
interp_gaps_y = interp_y * 1.0
interp_gaps_y[interp_gaps] = np.nan


def test_deg_unwrap():
    u = data.deg_unwrap(angles_deg)

    # New values, with cycles added, but zeroth same
    new = angles_deg + np.abs(randints) * 360.0
    new[0] = angles_deg[0]
    u2 = data.deg_unwrap(new)

    np.testing.assert_allclose(u, u2, err_msg="Not close enough")


def test_modpos():
    u = data.modpos(angles_deg)
    u2 = data.modpos(angles_deg + randints * 360.0)

    np.testing.assert_allclose(u, u2, err_msg="Not close enough")


def test_remove_non_edge_intersecting():
    img = np.zeros((8, 8), dtype=int)

    img[4, 4] = 1

    # Fill a few so that it does something
    img[0, 0] = 1
    img[0, 7] = 1
    img[7, 0] = 1
    img[7, 7] = 1
    res = data.remove_none_edge_intersecting(img)

    assert np.sum(img) != np.sum(res), "Array unchanged"


def test_interp_safe():
    """Basic tests only"""

    ok = np.isfinite(interp_gaps_y)

    unchanged = data.interp_safe(
        interp_t,
        interp_t,
        interp_gaps_y,
        max_step=0.000001,
    )

    np.testing.assert_allclose(
        interp_gaps_y, unchanged, err_msg="Should be the same"
    )

    result = data.interp_safe(
        interp_t_new, interp_t[ok], interp_gaps_y[ok], max_step=0.1
    )

    assert np.any(np.isnan(result)), "Should have NaN in"

    assert np.isnan(result[0]), "Left should be NaN"

    (big_gap,) = np.where((interp_t_new > 0.0) & (interp_t_new < 0.5))
    (little_gap,) = np.where((interp_t_new > 0.7) & (interp_t_new < 0.75))

    # print(interp_t)
    print(interp_t_new)
    print(result)
    # print(result[big_gap])

    # Gap should be Nan
    assert np.isnan(
        np.sum((result[big_gap]))
    ), "Should be some NaNs in the big_gap"

    print(interp_t, interp_gaps_y)
    print(little_gap, result[little_gap])
    assert np.all(np.isfinite(result[little_gap])), "Should be all finite"

    # Interpolate over all big gaps
    # No NaNs remain, except for "leftmost"
    result2 = data.interp_safe(
        interp_t_new, interp_t[ok], interp_gaps_y[ok], max_step=5.0
    )
    i = 0
    while True:
        if np.isnan(result2[i]):
            i += 1
        else:
            break

    assert np.all(np.isfinite(result2[i:])), "Should be all finite"
