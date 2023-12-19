import matplotlib.pyplot as plt
import numpy as np
import pytest

import irfuplanets.plot as p


@pytest.mark.parametrize("left", (True, False))
@pytest.mark.parametrize("half", (True, False))
@pytest.mark.parametrize("upper", (True, False))
def test_make_colorbar_cax(left, half, upper):
    fig = plt.figure()
    plt.plot((0.0, 1.0))

    cax = p.make_colorbar_cax(left=left, half=half, upper=upper)
    assert cax in fig.axes, "axes not created?"


def test_make_colorbar_cax_horizontal():
    fig = plt.figure()
    plt.plot((0.0, 1.0))

    cax = p.make_colorbar_cax_horizontal()

    assert cax in fig.axes, "axes not created?"


@pytest.mark.parametrize(
    "n",
    (
        1,
        2,
        3,
        10,
        30,
    ),
)
def test_n_random_colors(n):
    assert len(p.n_random_colors(n)) == n, "n mismatch"


@pytest.mark.parametrize(
    "n",
    (
        1,
        2,
        3,
        10,
        30,
    ),
)
def test_n_colors(n):
    assert len(p.n_colors(n)) == n, "n mismatch"


@pytest.mark.parametrize("sslat", (-90.0, -45.0, 0.0, 5.0))
@pytest.mark.parametrize("sslon", (0.0, 90.0))
@pytest.mark.parametrize("repeat", (True, False))
def test_terminator(sslat, sslon, repeat):
    plt.figure()
    x, y = p.terminator(sslat, sslon, repeat=repeat)
    assert x.shape == y.shape, "Shape mismatch"
    assert np.all(np.isfinite(x * y)), "No NaN should be present"


@pytest.mark.parametrize("radius", (0.1, 1.0, 2.0))
@pytest.mark.parametrize(
    "orientation", (None, "dawn", "dusk", "noon", "midnight")
)
def test_plot_planet(radius, orientation):
    plt.figure()
    v = p.plot_planet(radius=radius, orientation=orientation)
    c = plt.gca().patches

    assert len(v) > 0, f"Bad return value: {v}"

    # We registered all patches
    assert all([vv in c for vv in v]), "Patches missing"


def test_map_along_line():
    x = np.linspace(-1.0, 1.0, 64)
    y = np.sin(x * 2.0 * np.pi)
    q = x**2

    plt.figure()
    lc = p.map_along_line(x, y, q)

    assert lc is not None


@pytest.mark.parametrize("dx", (0.1, 0.5, 1.0, 10.0))
@pytest.mark.parametrize("x0", (0.0, 0.3))
@pytest.mark.parametrize("units", ("degrees", "radians", "natural"))
@pytest.mark.parametrize("nmax", (3, 5))
def test_circular_locator(dx, x0, units, nmax):
    plt.figure()
    scale = 1.0
    if units == "degrees":
        scale = 360.0

    if units == "radians":
        scale = np.pi * 2.0
    _dx = dx * scale
    _x0 = x0 * scale

    c = p.CircularLocator(units=units, nmax=nmax)
    c = plt.gca().xaxis.set_major_locator(c)
    plt.plot((_x0 - _dx), (_x0 + _dx), (-1.0, 1.0))


@pytest.mark.parametrize(
    "location", ("top left", "top right", "bottom left", "bottom right")
)
def test_corner_label(location):
    plt.figure()
    plt.plot((0.0, 1.0))
    i = iter("XYZ")
    p.corner_label("A", location=location)
    p.corner_label(i, location=location)


def test_boxplot():
    plt.figure()

    x = np.random.randn(64)
    y = np.random.randn(64)

    p.boxplot(x, y)


@pytest.mark.parametrize("top", (True, False))
def test_add_labelled_bar(top):
    plt.figure()
    start, finish = (-1.0, 1.0)
    t = np.linspace(start, finish, 512)
    q = t - start
    plt.plot(t, q)
    p.add_labelled_bar(t, q, top=top)
