import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from irfuplanets.planets.mars import constants
from irfuplanets.plot import plot_planet

__author__ = "David Andrews"
__copyright__ = "Copyright 2023, David Andrews"
__license__ = "MIT"
__email__ = "david.andrews@irfu.se"


def _hold_xylim(func):
    def wrapped(*args, **kwargs):
        xlim = plt.xlim()
        ylim = plt.ylim()
        out = func(*args, **kwargs)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        return out

    return wrapped


@_hold_xylim
def plot_mpb_model_sza(
    fmt="k-", model="TROTIGNON06", shadow=True, return_values=False, **kwargs
):
    if model == "VIGNES00":
        t = np.arange(0.0, np.pi, 0.01)
        x = 0.78 + 0.96 * np.cos(t) / (1 + 0.9 * np.cos(t))
        y = 0.96 * np.sin(t) / (1 + 0.92 * np.cos(t))
    elif model == "EDBERG08":
        t = np.arange(0.0, np.pi, 0.01)
        x = 0.86 + 0.9 * np.cos(t) / (1 + 0.92 * np.cos(t))
        y = 0.9 * np.sin(t) / (1 + 0.92 * np.cos(t))
    elif model == "DUBININ06":
        t = np.arange(0.0, np.pi, 0.01)
        x = 0.70 + 0.96 * np.cos(t) / (1 + 0.9 * np.cos(t))
        y = 0.96 * np.sin(t) / (1 + 0.9 * np.cos(t))
    elif model == "TROTIGNON06":
        t = np.arange(0.0, np.pi, 0.01)
        # x > 0:
        x1 = 0.64 + 1.08 * np.cos(t) / (1 + 0.77 * np.cos(t))
        y1 = 1.08 * np.sin(t) / (1 + 0.77 * np.cos(t))
        # x < 0:
        x2 = 1.60 + 0.528 * np.cos(t) / (1 + 1.009 * np.cos(t))
        y2 = 0.528 * np.sin(t) / (1 + 1.009 * np.cos(t))

        inx1 = x1 > 0.0
        inx2 = x2 < 0.0

        x = np.hstack((x1[inx1], x2[inx2]))
        y = np.hstack((y1[inx1], y2[inx2]))

    alt = np.sqrt(x * x + y * y) - 1.0
    sza = np.arctan2(y, x) / np.pi * 180.0

    if shadow:
        # y = np.arange(0., 6000., 10.)
        y = 10.0 ** (np.linspace(0.0, 4.0, 256))
        x = 90.0 + 180.0 / np.pi * np.arccos(3376.0 / (3376.0 + y))

        if return_values:
            return ((sza, alt * constants.mars_mean_radius_km), (x, y))
        plt.plot(x, y, fmt[0] + "--", **kwargs)

    if return_values:
        return (sza, alt * constants.mars_mean_radius_km)

    plt.plot(sza, alt * constants.mars_mean_radius_km, fmt, **kwargs)


class SegmentedMSphere(object):
    """Segmentation of the Martian Magnetosphere, following Crider 03.
    Used to map coordinates or trajectories in MSO rho-x coordinates into
    magnetospheric regions, and interpolate positions between them.


    Examples
    --------
    >>> c = SegmentedMSphere()
    >>> c.plot(colorbar=False)
    >>> regions = c(sc_x, sc_rho)
    >>> inx, = np.where(regions == c.map["SOLAR_WIND"])
    """

    def __init__(self):
        super(SegmentedMSphere, self).__init__()

        self.x = np.arange(-5.0, 5, 0.01)  # mso x
        self.y = np.arange(0.0, 5, 0.01)  # mso rho

        self.nx = np.arange(self.x.shape[0])
        self.ny = np.arange(self.y.shape[0])

        self.extent = (self.x[0], self.x[-1], self.y[0], self.y[-1])

        self.map = dict()
        self.map[-1] = "BAD_DATA [2]"
        self.map[0] = "BAD_DATA [1]"
        self.map[1] = "SOLAR_WIND"
        self.map[2] = "MAGNETOSHEATH"
        self.map[3] = "MAGNETIC_PILE_UP_REGION"
        self.map[4] = "MAGNETOTAIL"
        self.map[5] = "DAYSIDE_IONOSPHERE"
        self.map[6] = "NIGHTSIDE_IONOSPHERE"
        self.map[7] = "TERMINATOR_IONOSPHERE"

        for k in list(self.map.keys()):
            v = self.map[k]
            self.map[v] = k

        # Build an image, value according to region using map values
        xx, yy = np.meshgrid(self.x, self.y)

        # Everything is the solar wind
        img = np.zeros((self.y.shape[0], self.x.shape[0]))
        img += self.map["SOLAR_WIND"]

        # Inside the bowshock
        x0 = 0.64
        rr = np.sqrt((xx - x0) ** 2.0 + yy**2.0)
        phi = np.arctan2(yy, xx - x0)
        r_bs = 2.04 / (1.0 + 1.03 * np.cos(phi))
        img[rr < r_bs] = self.map["MAGNETOSHEATH"]

        # Inside the MPB
        x0 = 0.78
        rr = np.sqrt((xx - x0) ** 2.0 + yy**2.0)
        phi = np.arctan2(yy, xx - x0)
        r_bs = 0.96 / (1.0 + 0.9 * np.cos(phi))
        img[rr < r_bs] = self.map["MAGNETIC_PILE_UP_REGION"]

        # Sort out the sheath at large distances to form the tail
        img[(xx < -3.7) & (yy < 2.2)] = self.map["MAGNETIC_PILE_UP_REGION"]

        # Tail
        # Everything in shadow
        img[(xx < 0.0) & (yy < 1.0)] = self.map["MAGNETOTAIL"]

        ion_alt = 1.15
        rr = np.sqrt(xx**2.0 + yy**2.0)
        # Dayside ionosphere
        img[(xx < 0.0) & (rr < ion_alt)] = self.map["DAYSIDE_IONOSPHERE"]

        # Nightside ionosphere
        img[(xx > 0.0) & (rr < ion_alt)] = self.map["NIGHTSIDE_IONOSPHERE"]

        # Terminator ionosphere (approx?)
        img[(np.abs(xx) < 0.2) & (rr < ion_alt)] = self.map["TERMINATOR_IONOSPHERE"]

        self.img = img
        self.xx = xx
        self.yy = yy

        self.vmin = np.min(self.img)
        self.vmax = np.max(self.img)

        self.cmap = ListedColormap(
            [
                "#fff983",  # sw
                "#ff756e",  # sheath
                "#ffe8ca",  # mpr #b3ffc8
                "#a3d879",  # tail
                "#0079a5",  # dayside ionosphere
                "#bfedfd",  # nightside ionosphere
                "#00bafb",
            ]
        )  # terminator ionosphere

        # Just in case
        self.cmap.set_under("black")
        self.cmap.set_over("black")

        bounds = [1, 2, 3, 4, 5, 6, 7, 8]
        self.norm = BoundaryNorm(bounds, self.cmap.N)

    def plot(self, ax=None, label=True, colorbar=False, **kwargs):
        """Plot the segmented magnetosphere image.

        Parameters
        ----------
        ax : matplotlib.axes, optional
            Axes to plot on, by default `plt.gca()`
        label : bool, optional
            Add axes labels, by default True
        colorbar : bool, optional
            Add colorbar, by default False
        **kwargs: passed to `plt.imshow`
        """
        if not ax:
            ax = plt.gca()
        else:
            plt.sca(ax)

        plt.imshow(
            self.img,
            extent=self.extent,
            origin="lower",
            interpolation="nearest",
            aspect="equal",
            norm=self.norm,
            cmap=self.cmap,
            **kwargs,
        )
        if colorbar:
            plt.colorbar()

        plot_planet(zorder=102, facecolor="white")

        plt.xlim(-3, 4)
        plt.ylim(0, 5)
        if label:
            plt.xlabel(r"$x_{MSO}$ / $R_M$")
            plt.ylabel(r"$\rho_{MSO}$ / $R_M$")

    def __call__(self, x, y):
        """Interpolate `x', `y' coordinates onto grid, and return map values
        there."""

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        xinx = np.interp(x, self.x, self.nx).astype(np.int32)
        yinx = np.interp(y, self.y, self.ny).astype(np.int32)

        out = np.zeros_like(x) - 1
        good = (xinx > 0) & (xinx < self.nx[-1]) & (yinx > 0) & (yinx < self.ny[-1])

        out[good] = self.img[yinx[good], xinx[good]]
        return out

    def describe(self, regions):
        """Describe the mapped regions.

        Parameters
        ----------
        regions : array of identified regions from `__call__`
        """
        n = float(regions.shape[0]) / 100.0
        for i in range(-1, 8):
            print("%s: %f%%" % (self.map[i], np.sum(regions == i) / n))
