"""
Code for plotting / interpolating the field topology Maps made by Brain, 07.
Brain is sensible, uses East Longitudes :)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav

from irfuplanets.plot import CircularLocator, make_colorbar_cax

ALL_DESCRIPTIONS = [
    "num_pads_night",
    "percent_open_day",
    "percent_void_day",
    "num_pads_day",
    "percent_open_night",
    "percent_trapped_night",
    "percent_open_terminator",
    "percent_closed_terminator",
    "num_pads_terminator",
    "percent_trapped_day",
    "percent_void_terminator",
    "percent_closed_night",
    "percent_unattached_day",
    "percent_unattached_terminator",
    "percent_unattached_night",
    "percent_trapped_terminator",
    "percent_void_night",
    "percent_closed_day",
]


def get_file():
    fname = "misc_mars/Mars_topology_maps_MGS_mapping_orbit.sav"

    dirs = [
        "/home/dja/data/",
        "/Volumes/MDATA/data/",
        "/Volumes/FC2TB/data/",
        "/Volumes/ETC/data/",
        "/Users/dave/data/",
    ]

    for d in dirs:
        if os.path.exists(d + fname):
            return d + fname

    raise IOError("No suitable file found in mounted volumes")


def plot_field_topology(
    description,
    ax=None,
    colorbar=True,
    fname=None,
    limits=True,
    circ_axis=True,
    vmin=0.0,
    vmax=100.0,
    labels=True,
    full_range=True,
    **kwargs,
):
    """
    Brain field topology maps.  Ammended/edited by Rob as well, so include him.
    """

    if description not in ALL_DESCRIPTIONS:
        raise KeyError(
            "%s not one of accepted descriptions: %s"
            % (description, str(ALL_DESCRIPTIONS))
        )

    if ax is None:
        ax = plt.gca()
    plt.sca(ax)

    if fname is None:
        fname = get_file()

    data = readsav(fname, verbose=False)

    img = data[description]

    if full_range:
        img2 = np.hstack((img, img, img))
        out = plt.imshow(
            img2,
            origin="lower",
            extent=(-360.0, 720.0, -90.0, 90.0),
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
    else:
        out = plt.imshow(
            img,
            origin="lower",
            extent=(0.0, 360.0, -90.0, 90.0),
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

    if limits:
        plt.xlim(0.0, 360.0)
        plt.ylim(-90.0, 90)

    if circ_axis:
        ax.xaxis.set_major_locator(CircularLocator())
        ax.yaxis.set_major_locator(CircularLocator())

    if labels:
        plt.ylabel("Latitude / deg")
        plt.xlabel("Longitude / deg")

    if colorbar:
        c = plt.colorbar(cax=make_colorbar_cax())
        if labels:
            c.set_label(description.replace("_", " "))

        out = (out, c)

    return out


def interpolate_along_trajectory(
    latitude, longitude, description=None, fname=None
):
    """Longitudes are necessarily positive east, as in the idlsaveile"""
    if latitude.shape != longitude.shape:
        raise ValueError("Shape mismatch")

    if not fname:
        fname = get_file()

    data = readsav(fname, verbose=False)

    inx_lat = (np.floor(latitude) + 90).astype(np.int)
    inx_lon = np.floor(((longitude % 360.0) + 360.0) % 360.0).astype(np.int)

    if description:
        img = data[description]

        return img[inx_lat, inx_lon]

    else:
        output = dict()

        for d in list(data.keys()):
            if "percent" not in d:
                continue

            output[d] = data[d][inx_lat, inx_lon]

        return output


if __name__ == "__main__":
    plt.close("all")

    if True:
        plt.figure()
        plot_field_topology("percent_open_night", full_range=True)

        plt.figure()
        plot_field_topology("percent_open_night", full_range=False)

        plt.show()

    if False:
        for d in ALL_DESCRIPTIONS:
            plt.figure()
            plot_field_topology(d)
            plt.title(d)

        lat = np.arange(-90.0, 90.0, 0.25)
        lon = np.arange(0.0, 360.0, 0.5)

        plt.figure()
        plt.plot(
            lon, interpolate_along_trajectory(lat, lon, ALL_DESCRIPTIONS[0])
        )

        plt.figure()
        trajs = interpolate_along_trajectory(lat, lon)
        for d in trajs:
            plt.plot(lon, trajs[d])

        data = readsav(get_file(), verbose=False)
        plt.show()
