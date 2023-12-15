import os

import cdflib
import matplotlib.pyplot as plt

# import numpy as np
from matplotlib.colors import LogNorm

from irfuplanets.maven.sdc_interface import yyyymmdd_to_spiceet
from irfuplanets.plot import make_colorbar_cax
from irfuplanets.time import (
    datetime64_to_spiceet,
    now,
    setup_time_axis,
    spiceet,
    utcstr,
)

__author__ = "David Andrews"
__copyright__ = "Copyright 2023, David Andrews"
__license__ = "MIT"
__email__ = "david.andrews@irfu.se"

STATIC_PRODUCTS = {
    "c0": "c0-64e2m",
    "c6": "c6-32e64m",
    "c8": "c8-32e16d",
}


def load_static_l2(
    start,
    finish,
    kind="c0",
    http_manager=None,
    delete_others=True,
    cleanup=False,
    verbose=None,
):
    raise NotImplementedError("Untried/untested!")
    import irfuplanets.maven

    kind = kind.lower()

    full_kind = STATIC_PRODUCTS[kind]

    if http_manager is None:
        http_manager = irfuplanets.maven.maven_http_manager

    t = start
    year, month = utcstr(t, "ISOC").split("-")[0:2]
    year = int(year)
    month = int(month)
    #  Each month:
    files = []
    while t < finish:
        files.extend(
            http_manager.query(
                "sta/l2/%04d/%02d/mvn_sta_l2_%s_*_v*_r*.cdf" % (year, month, full_kind),
                start=start,
                finish=finish,
                version_function=lambda x: (
                    x[0],
                    float(x[1]) + float(x[2]) / 100.0,
                ),
                date_function=lambda x: yyyymmdd_to_spiceet(x[0]),
                cleanup=cleanup,
                verbose=verbose,
            )
        )
        month += 1
        if month > 12:
            month = 0o1
            year += 1
        t = spiceet("%d-%02d-01T00:00" % (year, month))

    # Check for duplicates:
    if len(files) != len(set(files)):
        raise ValueError("Duplicates appeared in files to load: " + ", ".join(files))

    if cleanup:
        print("static L2 Cleanup complete")
        return

    for f in sorted(files):
        if not os.path.exists(f):
            raise IOError("%s does not exist" % f)

    if kind == "c0":
        output = {"blocks": [], "static_kind": "c0"}
        for f in sorted(files):
            c = cdflib.CDF(f)

            # data = c["eflux"]
            # last_ind = None
            # last_block_start = None
            # N = data.shape[0]
            # print(c['eflux'].shape, c['energy'].shape, c['epoch'].shape)
            output["blocks"].append([c["epoch"], c["energy"], c["eflux"]])

            c.close()

    else:
        raise ValueError("Input kind='%s' not recognized" % kind)

    for b in output["blocks"]:
        b[0] = cdflib.cdfepoch.to_datetime(b[0], to_np=True)
        b[0] = datetime64_to_spiceet(b[0])

    return output


def plot_static_l2_summary(
    static_data,
    plot_type="Energy",
    max_times=4096,
    cmap=None,
    norm=None,
    labels=True,
    ax=None,
    colorbar=True,
):
    if "static_kind" not in static_data:
        raise ValueError("Data supplied not from static?")

    if not static_data["static_kind"] == "c0":
        raise ValueError("I only know about C0, for now")

    if cmap is None:
        cmap = "Spectral_r"
    if norm is None:
        norm = LogNorm(1e3, 1e9)

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    imgs = []

    for time, energy, data in static_data["blocks"]:
        img = plt.pcolormesh(
            time,
            energy,
            data,
            interpolation="nearest",
            origin="lower",
            norm=norm,
            cmap=cmap,
            aspect="auto",
        )
        imgs.append(img)
    plt.yscale("log")

    if labels:
        plt.ylabel("E / eV")

    if colorbar:
        plt.colorbar(cax=make_colorbar_cax()).set_label("static D.E.F.")

    return imgs


def cleanup(start=None, finish=None):
    if not start:
        start = spiceet("2014-09-22T00:00")
    if not finish:
        finish = now()

    # Cleanup commands
    load_static_l2(start, finish, cleanup=True)


if __name__ == "__main__":
    plt.close("all")
    t0 = spiceet("2015-01-08")
    t1 = t0 + 86400.0 * 2.0 + 1
    t1 = t0 + 86400.0 - 1.0

    # d = load_static_l2(t0, t1, kind='onboardsvyspec')
    # plot_static_l2_summary(d)

    d = load_static_l2(t0, t1, kind="c0")
    plot_static_l2_summary(d)
    # plt.subplot(211)
    # plt.plot(d['time'], d['density'])
    # plt.subplot(212)
    # plt.plot(d['time'], d['velocity'][0], 'r.')
    # plt.plot(d['time'], d['velocity'][1], 'g.')
    # plt.plot(d['time'], d['velocity'][2], 'b.')
    #
    setup_time_axis()

    plt.show()
