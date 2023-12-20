import os
from functools import wraps

import cdflib
import matplotlib.pyplot as plt
import numpy as np
import spiceypy
from matplotlib.colors import LogNorm
from spiceypy.utils.exceptions import SpiceyError

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


def spice_wrapper(n=1):
    """Wrapper for spice functions that handles array inputs, and fills NaNs
    on failed calculations"""

    def actual_decorator(f):
        def g(t):
            try:
                return f(t)
            except SpiceyError:
                return np.repeat(np.nan, n)

        @wraps(f)
        def inner(time):
            if hasattr(time, "__iter__"):
                return np.vstack([g(t) for t in time]).T
            else:
                return g(time)

        return inner

    return actual_decorator


@spice_wrapper(n=3)
def ram_angles(time):
    """Return the angle between SC-Y and the ram direction (0. = Y to ram).
    Note: Need to make sure you have correct kernels loaded."""
    p = spiceypy.spkezr("MAVEN", time, "IAU_MARS", "NONE", "MARS")[0][3:]
    r = spiceypy.pxform("IAU_MARS", "MAVEN_SPACECRAFT", time)
    a = spiceypy.mxv(r, p)

    e = np.arctan(np.sqrt(a[1] ** 2.0 + a[2] ** 2.0) / a[0]) * 180.0 / np.pi
    f = np.arctan(np.sqrt(a[0] ** 2.0 + a[2] ** 2.0) / a[1]) * 180.0 / np.pi
    g = np.arctan(np.sqrt(a[0] ** 2.0 + a[1] ** 2.0) / a[2]) * 180.0 / np.pi
    if e < 0.0:
        e = e + 180.0
    if f < 0.0:
        f = f + 180.0
    if g < 0.0:
        g = g + 180.0

    return np.array((e, f, g))


@spice_wrapper(n=3)
def sun_angles(time):
    """Return the angle between SC-Y and the sun direction (0. = Y to sun).
    Note: Need to make sure you have correct kernels loaded."""
    a = spiceypy.spkpos("SUN", time, "MAVEN_SPACECRAFT", "NONE", "MAVEN")[0]

    e = np.arctan(np.sqrt(a[1] ** 2.0 + a[2] ** 2.0) / a[0]) * 180.0 / np.pi
    f = np.arctan(np.sqrt(a[0] ** 2.0 + a[2] ** 2.0) / a[1]) * 180.0 / np.pi
    g = np.arctan(np.sqrt(a[0] ** 2.0 + a[1] ** 2.0) / a[2]) * 180.0 / np.pi
    if e < 0.0:
        e = e + 180.0
    if f < 0.0:
        f = f + 180.0
    if g < 0.0:
        g = g + 180.0

    return np.array((e, f, g))


def lpw_l2_load(
    start, finish, kind="lpnt", http_manager=None, cleanup=False, verbose=None
):
    """Finds and loads LPW L2 data"""

    import irfuplanets.maven

    if http_manager is None:
        http_manager = irfuplanets.maven.maven_http_manager
    kind = kind.lower()

    t = start
    year, month = utcstr(t, "ISOC").split("-")[0:2]
    year = int(year)
    month = int(month)
    #  Each month:
    files = []
    while t < finish:
        # print year, month
        files.extend(
            http_manager.query(
                "lpw/l2/%04d/%02d/mvn_lpw_l2_%s_*_v*_r*.cdf"
                % (year, month, kind),
                start=start,
                finish=finish,
                version_function=lambda x: (
                    x[0],
                    float(x[1]) + float(x[2]) / 100.0,
                ),
                date_function=lambda x: yyyymmdd_to_spiceet(x[0]),
                verbose=verbose,
            )
        )
        month += 1
        if month > 12:
            month = 1
            year += 1
        t = spiceet("%d-%02d-01T00:00" % (year, month))

    # Check for duplicates:
    if len(files) != len(set(files)):
        raise ValueError(
            "Duplicates appeared in files to load: " + ", ".join(files)
        )

    if cleanup:
        print("LPW L2 cleanup complete")
        return

    if not files:
        raise IOError("No data found")

    for f in sorted(files):
        if not os.path.exists(f):
            raise IOError("%s does not exist" % f)

    time_key = "epoch"

    if kind == "lpnt":
        output = dict(time=None, ne=None, te=None, usc=None)
        for f in sorted(files):
            c = cdflib.CDF(f)
            if output["time"] is None:
                # inx =
                output["time"] = c[time_key]
                output["ne"] = c["data"][:, 0]
                output["te"] = c["data"][:, 1]
                output["usc"] = c["data"][:, 2]
            else:
                output["time"] = np.hstack((output["time"], c[time_key]))

                for v, i in zip(("ne", "te", "usc"), (0, 1, 2)):
                    output[v] = np.hstack((output[v], c["data"][:, i]))

    elif kind == "wn":
        output = dict(time=None, ne=None)
        for f in sorted(files):
            print(f)
            c = cdflib.CDF(f)
            if output["time"] is None:
                # inx =
                output["time"] = c[time_key]
                output["ne"] = c["data"]
            else:
                output["time"] = np.hstack((output["time"], c[time_key]))
                output["ne"] = np.hstack((output["ne"], c["data"]))

                # for v, i in zip(('ne', 'te', 'usc'), (0,1,2)):
                #     output[v] = np.hstack((output[v],
                #   c.varget('data'][:,i])))

    elif kind == "wspecact":
        output = dict(time=None, spec=None, freq=None)
        for f in sorted(files):
            print(f)
            c = cdflib.CDF(f)

            if output["time"] is None:
                output["time"] = c[time_key]
                output["spec"] = c["data"].T
                output["freq"] = c["freq"][0, :]
            else:
                output["time"] = np.hstack((output["time"], c[time_key]))
                output["spec"] = np.hstack((output["spec"], c["data"].T))

        # print 'Warning: spectra output is not interpolated!'

    elif kind == "wspecpas":
        output = dict(time=None, spec=None, freq=None)
        for f in sorted(files):
            print(f)
            c = cdflib.CDF(f)

            if output["time"] is None:
                output["time"] = c[time_key]
                output["spec"] = c["data"].T
                output["freq"] = c["freq"][0, :]
            else:
                output["time"] = np.hstack((output["time"], c[time_key]))
                output["spec"] = np.hstack((output["spec"], c["data"].T))
            # print 'Warning: spectra output is not interpolated!'

    elif kind == "lpiv":
        output = dict(time=None, current=None, volt=None)
        for f in sorted(files):
            c = cdflib.CDF(f)

            if output["time"] is None:
                output["time"] = c[time_key]
                output["current"] = c["data"].T
                output["volt"] = c["volt"].T
            else:
                output["time"] = np.hstack((output["time"], c[time_key]))
                output["current"] = np.hstack((output["current"], c["data"].T))
                output["volt"] = np.hstack((output["volt"], c["volt"].T))

    elif kind == "we12":
        output = dict(time=None, we12=None)
        for f in sorted(files):
            c = cdflib.CDF(f)
            if output["time"] is None:
                # inx =
                output["time"] = c[time_key]
                output["we12"] = c["data"]
            else:
                output["time"] = np.hstack((output["time"], c[time_key]))

                output["we12"] = np.hstack((output["we12"], c["data"]))

    else:
        raise ValueError("Input kind='%s' not recognized" % kind)

    output["time"] = cdflib.cdfepoch.to_datetime(output["time"])
    output["time"] = datetime64_to_spiceet(output["time"])

    return output


def lpw_plot_spec(
    s,
    ax=None,
    cmap=None,
    norm=None,
    max_frequencies=512,
    max_times=2048,
    fmin=None,
    fmax=None,
    labels=True,
    colorbar=True,
    full_resolution=False,
):
    """Transform and plot a spectra dictionary generated by lpw_load.
    Doesn't interpolate linearly, but just rebins data.  Appropriate for
    presentation purposes, but don't do science with the results.
    """

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    if cmap is None:
        cmap = "Spectral_r"
    if norm is None:
        norm = LogNorm(1e-16, 1e-8)

    img_obj = plt.pcolormesh(
        s["time"], s["freq"], s["spec"], cmap=cmap, norm=norm
    )

    plt.yscale("log")
    # plt.xlim(t0, t1)

    if labels:
        plt.ylabel("f / Hz")
    if colorbar:
        cbar = plt.colorbar(cax=make_colorbar_cax())
        cbar.set_label(r"V$^2$ m$^{-2}$ Hz$^{-1}$")
    else:
        cbar = None

    return img_obj, cbar


def lpw_plot_iv(
    s,
    boom=1,
    ax=None,
    cmap=None,
    norm=None,
    start=None,
    finish=None,
    voltage=None,
    labels=True,
    colorbar=True,
    log_abs=True,
):
    """Plot LP IV sweeps as a time series."""

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    if cmap is None:
        plt.set_cmap("viridis")
        if log_abs is False:
            plt.set_cmap("RdBu_r")
        cmap = plt.get_cmap()
        cmap.set_bad("grey")

    if not norm:
        norm = plt.Normalize(1e-7, 1e-7)
        if log_abs:
            norm = LogNorm(1e-9, 1e-5)

    d = s["current"]
    if log_abs:
        d = np.abs(d)

    img_obj = plt.pcolormesh(s["time"], s["volt"], d, cmap=cmap, norm=norm)

    if labels:
        plt.ylabel(r"U$_{Bias}$ / V")

    if colorbar:
        cbar = plt.colorbar(cax=make_colorbar_cax(ax))
        cbar.set_label(r"i / A")
    else:
        cbar = None

    return img_obj, cbar


def cleanup(start=None, finish=None):
    if not start:
        start = spiceet("2014-09-22T00:00")
    if not finish:
        finish = now()

    # Cleanup commands
    lpw_l2_load(start, finish, cleanup=True, verbose=True)


if __name__ == "__main__":
    plt.close("all")
    start = spiceet("2015-04-23T06:00")
    finish = start + 86400.0 / 2.0

    # finish = start + 86400. * 2. - 1.

    xl = np.array((start, finish))
    xo = np.array((1, 1))

    o = lpw_l2_load(kind="wspecact", start=start, finish=finish)
    o2 = lpw_l2_load(kind="wn", start=start, finish=finish)
    o3 = lpw_l2_load(kind="lpnt", start=start, finish=finish)

    inx = np.isfinite(o2["ne"])
    ne_w = np.interp(o3["time"], o2["time"][inx], o2["ne"][inx])

    fig, axs = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(8, 12),
        gridspec_kw=dict(height_ratios=(5, 2, 2, 2)),
    )

    plt.subplots_adjust(hspace=0.01)
    plt.sca(axs[0])
    lpw_plot_spec(o, colorbar=False, full_resolution=True, fmin=2e4)
    # plt.ylim(1e4, 2e6)
    plt.plot(o2["time"], 8980 * np.sqrt(o2["ne"]), "k+")
    plt.plot(o3["time"], 8980 * np.sqrt(o3["ne"]), "r+")
    # plt.plot(o3['time'], 8980*np.sqrt(ne_w), 'b*')

    plt.sca(axs[1])
    plt.plot(o3["time"], np.sqrt(o3["ne"] / ne_w), "k.")
    plt.plot(xl, xo * 1.0, "b--")
    plt.plot(xl, xo * 2.0, "b--")
    plt.plot(xl, xo * np.sqrt(2), "b:")
    plt.yscale("log")
    plt.ylim(0.1, 10.0)
    plt.ylabel(r"$f_{pe,IV} / f_{pe,W}$")

    plt.sca(axs[2])
    plt.plot(o3["time"], o3["te"] / 11604.0, "k.")
    plt.yscale("log")
    plt.ylabel("Te / eV")

    plt.sca(axs[3])
    plt.plot(o3["time"], 0.069 * np.sqrt(o3["te"] / 11604.0 / o3["ne"]), "k.")
    plt.plot(o3["time"], 0.069 * np.sqrt(o3["te"] / 11604.0 / ne_w), "r.")

    plt.plot(xl, xo * 0.0063 / 2.0, "b--")
    plt.plot(xl, xo * 0.05 / 2.0, "b:")
    plt.plot(xl, xo * 0.4 / 2.0, "b--")
    plt.ylim(5e-5, 2e-3)
    plt.yscale("log")
    plt.ylabel(r"$\lambda_D$/m")

    setup_time_axis()

    plt.figure()
    plt.scatter(
        0.069 * np.sqrt(o3["te"] / 11604.0 / o3["ne"]),
        np.sqrt(o3["ne"] / ne_w),
        c=o3["time"],
        marker=".",
        edgecolor="none",
    )
    plt.ylabel(r"$f_{pe,IV} / f_{pe,W}$")
    plt.xlabel(r"$\lambda_D$/m")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(5e-5, 2e-3)
    plt.ylim(0.1, 10.0)
    x = np.array((5e-5, 2e-3))
    plt.plot(x, 10.0 ** (-0.24 * np.log10(x) - 0.7))

    plt.show()
