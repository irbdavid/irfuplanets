import os

import cdflib
import matplotlib.pyplot as plt
import numpy as np
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


def load_swia_l2_summary(
    start,
    finish,
    kind="onboardsvymom",
    http_manager=None,
    delete_others=True,
    cleanup=False,
    verbose=None,
):
    import irfuplanets.maven

    kind = kind.lower()

    if not delete_others:
        raise RuntimeError("Not written yet")

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
                "swi/l2/%04d/%02d/mvn_swi_l2_%s_*_v*_r*.cdf"
                % (year, month, kind),
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
        raise ValueError(
            "Duplicates appeared in files to load: " + ", ".join(files)
        )

    if cleanup:
        print("SWIA L2 Cleanup complete")
        return

    for f in sorted(files):
        if not os.path.exists(f):
            raise IOError("%s does not exist" % f)

    if kind == "onboardsvyspec":
        output = {"time": None, "def": None}
        for f in sorted(files):
            c = cdflib.CDF(f)

            if output["time"] is None:
                output["time"] = c["epoch"]
                output["def"] = c["spectra_diff_en_fluxes"].T

                # Some weird formatting here:
                output["energy"] = np.array(
                    [
                        c["energy_spectra"][i]
                        for i in range(c["energy_spectra"].shape[0])
                    ]
                )
                output["energy"] = output["energy"][::-1]
            else:
                output["time"] = np.hstack((output["time"], c["epoch"]))
                output["def"] = np.hstack(
                    (output["def"], c["spectra_diff_en_fluxes"].T)
                )

                if output["energy"].shape != c["energy_spectra"].shape:
                    raise ValueError("Energy range has changed!")

        output["def"] = output["def"][::-1, :]

    elif kind == "onboardsvymom":
        output = {
            "time": None,
            "velocity": None,
            "density": None,
            "temperature": None,
            "quality_flag": None,
        }
        for f in sorted(files):
            c = cdflib.CDF(f)

            if output["time"] is None:
                output["time"] = c["epoch"]
                output["quality_flag"] = c["quality_flag"]
                output["density"] = c["density"]
                output["velocity"] = c["velocity_mso"].T
                output["temperature"] = c["temperature_mso"].T
            else:
                output["time"] = np.hstack((output["time"], c["epoch"]))
                output["quality_flag"] = np.hstack(
                    (output["quality_flag"], c["quality_flag"])
                )
                output["density"] = np.hstack(
                    (output["density"], c["density"])
                )
                output["velocity"] = np.hstack(
                    (output["velocity"], c["velocity"].T)
                )
                output["temperature"] = np.hstack(
                    (output["temperature"], c["temperature"].T)
                )

    else:
        raise ValueError("Input kind='%s' not recognized" % kind)

    output["time"] = cdflib.cdfepoch.to_datetime(
        output["time"],
    )
    output["time"] = datetime64_to_spiceet(output["time"])

    return output


def plot_swia_l2_summary(
    swia_data,
    max_times=4096,
    cmap=None,
    norm=None,
    labels=True,
    ax=None,
    colorbar=True,
):
    if "def" not in swia_data:
        print("No data given?")
        return

    d = swia_data["def"]
    t = swia_data["time"]

    if d.shape[1] > max_times:
        n = int(np.floor(d.shape[1] / max_times))
        d = d[:, ::n]
        t = t[::n]

    # extent = (t[0], t[-1], swia_data["energy"][0], swia_data["energy"][-1])

    if cmap is None:
        cmap = "Spectral_r"
    if norm is None:
        norm = LogNorm(1e6, 1e8)

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    # img = plt.imshow(
    #     d, extent=extent, interpolation='nearest', origin='lower',
    #     norm=norm, cmap=cmap, aspect='auto'
    # )
    img = plt.pcolormesh(t, swia_data["energy"], d, norm=norm, cmap=cmap)
    plt.yscale("log")
    # plt.xlim(t0, t1)
    plt.ylim(swia_data["energy"][0], swia_data["energy"][-1])

    if labels:
        plt.ylabel("E / eV")

    if colorbar:
        plt.colorbar(cax=make_colorbar_cax()).set_label("SWIA D.E.F.")

    return img


def cleanup(start=None, finish=None):
    if not start:
        start = spiceet("2014-09-22T00:00")
    if not finish:
        finish = now()

    # Cleanup commands
    load_swia_l2_summary(start, finish, cleanup=True)


if __name__ == "__main__":
    plt.close("all")
    t0 = spiceet("2015-01-08")
    t1 = t0 + 86400.0 * 2.0 + 1
    t1 = t0 + 86400.0 - 1.0

    # d = load_swia_l2_summary(t0, t1, kind='onboardsvyspec')
    # plot_swia_l2_summary(d)

    d = load_swia_l2_summary(t0, t1, kind="onboardsvymom")
    plt.subplot(211)
    plt.plot(d["time"], d["density"])
    plt.subplot(212)
    plt.plot(d["time"], d["velocity"][0], "r.")
    plt.plot(d["time"], d["velocity"][1], "g.")
    plt.plot(d["time"], d["velocity"][2], "b.")

    setup_time_axis()

    plt.show()
