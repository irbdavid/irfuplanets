import os

import matplotlib.pyplot as plt
import numpy as np

from irfuplanets.maven import sdc_interface
from irfuplanets.time import now, setup_time_axis, spiceet, utcstr

__author__ = "David Andrews"
__copyright__ = "Copyright 2023, David Andrews"
__license__ = "MIT"
__email__ = "david.andrews@irfu.se"

ION_MASSES = {"H2": 2, "HE": 4, "O": 16, "O2": 32, "CO2": 44, "AR": 40}


def load_ngims_l2(
    start,
    finish,
    kind="ion",
    species="all",
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

    if species == "all":
        species = list(ION_MASSES.keys())

    t = start
    year, month = utcstr(t, "ISOC").split("-")[0:2]
    year = int(year)
    month = int(month)
    #  Each month:
    files = []

    version = {"ion": "08", "cso": "07", "csn": "08"}

    while t < finish:
        # A note here: Formats change with versions, stick to 8 for now
        files.extend(
            http_manager.query(
                "ngi/l2/%04d/%02d/mvn_ngi_l2_%s-abund-*_v%s_r*.csv"
                % (year, month, kind, version[kind]),
                start=start,
                finish=finish,
                version_function=lambda x: (
                    x[0],
                    float(version[kind]) + float(x[1]) / 100.0,
                ),
                date_function=lambda x: sdc_interface.yyyymmdd_to_spiceet(
                    x[0].split("_")[1]
                ),
                cleanup=cleanup,
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
        print("NGIMS L2 cleanup complete")

    if not files:
        raise IOError("No NGIMS data found")

    for f in sorted(files):
        if not os.path.exists(f):
            raise IOError("%s does not exist" % f)

    if kind == "ion":
        output = {}

        for f in sorted(files):
            if verbose:
                print(f)
            d = np.loadtxt(
                f,
                skiprows=1,
                delimiter=",",
                usecols=(0, 12, 14),
                converters={
                    0: lambda x: spiceet(x),
                    7: lambda x: float(x or "NaN"),
                    9: lambda x: float(x or "NaN"),
                },
            ).T
            # count = None

            for s in species:
                mass = ION_MASSES[s]
                (inx,) = np.where(d[1] == mass)
                # if count is None:
                #     count = inx.size
                # else:
                #     if count != inx.size:
                #         raise ValueError("Malformed file?")

                if s not in output:
                    output[s] = {}
                    output[s]["time"] = d[0, inx]
                    output[s]["density"] = d[2, inx]
                else:
                    output[s]["time"] = np.hstack(
                        (output[s]["time"], d[0, inx])
                    )
                    output[s]["density"] = np.hstack(
                        (output[s]["density"], d[2, inx])
                    )

    elif kind == "csn":
        output = {}

        for f in sorted(files):
            if verbose:
                print(f)
            d = np.loadtxt(
                f,
                skiprows=1,
                delimiter=",",
                usecols=(0, 12, 15),
                converters={
                    0: lambda x: spiceet(x),
                    12: lambda x: int(x),
                },
            ).T
            # count = None

            for s in species:
                mass = ION_MASSES[s]
                (inx,) = np.where(d[1] == mass)
                # if count is None:
                #     count = inx.size
                # else:
                #     if count != inx.size:
                #         raise ValueError("Malformed file?")

                if s not in output:
                    output[s] = {}
                    output[s]["time"] = d[0, inx]
                    output[s]["density"] = d[2, inx]
                else:
                    output[s]["time"] = np.hstack(
                        (output[s]["time"], d[0, inx])
                    )
                    output[s]["density"] = np.hstack(
                        (output[s]["density"], d[2, inx])
                    )

    else:
        raise ValueError("Input kind='%s' not recognized" % kind)

    raise NotImplementedError("Unchecked/untested")

    return output


def cleanup(start=None, finish=None):
    if not start:
        start = spiceet("2014-09-22T00:00")
    if not finish:
        finish = now()

    # Cleanup commands
    load_ngims_l2(start, finish, cleanup=True, verbose=True)


if __name__ == "__main__":
    plt.close("all")
    t0 = spiceet("2015-04-30T00:00")
    import irfuplanets.maven

    mgr = irfuplanets.maven.sdc_interface.maven_http_manager

    # t1 = t0 + 86400. * 2. + 1
    t1 = t0 + 86400.0 - 1.0

    # plot_ngims_l2_summary(d)
    # d = load_ngims_l2(t0, t1, kind='ion', http_manager=mgr, verbose=True)
    d = load_ngims_l2(t0, t1, kind="csn", http_manager=mgr, verbose=True)

    for s in list(ION_MASSES.keys()):
        try:
            plt.plot(d[s]["time"], d[s]["density"])
        except Exception as e:
            print(e)

    plt.yscale("log")

    setup_time_axis()

    plt.show()
