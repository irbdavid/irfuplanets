import os
import subprocess
import time
from pathlib import Path

import irfuplanets

__author__ = "David Andrews"
__copyright__ = "Copyright 2023, David Andrews"
__license__ = "MIT"
__email__ = "david.andrews@irfu.se"

# All except CK:
# https://naif.jpl.nasa.gov/pub/naif/pds/data/mex-e_m-spice-6-v2.0/mexsp_2000/DATA/

# Orbnum
# https://naif.jpl.nasa.gov/pub/naif/pds/data/mex-e_m-spice-6-v2.0/mexsp_2000/EXTRAS/
# https://archives.esac.esa.int/psa/ftp/MARS-EXPRESS/SPICE/MEX-E-M-SPICE-6-V2.0/DATA/


def _wget(
    server,
    path,
    verbose=True,
    cut_dirs=True,
    test=False,
    cmd=None,
    quota=None,
    reject=None,
):
    if cmd is None:
        cmd = "wget -m -nH -nv -np"

    # m = mirror, = -r, -N -l inf --no-remove-listing
    # nH no host name
    # nv no verbose
    # np no parent (don't go outside the requested dir, only within it)

    if cut_dirs:
        n = path.count("/")

        if path[-1] == "/":
            n -= 1
        cmd += f" --cut-dirs={n}"

    # if test:
    #     cmd += " --spider"

    if quota is not None:
        cmd += f" --quota={quota}"

    if reject is not None:
        cmd += f" --reject='{reject}'"

    cmd += f" {server}{path}"
    print(cmd)

    if test:
        print(cmd)
        return

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True,
    )
    std_out, std_err = process.communicate()

    if verbose or (process.returncode != 0):
        print(std_out.strip(), std_err)


def check_update_lsk_kernel():
    # https://naif.jpl.nasa.gov/pub/naif/
    # generic_kernels/lsk/latest_leapseconds.tls

    top_data_dir = irfuplanets.config["irfuplanets"]["data_directory"]
    top_data_dir = Path(top_data_dir)

    # assert top_data_dir.is_dir(), \
    #       f"Bad config data_directory: {top_data_dir}"

    if not top_data_dir.exists:
        top_data_dir.mkdir(parents=True)

    lsk_filename = top_data_dir.absolute() / "latest_leapseconds.tls"

    server = "https://naif.jpl.nasa.gov"
    path = "/pub/naif/generic_kernels/lsk/latest_leapseconds.tls"

    time_now = time.time()
    try:
        age = time_now - lsk_filename.stat().st_mtime
        print(f"Leapsecond file is {int(age / 86400)} days old")
    except IOError:
        print("No sign of a local leapseconds kernel file.")
        age = 1e99

    if age > (86400 * 100):
        print(f"Obtaining newest leap-seconds kernel from {server}...")

        orig_dir = None

        try:
            orig_dir = Path.cwd()
            os.chdir(orig_dir)
            _wget(
                server,
                path,
                test=False,
                cmd=f"wget -O {lsk_filename}",
                cut_dirs=False,
                verbose=True,
            )

            # Wget doesn't set the right timestamp or something?
            os.utime(lsk_filename, (time_now, time_now))

        finally:
            os.chdir(orig_dir)

    return lsk_filename


def _update_sc(ops, server=None, test=False, **kwargs):
    for path, local_dir in ops:
        print(f"Updating {server}{path} -> {local_dir}")

        if not local_dir.exists:
            if not test:
                local_dir.mkdir(parents=True)

        try:
            orig_dir = Path.cwd()
            os.chdir(local_dir)
            _wget(server, path, test=test, **kwargs)
        finally:
            os.chdir(orig_dir)


def update_mex(server=None, basepath=None, local=None, test=False, **kwargs):
    import irfuplanets

    if local is None:
        local = (
            Path(irfuplanets.config["mex"]["data_directory"]).absolute()
            / "spice/"
        )

    if server is None:
        server = "https://archives.esac.esa.int"

    if basepath is None:
        basepath = "/psa/ftp/MARS-EXPRESS/SPICE/MEX-E-M-SPICE-6-V2.0/"

    ops = [
        (basepath + f"DATA/{x}/", local / f"{x.lower()}/")
        for x in ("FK", "IK", "LSK", "PCK", "SCLK", "SPK", "DSK")
    ]

    ops = []

    ops.append(
        (
            basepath + "EXTRAS/ORBNUM/",
            local / "orbnum/",
        ),
    )

    _update_sc(ops, server=server, test=test, **kwargs)


def update_maven(server=None, basepath=None, local=None, test=False, **kwargs):
    import irfuplanets

    if local is None:
        local = Path(
            irfuplanets.config["maven"]["kernel_directory"]
        ).absolute()

    if server is None:
        server = "https://naif.jpl.nasa.gov"
    # https://naif.jpl.nasa.gov/pub/naif/pds/
    #        pds4/maven/maven_spice/spice_kernels/
    if basepath is None:
        basepath = "/pub/naif/pds/pds4/maven/maven_spice"

    ops = [
        (basepath + f"/spice_kernels/{x.lower()}/", local / f"{x.lower()}/")
        for x in ("FK", "IK", "LSK", "PCK", "SCLK", "SPK")
    ]

    ops.append(
        (
            basepath + "/miscellaneous/orbnum/",
            local / "spk/",
            # Because, that was how it used to work
        ),
    )

    _update_sc(ops, server=server, test=test, **kwargs)


def update_all_kernels(spacecraft="ALL", **kwargs):
    if spacecraft == "ALL":
        spacecraft = "MAVEN MEX"

    if "MAVEN" in spacecraft:
        update_maven(**kwargs)

    if "MEX" in spacecraft:
        update_mex(**kwargs)


def create_dirs(interactive=True, test=False):
    """Setup needed directories based on config"""
    cfg = irfuplanets.config

    for root_cfg in cfg:
        if root_cfg == "DEFAULT":
            continue
        for attr in cfg[root_cfg]:
            p = Path(cfg[root_cfg][attr])
            if p.exists():
                print(f"{p} already exists")
                continue

            if p.is_dir() and "directory" in attr.lower():
                if interactive:
                    val = input(f"Create {p} [Enter Y/N]? ")
                    if val.upper() != "Y":
                        continue
                else:
                    print(f"Making {p}")

                if test:
                    continue

                p.mkdir(parents=True)


def first_run(sc="ALL", interactive=True, test=False, **kwargs):
    print("Initial setup of irfuplanets directories...")
    create_dirs(interactive=interactive, test=test)

    print("Get a leapsecond kernel...")
    check_update_lsk_kernel()

    print("Get spice kernels...")
    update_all_kernels(spacecraft=sc, reject=r"\?C=")


if __name__ == "__main__":
    import sys

    print("Initial setup of irfuplanets directories...")

    first_run()

    print("Get a leapsecond kernel...")
    check_update_lsk_kernel()

    if len(sys.argv) > 1:
        sc = str(sys.argv[len(sys.argv) - 1]).upper()
    else:
        sc = "ALL"

    print("Get spice kernels...")
    update_all_kernels(spacecraft=sc, reject=r"\?C=")
