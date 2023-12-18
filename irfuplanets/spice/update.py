import os
import stat
import subprocess
import time

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


def _wget(server, path, verbose=True, cut_dirs=True, test=False, cmd=None):
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

    if verbose:
        print(std_out.strip(), std_err)


def check_update_lsk_kernel():
    # https://naif.jpl.nasa.gov/pub/naif/
    # generic_kernels/lsk/latest_leapseconds.tls

    top_data_dir = irfuplanets.config["irfuplanets"]["data_directory"]

    if not os.path.exists(top_data_dir):
        os.makedirs(top_data_dir)

    lsk_filename = top_data_dir + "latest_leapseconds.tls"

    server = "https://naif.jpl.nasa.gov"
    path = "/pub/naif/generic_kernels/lsk/latest_leapseconds.tls"

    time_now = time.time()
    try:
        age = time_now - os.stat(lsk_filename)[stat.ST_MTIME]
        print(f"Leapsecond file is {int(age / 86400)} days old")
    except IOError:
        print("No sign of a local leapseconds kernel file.")
        age = 1e99

    if age > (86400 * 100):
        print(f"Obtaining newest leap-seconds kernel from {server}...")

        orig_dir = os.getcwd()

        try:
            orig_dir = os.getcwd()
            os.chdir(top_data_dir)
            _wget(
                server,
                path,
                test=False,
                cmd=f"wget -O {lsk_filename}",
                cut_dirs=False,
            )

            # Wget doesn't set the right timestamp or something?
            os.utime(lsk_filename, (time_now, time_now))

        finally:
            os.chdir(orig_dir)

    return lsk_filename


def update_mex(server=None, basepath=None, local=None, test=False):
    import irfuplanets

    orig_dir = os.getcwd()

    if local is None:
        local = irfuplanets.config["mex"]["data_directory"] + "spice/"

    if server is None:
        server = "https://archives.esac.esa.int"

    if basepath is None:
        basepath = "/psa/ftp/MARS-EXPRESS/SPICE/MEX-E-M-SPICE-6-V2.0/"

    ops = [
        (basepath + f"DATA/{x}/", local + f"{x.lower()}/")
        for x in ("FK", "IK", "LSK", "PCK", "SCLK", "SPK", "DSK")
    ]

    ops.append(
        (
            basepath + "EXTRAS/ORBNUM/",
            local + "/orbnum/",
        ),
    )

    for path, local_dir in ops:
        print(f"Updating {server}{path} -> {local_dir}")
        try:
            orig_dir = os.getcwd()
            os.chdir(local_dir)
            _wget(server, path, test=test)
        finally:
            os.chdir(orig_dir)


def update_maven(server=None, basepath=None, local=None, test=False):
    import irfuplanets

    orig_dir = os.getcwd()

    if local is None:
        local = irfuplanets.config["maven"]["kernel_directory"]

    if server is None:
        server = "https://naif.jpl.nasa.gov"
    # https://naif.jpl.nasa.gov/pub/naif/pds/
    #        pds4/maven/maven_spice/spice_kernels/
    if basepath is None:
        basepath = "/pub/naif/pds/pds4/maven/maven_spice"

    ops = [
        (basepath + f"/spice_kernels/{x.lower()}/", local + f"{x.lower()}/")
        for x in ("FK", "IK", "LSK", "PCK", "SCLK", "SPK")
    ]

    ops.append(
        (
            basepath + "/miscellaneous/orbnum/",
            local + "/spk/",
            # Because, that was how it used to work
        ),
    )

    for path, local_dir in ops:
        print(f"Updating {server}{path} -> {local_dir}")
        try:
            orig_dir = os.getcwd()
            os.chdir(local_dir)
            _wget(server, path, test=test)
        finally:
            os.chdir(orig_dir)


def update_all_kernels(spacecraft="ALL", **kwargs):
    if spacecraft == "ALL":
        spacecraft = "MAVEN MEX"

    if "MAVEN" in spacecraft:
        update_maven(**kwargs)

    if "MEX" in spacecraft:
        update_mex(**kwargs)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        sc = str(sys.argv[len(sys.argv) - 1]).upper()
    else:
        sc = "ALL"

    update_all_kernels(spacecraft=sc)
