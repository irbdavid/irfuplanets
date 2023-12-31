"""Routines for manipulating SPICE kernels, and other SPICE related things."""

import ftplib
import glob
import json
import os
import re
from functools import wraps

import numpy as np
import spiceypy
from spiceypy.utils.exceptions import SpiceyError

__author__ = "David Andrews"
__copyright__ = "Copyright 2023, David Andrews"
__license__ = "MIT"
__email__ = "david.andrews@irfu.se"


def spice_wrapper(f):
    """Wrapper for a function :f: that will make necessary multiple calls for
    vectorized inputs."""

    @wraps(f)
    def inner(time):
        if hasattr(time, "__iter__"):
            return np.array([f(t) for t in time]).T
        else:
            return f(time)

    return inner


def better_spice_wrapper(n=1):
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


def describe_loaded_kernels(kind="all"):
    """Print a list of loaded spice kernels of :kind:"""

    all_kinds = ("spk", "pck", "ck", "ek", "text", "meta")
    if kind == "all":
        for k in all_kinds:
            describe_loaded_kernels(k)
        return

    n = spiceypy.ktotal(kind)
    if n == 0:
        print("No loaded %s kernels" % kind)
        return

    print("Loaded %s kernels:" % kind)
    for i in range(n):
        data = spiceypy.kdata(i, kind, 100, 10, 100)
        print("\t%d: %s" % (i, data[0]))


class SpiceManager(object):
    """A class for managing spice kernels - updates a local copy from a
    remote server, given a manifest file. Comparison of file versions not yet
    implemented - would be much more complicated. This code works fine if the
    ftp directory is only populated with the most recent versions of each file,
    e.g. MEX@ESAC, but not CASSINI@NAIF.
    """

    def __init__(self, config_file, verbose=False, silent=False):
        super(SpiceManager, self).__init__()
        self.config_file = config_file
        self.verbose = verbose
        self.silent = silent
        if self.verbose:
            self.silent = False
        self.config = None
        self.load_config_file()

        raise DeprecationWarning(
            """This functionality is pretty much obsolete, as NAIF's FTP server
            was shutdown in prior to 2023."""
        )

    def load_config_file(self):
        with open(self.config_file) as f:
            self.config = json.load(f)

        self.local_directory = self.config["local_directory"]
        self.kernels = self.config["kernels"]
        self.server_name = self.config["server_name"]
        self.server_path = self.config["server_path"]
        self.name = self.config["name"]

        if "$" in self.local_directory:
            if self.local_directory == "$HERE":
                self.local_directory = self.config_file.rsplit("/", 1)[0]
            else:
                self.local_directory = os.path.expandvars(self.local_directory)
                self.local_directory = os.path.expanduser(self.local_directory)

            if self.verbose:
                print("Updating local directory to " + self.local_directory)

        if not os.path.exists(self.local_directory):
            raise IOError(
                "Local directory %s does not exist" % self.local_directory
            )

    def update(self, dry_run=False):
        if not self.config:
            raise IOError("No configuration file loaded")

        if "NAIF" in self.server_name.upper():
            raise IOError(
                f"Will not be able to connect to {self.server_name}"
                " since it's no longer available.  Use instead"
                "     > irfuplanets.spice.update_all_kernels()"
                " which is not as fancy but [should] work."
            )

        ftp = ftplib.FTP(self.server_name)
        ftp.login()

        # remote_files = []
        for k in self.kernels:
            force_download = "$FORCE" in k
            if force_download:
                k = k.replace("$FORCE", "")

            fname = k.split("/")[-1]
            this_dir = "/".join(k.split("/")[:-1]) + "/"
            fname_v = fname.replace("$VERSION", "*")

            # check_version = "$VERSION" in k

            if self.verbose:
                print("CMD: " + k)

            local_str = self.local_directory + k.replace("$VERSION", "*")
            prog = re.compile(fname_v.replace("*", "(.*)"))
            # version_index = k.rstrip("$VERSION").count("*")

            # newest = None
            # newest_date = None

            local_matches = glob.glob(local_str)
            local_matches = [f.split("/")[-1] for f in local_matches]

            ftp.cwd(self.server_path + this_dir)
            remote_matches = ftp.nlst()

            remote_matches = [f for f in remote_matches if prog.match(f)]
            for f in remote_matches:
                # if check_version:
                #     this_version = prog.match(f).groups()[version_index]
                #     for f2 in remote_matches:
                local_test_file = self.local_directory + this_dir + f
                if os.path.exists(local_test_file) and (not force_download):
                    if self.verbose:
                        print("\tOK: " + local_test_file)
                    try:
                        local_matches.remove(f)
                    except ValueError:
                        pass
                else:
                    # copy remote-local
                    try:
                        if not dry_run:
                            parent_dirs = os.path.dirname(local_test_file)
                            if not os.path.exists(parent_dirs):
                                os.makedirs(parent_dirs)
                            ftp.retrbinary(
                                "RETR " + f, open(local_test_file, "wb").write
                            )
                        if not self.silent:
                            print("\tDL: " + local_test_file)
                        try:
                            local_matches.remove(f)
                        except ValueError:
                            pass
                    except IOError as e:
                        print(e)

            # Remove all files for which a remote match was not found
            for f in local_matches:
                try:
                    if not dry_run:
                        os.remove(self.local_directory + this_dir + f)
                    if not self.silent:
                        print("\tRM: " + self.local_directory + this_dir + f)
                except IOError as e:
                    print(e)

        ftp.quit()


if __name__ == "__main__":
    pass
