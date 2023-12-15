"""Routines to access and plot MAVEN L2 data.
"""

import os
from glob import glob

import irfuplanets
from irfuplanets.maven.maven_sc import (
    check_spice_furnsh,
    describe_loaded_kernels,
    iau_mars_position,
    iau_pgr_alt_lat_lon_position,
    iau_r_lat_lon_position,
    load_kernels,
    modpos,
    mso_position,
    mso_r_lat_lon_position,
    position,
    read_maven_orbits,
    sub_solar_latitude,
    sub_solar_longitude,
    unload_kernels,
)
from irfuplanets.maven.sdc_interface import HTTP_Manager
from irfuplanets.time import OrbitDict

__author__ = "David Andrews"
__copyright__ = "Copyright 2023, David Andrews"
__license__ = "MIT"
__email__ = "david.andrews@irfu.se"

DIRECTORY = irfuplanets.config["maven"]["kernel_directory"] + "spk/"

# Load Kernels
load_kernels()

# Read orbits
orbits = OrbitDict()
for f in sorted(glob(DIRECTORY + "maven_orb_rec_*.orb")):
    tmp = read_maven_orbits(f)

    for k, v in tmp.items():
        if k in orbits:
            raise IOError(
                "Duplicate information contained in %s: Orbit %d repeated?" % (f, k)
            )
        orbits[k] = v

if not orbits:
    raise IOError("No reconstructed orbits found?")

# Do these last:
# tmp = read_maven_orbits(DIRECTORY + "maven_orb_rec.orb")
# for k, v in tmp.items():
#     if k in orbits:
#         # raise IOError('Duplicate information
#               contained in %s: Orbit %d repeated?' % (f, k))
#         print(
#             "Duplicate information contained in %s: Orbit %d repeated?"
#             % (f, k)
#         )

#     orbits[k] = v

print("Read information for %d orbits" % len(orbits))

stored_data = {}

if os.getenv("MAVENPFP_USER_PASS") is None:  # Public access
    print("Setting up SDC access (public)")
    raise NotImplementedError(
        "URL broken, change to (??): "
        "'http://sprg.ssl.berkeley.edu/data/maven/data/sci/'"
    )
    maven_http_manager = HTTP_Manager(
        "http://lasp.colorado.edu/maven/sdc/public/data/sci/",
        "",
        "",
        irfuplanets.config["maven"]["data_directory"],
        verbose=False,
    )
else:
    print("Setting up Berkeley access (private)")
    maven_http_manager = HTTP_Manager(
        "http://sprg.ssl.berkeley.edu/data/maven/data/sci/",
        os.getenv("MAVENPFP_USER_PASS").split(":")[0],
        os.getenv("MAVENPFP_USER_PASS").split(":")[1],
        irfuplanets.config["maven"]["data_directory"],
        verbose=False,
    )

print("MAVEN setup complete\n----")

__all__ = [
    "check_spice_furnsh",
    "describe_loaded_kernels",
    "iau_mars_position",
    "iau_pgr_alt_lat_lon_position",
    "iau_r_lat_lon_position",
    "load_kernels",
    "modpos",
    "mso_position",
    "mso_r_lat_lon_position",
    "position",
    "read_maven_orbits",
    "sub_solar_latitude",
    "sub_solar_longitude",
    "unload_kernels",
    "maven_http_manager",
    "stored_data",
]
