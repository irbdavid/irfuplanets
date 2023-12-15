"""Python routines for MEX/MARSIS analsyis (and some MEX/ASPERA)"""

import irfuplanets

# mex position stuff
from irfuplanets.mex.mex_sc import (
    MEXException,
    check_create_file,
    check_spice_furnsh,
    iau_mars_position,
    iau_pgr_alt_lat_lon_position,
    iau_r_lat_lon_position,
    load_kernels,
    mex_mission_phase,
    mso_position,
    mso_r_lat_lon_position,
    plot_mex_orbits_bar,
    position,
    read_all_mex_orbits,
    read_mex_orbits,
    solar_longitude,
    sub_solar_latitude,
    sub_solar_longitude,
    unload_kernels,
)
from irfuplanets.mex.orbit_plots import plot_bs, plot_mpb, plot_planet

__author__ = "David Andrews"
__copyright__ = "Copyright 2023, David Andrews"
__license__ = "MIT"
__email__ = "david.andrews@irfu.se"

data_directory = irfuplanets.config["mex"]["data_directory"]

datastore = {}

if "orbits" not in locals():
    orbits = read_all_mex_orbits()

__all__ = [
    "plot_bs",
    "plot_mpb",
    "plot_planet",
    "MEXException",
    "check_create_file",
    "check_spice_furnsh",
    "iau_mars_position",
    "iau_pgr_alt_lat_lon_position",
    "iau_r_lat_lon_position",
    "load_kernels",
    "mex_mission_phase",
    "mso_position",
    "mso_r_lat_lon_position",
    "plot_mex_orbits_bar",
    "position",
    "read_all_mex_orbits",
    "read_mex_orbits",
    "solar_longitude",
    "sub_solar_latitude",
    "sub_solar_longitude",
    "unload_kernels",
]
