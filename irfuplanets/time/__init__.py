import spiceypy

from irfuplanets.spice.update import check_update_lsk_kernel
from irfuplanets.time.time import (
    CelsiusTime,
    Orbit,
    OrbitDict,
    celsiustime_to_spiceet,
    datetime64_to_spiceet,
    doy2004,
    mplnum,
    mplnum_to_spiceet,
    now,
    spiceet,
    spiceet_to_celsiustime,
    spiceet_to_datetime64,
    spiceet_to_mplnum,
    spiceet_to_utcstr,
    time_convert,
    utcstr,
    utcstr_to_spiceet,
)
from irfuplanets.time.time_axes import (
    SpiceetFormatter,
    SpiceetLocator,
    setup_time_axis,
)

__author__ = "David Andrews"
__copyright__ = "Copyright 2023, David Andrews"
__license__ = "MIT"
__email__ = "david.andrews@irfu.se"

__all__ = [
    "CelsiusTime",
    "Orbit",
    "OrbitDict",
    "doy2004",
    "mplnum",
    "now",
    "spiceet",
    "time_convert",
    "utcstr",
    "SpiceetFormatter",
    "SpiceetLocator",
    "setup_time_axis",
    "spiceet_to_utcstr",
    "utcstr_to_spiceet",
    "mplnum_to_spiceet",
    "spiceet_to_mplnum",
    "celsiustime_to_spiceet",
    "spiceet_to_celsiustime",
    "datetime64_to_spiceet",
    "spiceet_to_datetime64",
]

# Load an initial leap seconds kernel, updating it if needed
# Idea being, if later furnsh/kclear are called, you're on your own
# to make sure that a LSK file is present.  Hard not to do so, in SPICE.
spiceypy.furnsh(check_update_lsk_kernel())
