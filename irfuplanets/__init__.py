"""Configuration stuff for irfuplanets"""

import configparser
import pathlib

__author__ = "David Andrews"
__copyright__ = "Copyright 2023, David Andrews"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "david.andrews@irfu.se"

path_to_config = pathlib.Path.home() / "irfuplanets.cfg"

# Long path made a mess :(
default_config_str = (
    """
[irfuplanets]
data_directory = ~/data/

[mex]
data_directory = ~/data/mex/

[maven]
data_directory = ~/data/maven/
kernel_directory = ~/data/maven/spg/data/misc/"""
    "spice/naif/MAVEN/kernels/"
)

default_config_str = default_config_str.replace("~", str(pathlib.Path.home()))


def _load_config(path):
    p = pathlib.Path(path)

    # initial setup
    default_config = configparser.ConfigParser()
    default_config.read_string(default_config_str)

    if p.exists():
        with open(str(p), "r") as f:
            default_config.read_file(f)
        print(f"Read {str(p)}")

    # Write back, in case of any new params:
    with open(str(p), "w") as f:
        default_config.write(f)

    return default_config


config = _load_config(path_to_config)
