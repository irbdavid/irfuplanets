from irfuplanets.mex.ais.ais_code import (
    AISEmpiricalCalibration,
    AISFileManager,
    AISTimeSeries,
    DigitizationDB,
    Ionogram,
    IonogramDigitization,
    ais_delays,
    ais_max_delay,
    ais_min_delay,
    ais_number_of_delays,
    ais_spacing_seconds,
    ais_vmax,
    ais_vmin,
    calibrator,
    compute_all_digitizations,
    fp_to_ne,
    get_ais_data,
    get_ais_index,
    laminated_delays,
    modb_to_td,
    ne_to_fp,
    plot_ais_coverage_bar,
    produce_ne_b_file,
    read_ais,
    read_ais_file,
    speed_of_light_kms,
    td_to_modb,
    write_yearly_ne_b_files,
)
from irfuplanets.mex.ais.aisreview import AISReview
from irfuplanets.mex.ais.aistool import AISTool

# import aisreview
# import aisworkflow

__author__ = "David Andrews"
__copyright__ = "Copyright 2015, David Andrews"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "david.andrews@irfu.se"


# Create a default instance
file_manager = AISFileManager(remote="NONE", verbose=True, brain=False)


__all__ = [
    "AISEmpiricalCalibration",
    "AISFileManager",
    "AISTimeSeries",
    "DigitizationDB",
    "Ionogram",
    "IonogramDigitization",
    "calibrator",
    "compute_all_digitizations",
    "fp_to_ne",
    "get_ais_data",
    "get_ais_index",
    "laminated_delays",
    "modb_to_td",
    "ne_to_fp",
    "plot_ais_coverage_bar",
    "produce_ne_b_file",
    "read_ais",
    "read_ais_file",
    "td_to_modb",
    "write_yearly_ne_b_files",
    "file_manager",
    "AISReview",
    "AISTool",
    "ais_spacing_seconds",
    "ais_number_of_delays",
    "ais_delays",
    "ais_max_delay",
    "ais_min_delay",
    "ais_vmin",
    "ais_vmax",
    "speed_of_light_kms",
]
