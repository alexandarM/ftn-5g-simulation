"""
config.py
=========

Central location for all constants used across the project.

Previously these values were scattered across different notebook cells
(some even duplicated). Grouping them here by category makes maintenance
easier – changing e.g. the scene frequency or antenna datasheet parameters
only needs to happen in one place.
"""

import numpy as np

# -----------------------------------------------------------------------
# 1. SCENE / CARRIER FREQUENCY
# -----------------------------------------------------------------------
SCENE_FREQUENCY_HZ = 3.5e9  # 3.5 GHz (NR band n78)
SPEED_OF_LIGHT_M_S = 299_792_458.0

# -----------------------------------------------------------------------
# 2. ANTENNA DATASHEET SPECIFICATIONS (Ericsson AIR 3278)
# -----------------------------------------------------------------------
# These values define the antenna pattern shape (see antenna_pattern.py)
# and are also the TARGET values that the measurement functions
# (antenna_measurements.py) should confirm after simulation.
V_BW_DATASHEET = 8.5      # Vertical -3 dB beamwidth [degrees]
H_BW_DATASHEET = 65.0     # Horizontal -3 dB beamwidth [degrees]
SLL_DATASHEET = 13.0      # Side-lobe level [dB]
FBR_DATASHEET = 25.0      # Front-to-back ratio [dB]
PHI_OFFSET_DEG = 0.0     # Fine horizontal asymmetry correction

PHI_OFFSET_RAD = np.deg2rad(PHI_OFFSET_DEG)

# Target peak gain derived from link budget:
# EIRP = 70 dBm, P_tx per port = 53 dBm  =>  G_peak = 70 - 53 = 17 dBi
TARGET_PEAK_GAIN_DBI = 17.0

# -----------------------------------------------------------------------
# 3. ORIENTATION CALIBRATION (compass -> Sionna yaw convention)
# -----------------------------------------------------------------------
# Sionna uses mathematical angle convention (0° = +X axis, CCW positive),
# while antenna azimuths are usually given in compass convention
# (0° = north, clockwise). YAW_OFFSET_DEG provides additional fine-tuning
# if needed when comparing with field measurements or datasheet values.
YAW_OFFSET_DEG = 0.0

# -----------------------------------------------------------------------
# 4. DEFAULT TRANSMITTER POWER
# -----------------------------------------------------------------------
TX_POWER_DBM_DEFAULT = 53.0

# -----------------------------------------------------------------------
# 5. COLOUR PALETTE FOR PLOTS (consistent across figures)
# -----------------------------------------------------------------------
COLORS = {
    "blue": "#2E86C1",
    "green": "#28B463",
    "red": "#E74C3C",
    "orange": "#E67E22",
    "purple": "#8E44AD",
    "teal": "#1ABC9C",
    "gray": "#95A5A6",
    "dark": "#2C3E50",
}

# -----------------------------------------------------------------------
# 6. FILE PATHS (Google Colab environment)
# -----------------------------------------------------------------------
MESHES_ZIP_PATH = "meshes.zip"
MESHES_DIR = "meshes"
SCENE_XML_PATH = "/content/meshes/kampus_test.xml"
PLY_FOLDER = "/content/meshes/meshes/"
FIGURES_DIR = "figures"

# Google Drive ID for campus mesh files (used with `gdown`)
CAMPUS_MESHES_GDRIVE_ID = "16oJFwMXwbNxxNwj7y_gZZVvNncxA6Yvy"

# -----------------------------------------------------------------------
# 7. BOUNDARIES FOR RANDOM USER (RX) GENERATION
# -----------------------------------------------------------------------
RX_AREA_X_MIN, RX_AREA_X_MAX = -330, 260
RX_AREA_Y_MIN, RX_AREA_Y_MAX = -330, 260
RX_HEIGHT_M = 1.5  # Typical mobile device height above ground

# -----------------------------------------------------------------------
# 8. GEOREFERENCING (used by geo_utils.py to overlay coverage on a real
#    satellite/OSM background)
# -----------------------------------------------------------------------
CAMPUS_REF_LAT = 45.2461012
CAMPUS_REF_LON = 19.8516968
CAMPUS_REF_ROTATION_DEG = 0.0  # set > 0 (clockwise) if the BLOSM import was rotated relative to true north
