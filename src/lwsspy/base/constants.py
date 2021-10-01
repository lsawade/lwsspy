from numpy import pi
import os

# ----------- GEO -------------------------------------------------------------
# mean earth radius in meter as defined by the International Union of
# Geodesy and Geophysics. Used for the spherical kd-tree and other things.
EARTH_RADIUS_M = 6371009.0
EARTH_RADIUS_KM = 6371009.0/1000.0
EARTH_CIRCUM_M = 2*pi*EARTH_RADIUS_KM
EARTH_CIRCUM_KM = 2*pi*EARTH_RADIUS_KM
DEG2M = EARTH_CIRCUM_M/360.0
DEG2KM = EARTH_CIRCUM_KM/360.0
M2DEG = 1.0/DEG2M
KM2DEG = 1.0/DEG2KM

# ----------- URLs ------------------------------------------------------------

# IRIS Earth Model collaboration
EMC_DATABASE = "https://ds.iris.edu/files/products/emc/emc-files/"


# ----------- Random ----------------------------------------------------------

abc = 'abcdefghijklmnopqrstuvwxyz'

# ----------- Directories -----------------------------------------------------

DOCFIGURES: str = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__))))),
    'docs', 'source', 'chapters', 'figures')
DOCFIGURESCRIPTDATA: str = os.path.join(DOCFIGURES, 'scripts', 'data')

DOWNLOAD_CACHE: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'download_cache')

FONTDIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'plot',
    'fonts')

CONSTANT_DATA: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'constant_data')

GCMT_DATA: str = os.path.join(CONSTANT_DATA, 'gcmt')
