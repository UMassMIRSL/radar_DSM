# radar_DSM
generate Digital Surface Models from Capella and TerraSAR-X/TanDEM-X SAR imagery
![DSM of Savannah, GA, USA](https://github.com/UMassMIRSL/radar_DSM/blob/main/savannah_example_DSM.png)

# USAGE
radar_DSM is a Python package for generating Digital Surface Models (DSMs) from Synthetic Aperture Radar (SAR) data using stereo radargrammetry. It has been tested using Rocky Linux 9/RHEL9 on the following data formats:

Capella Space:
        Geocoded Ellipsoid Corrected:
                -Stripmap
                -Spotlight
                -Sliding Spotlight

TerraSAR-X/TanDEM-X:
        Geocoded Ellipsoid Corrected
                -Stripmap

The methodology of the package is described in: S. Beninati and S. Frasier (2025 submitted), "A radargrammetry algorithm for high-resolution SAR satellite constellations."

radargrammetry.py is the main file and contains the main methods of the code: radargrammetry() and radargrammetry_full()

# EXAMPLE
An example code using SAR imagery from Capella Space's Open Data Repository is located at: https://github.com/UMassMIRSL/radar_DSM/blob/main/radar_DSM_example_capella.py
The SAR imagery must be downloaded from the Open Data Repository as described in the example file

# INSTALLATION
The.py files should be downloaded into a directory named radar_DSM. This directory should be added to your PYTHONPATH variable.
This package has been tested using Conda and Python 3.12.8. Dependencies from the Conda environment are located in requirements.txt


# CREDIT
sgm5.py modified from:
DA. Beaupre, “Semi-global matching,” github.com. [https://github.com/beaupreda/semi-global-matching](https://github.com/beaupreda/semi-global-matching) (accessed Jul. 10, 2023).

This project was supported by the Office of Naval Research Code 322 under grant N00014-2112550, National Oceanographic Partnership Program (NOPP): Hurricane Coastal Impacts.

# CITATION
S. Beninati and S. Frasier, "radar_DSM," github.com. https://github.com/UMassMIRSL/radar_DSM, (Accessed: date accessed). doi: https://doi.org/10.5281/zenodo.15242906
