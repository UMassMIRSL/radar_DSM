"""
This example code uses data from the Capella Open Data repository to generate a DSM of the Savannah, GA, USA area. To run the example, Download the GeoTIFF image and the Extended Metadata JSON for both images from the Capella Open Data Repository:
CAPELLA_C09_SS_GEC_HH_20241004005327_20241004005341: https://radiantearth.github.io/stac-browser/#/external/capella-open-data.s3.amazonaws.com/stac/capella-open-data-by-datetime/capella-open-data-2024/capella-open-data-2024-10/capella-open-data-2024-10-04/CAPELLA_C09_SS_GEC_HH_20241004005327_20241004005341/CAPELLA_C09_SS_GEC_HH_20241004005327_20241004005341.json?.language=en&.asset=asset-metadata


CAPELLA_C10_SS_GEC_HH_20241002021116_20241002021130: https://radiantearth.github.io/stac-browser/#/external/capella-open-data.s3.amazonaws.com/stac/capella-open-data-by-datetime/capella-open-data-2024/capella-open-data-2024-10/capella-open-data-2024-10-02/CAPELLA_C10_SS_GEC_HH_20241002021116_20241002021130/CAPELLA_C10_SS_GEC_HH_20241002021116_20241002021130.json?.language=en&.asset=asset-metadata

Put the files into the same directory as this code, and then execute using the python command.
NOTE 1: The example as-is uses 16 CPU cores. The number of cores can be changed to fit the computer you are using.
NOTE 2: The example as-is is run at full resolution, and will use a large amount of memory and may take a long time. Time and memory use can be reduced significantly by setting the nl_x and nl_y keywords to values greater than 1 to enable multilooking, at the cost of vertical resolution.

MIT License

Copyright (c) 2025 UMass MIRSL

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


"""
from radar_DSM import radargrammetry as rg
rg.radargrammetry_full('CAPELLA_C09_SS_GEC_HH_20241004005327_20241004005341.tif', 'CAPELLA_C10_SS_GEC_HH_20241002021116_20241002021130.tif', 'Capella', 500., 16, 'savannah_2024_open_dem.tif', min_disp = -16, max_disp = 40, offset = 7.333, fill = True)  

