"""
File Name: radargrmmetry.py
Author: Steven Beninati

Description: Module for the generation of DEMs from SAR data using stereo radargrammetry. 
Currently supported data types: 
- GEC imagery from Capella Space
- GRD imagery from TerraSAR-X/TanDEM-X



"""
from radar_DSM import get_md, imgpro, sgm5
import os
import numpy as np
import numpy.ma as ma
import cv2
import csv
from scipy import signal, ndimage, optimize
from scipy.interpolate import griddata
from osgeo import gdal, osr
import matplotlib as mpl
from matplotlib import pyplot as plt
import multiprocessing as mp
import time as t
import sys
import shutil
import glob

def radargrammetry_full(img_hi_file, img_lo_file, fmt, dim, num_cpu, out_file, src_dir = ".", dem_file = None , out_dir = '.', dB = False, img_med = 5, min_disp = 0, max_disp = 64, P1 = 50, P2 = 350, offset = None, fill = True, msl = None, compare = False, start_tile = 0, out_dem_file = None, cleanup = True, min_ht = None, max_ht = None, nl_x = 1, nl_y = 1, min_cts = 20):
    """
    perform radargrmmetry on tiles of size dim for a full image
    REQUIRED INPUTS:
         img_hi_file:   String: high incidence angle image file name (directory for TDX)
         img_lo_file:   String: low incidence angle image file name (directory for TDX)
         fmt:           String: import sar image format. currently supoported: "Capella", "TDX"
         dim:           Float:  output tile size in same geographic deimensions as img_hi_file. If the fill keyword is False, then this is also the size of the input region
         n_cpu:         Int:    number of CPUs available for parallel processing. Should be speicfied as a multiple of 4
         out_file:      String: name of output DEM file.
    OPTIONAL INPUTS:
         src_dir:       (Optional) String: source directory. Default: current directory
         dem_file:      (Optional) String: dem file name. Default: None
         out_dir:       (Optional) String: output directory. Default: current directory
         dB:            (Optional) Boolean: db scaling option. If this is set to true, the images will be dB scaled before semi-global matching. Default: False
         img_med:       (Optional) Int: pixel size of median filter to be applied to SAR images before SGM algorithm is applied. Default: 9
         min_disp:      (Optional)Integer specifying the minimum disparity value to check for in the SGM algorithm. Default: 0. For heights below thegeocoding projection plane,
                    min_disp should be a negative value.
         max_disp:      (Optional) Int: maximum disparity value to check for in the SGM algorithm. Default: 64.
         P1:            (Optional) Float: penalty value to be applied to neighbor pixels which differ by a disparity of exactly 1 in the SGM algorithm. Default: 50.
         P1:            (Optional) Float: penalty value to be applied to neighbor pixels which differ by a disparity of >1 in the SGM algorithm. Should be greatre than P1. Default: 350.
         offset:        (Optional) Float: offset of the SAR imagery from 0 height in the output datum. Default: None. By default, the program uses the projection
                        height of the high incidence angle image. If a DEM is supplied, a height conversion will be attempted from the SAR imagery coordinate system to the 
                        DEM coordinate system. It should be noted this height conversion uses the GDAL TransformPoint function, which is only somewhat successful for 
                        3D datum conversion. If the DEM is in a different vertical CRS than the SAR imagery, it is recommended to use another tool 
                        (such as NOAA's Online VDatum tool: https://vdatum.noaa.gov/vdatumweb/) to find the correct height for the output datum and manually set the 
                        offset keyword to that value.
         msl:           (Optional) Float: "mean sea level" water level height in the reference DEM. Pixels below this value are flagged as no data. Default: None
         compare:       (Optional) Boolean: setting this value to True with a defined dem_file will compare the output DEM to dem_file and display the comparision results. Default: False
         start_tile;    (Optional) Int: tile number to start generating tiles from. Generally used if the DEM generatino process is interrupted to pick up where the program left off. Default: 0
         out_dem_file:  (Optional) String: name of intermediate DEM tile files. Setting this value to None will create a default name of <img_hi_file>_<fmt>_<number>_dem.tif . Default: None.
         cleanup:       (Optional) Boolean: setting this value to True will clean up intermediate files used in DEM creation. Default: True
         min_ht:        (Optional) Float: minimum height to used in comparison with dem_file. Default: None 
         max_ht:        (Optional) Float: maximum height to used in comparison with dem_file. Default: None 
         no_data_out:   (Optional) Float: output no data value. Default: np.nan
         nl_x:          (Optional) Int: number of looks to multilook in x direction. Default: 1
         nl_y:          (Optional) Int: number of looks to multilook in y direction. Default: 1
         min_cts:       (Optional) Int: minimum number of counts in 2D histogram bin to display that bin. Default: 20
    """
    dim = np.float64(dim)
    no_data_out = np.nan
    if fmt.casefold() == "Capella".casefold(): 
        md_hi = get_md.capella(img_hi_file, src_dir = src_dir)
        md_lo = get_md.capella(img_lo_file, src_dir = src_dir)
        if out_dem_file is None:
            ext = img_hi_file.find('.tif')
            out_dem_file = img_hi_file[0:ext]+'_Capella'

    elif fmt.casefold() == "TDX".casefold(): 
        md_hi = get_md.tdx(img_hi_file, src_dir = src_dir)
        md_lo = get_md.tdx(img_lo_file, src_dir = src_dir)
        if out_dem_file is None:
            out_dem_file = img_hi_file+'_TDX'
    else:
        print("Error: unsupported SAR image format. Please select from the folllowing list: 'Capella', 'TDX'")
    x_max = np.min(np.asarray([md_hi.x0 + np.abs(md_hi.dx) * md_hi.nx, md_lo.x0 + np.abs(md_lo.dx) * md_lo.nx]))
    x_min = np.max(np.asarray([md_hi.x0, md_lo.x0]))
    y_max = np.min(np.asarray([md_hi.y0, md_lo.y0]))
    y_min = np.max(np.asarray([md_hi.y0 - np.abs(md_hi.dy) * md_hi.ny, md_lo.y0 - np.abs(md_lo.dy) * md_lo.ny]))
    x_ctrs = np.arange(x_min+dim/2, x_max-dim/2, dim)
    y_ctrs = np.arange(y_min+dim/2, y_max-dim/2, dim)
    x_grid = np.repeat(x_ctrs.reshape(1,x_ctrs.shape[0]), y_ctrs.shape[0], axis = 0)
    y_grid = np.repeat(y_ctrs.reshape((y_ctrs.shape[0],1)), x_ctrs.shape[0], axis = 1)
    ctrs = np.dstack((x_grid, y_grid))
    dawn = t.time()
    ctr_list = list([(ctrs[i,j,0], ctrs[i,j,1]) for i in range(ctrs.shape[0]) for j in range(ctrs.shape[1])])
    c = start_tile
    while c  < len(ctr_list):
        i = 0
        processes = []
        while i < num_cpu/4:
            kwargs = {'src_dir': src_dir, 'dem_file':dem_file, 'out_dir':out_dir, 'dB':dB, 'img_med': img_med, 'min_disp':min_disp, 'max_disp':max_disp, 'P1':P1, 'P2':P2, 'offset':offset, 'fill':fill, 'msl':msl, 'out_dem_file':out_dem_file+f'_tile_{c}.tif', 'compare':compare, 'cleanup':cleanup, 'min_ht':min_ht, 'max_ht':max_ht, 'no_data_out':no_data_out, 'min_cts':min_cts, 'nl_x':nl_x, 'nl_y':nl_y}
            sys.stdout.flush()
            processes.append(mp.Process(target = radargrammetry, args=(img_hi_file, img_lo_file, fmt, ctr_list[c], dim), kwargs=kwargs))
            processes[-1].start()
            i += 1
            c += 1
            print(i)
            print(c)
            if c >= len(ctr_list):
                break
        [p.join() for p in processes] 
    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))
    print('merging tiles...')
    tile_list = glob.glob(out_dem_file+'_tile_*.tif')
    vrt_list = glob.glob(out_dem_file+'_tile_*.vrt')
    warp_opt = gdal.WarpOptions(srcNodata = no_data_out, dstNodata = no_data_out)
    gdal.Warp(out_file, tile_list, options = warp_opt)
    if cleanup:
        for i in np.arange(len(tile_list)):
            if os.path.isfile(tile_list[i]):
                os.remove(tile_list[i])
        for i in np.arange(len(vrt_list)):
            if os.path.isfile(vrt_list[i]):
                os.remove(vrt_list[i])
    
def radargrammetry(img_lo_file, img_hi_file, fmt, ctr, dim, src_dir = ".", dem_file = None , out_dir = '.', dB = False, img_med = 5, min_disp = 0, max_disp = 64, P1 = 50, P2 = 350, offset = None, fill = False, msl = None, out_dem_file = None, compare = False, cleanup = False, min_ht = None, max_ht = None, no_data_out = np.nan, nl_x = 1, nl_y = 1, min_cts = 20):
    """
    REQUIRED INPUTS:
         img_hi_file:   String: high incidence angle image file name (directory for TDX)
         img_lo_file:   String: low incidence angle image file name (directory for TDX)
         fmt:           String: import sar image format. currently supoported: "Capella", "TDX"
         ctr:           2-element tuple of Floats: center point of area of interest in the coordinate system of the SAR imagery
         dim:           Float: size of the output tile in geographic dimension. If the fill keyword is False, then this is also the size of the input region
    OPTIONAL INPUTS:
         src_dir:       (Optional) String: source directory. Default: current directory
         dem_file:      (Optional) String: reference dem file name. Default: None
         out_dir:       (Optional) String: the output directory. Default: current directory
         dB:            (Optional) Boolean: db scaling option. If this is set to true, the images will be dB scaled before semi-global matching. Default: False
         img_med:       (Optional) Int: pixel size of median filter to be applied to SAR images before SGM algorithm is applied. Default: 9
         min_disp:      (Optional) Int: specifying the minimum disparity value to check for in the SGM algorithm. Default: 0. For heights below thegeocoding projection plane,
                        min_disp should be a negative value.
         max_disp:      (Optional) Int: maximum disparity value to check for in the SGM algorithm. Default: 64.
         P1:            (Optional) Float: penalty value to be applied to neighbor pixels which differ by a disparity of exactly 1 in the SGM algorithm. Default: 50.
         P1:            (Optional) Float: penalty value to be applied to neighbor pixels which differ by a disparity of >1 in the SGM algorithm. Should be greatre than P1. Default: 350.
         offset:        (Optional) Float: offset of the SAR imagery from 0 height in the output datum. Default: None. By default, the program uses the projection
                           height of the high incidence angle image. If a DEM is supplied, a height conversion will be attempted from the SAR imagery coordinate system to the 
                           DEM coordinate system. It should be noted this height conversion uses the GDAL TransformPoint function, which is only somewhat successful for 
                           3D datum conversion. If the DEM is in a different vertical CRS than the SAR imagery, it is recommended to use another tool 
                           (such as NOAA's Online VDatum tool: https://vdatum.noaa.gov/vdatumweb/) to find the correct height for the output datum and manually set the 
                           offset keyword to that value.
         fill:          (Optional) Boolean: True: fill the output tile of size dim. If this keyword is False, then the extracted area will be of size dim and will
                           not fill the output tile. setting fill will process a region of 1.75*dim and clip the output to size dim. Default: False
         msl:           (Optional) Float: "mean sea level" water level height in the reference DEM. Pixels below this value are flagged as no data. Default: None
         out_dem_file:  (Optional) String: name of output DEM file. Setting this value to None will create a default name of <img_hi_file>_<fmt>_dem.tif . Default: None.
         compare:       (Optional) Boolean: setting this value to True with a defined dem_file will compare the output DEM to dem_file and display the comparision results. Default: False
         cleanup:       (Optional) Boolean: setting this value to True will clean up intermediate files used in DEM creation. Default: False
         min_ht:        (Optional) Float: minimum height to used in comparison with dem_file. Default: None 
         max_ht:        (Optional) Float: maximum height to used in comparison with dem_file. Default: None 
         no_data_out:   (Optional) Float: output no data value. Default: np.nan
         nl_x:          (Optional) Int: number of looks to multilook in x direction. Default: 1
         nl_y:          (Optional) Int: number of looks to multilook in y direction. Default: 1
         min_cts:       (Optional) Int: minimum number of counts in 2D histogram bin to display that bin. Default: 20
    OUTPUT:
         out_dem_file:  String: Final DEM output file name
    """
    if fmt.casefold() == "Capella".casefold(): 
        md_hi = get_md.capella(img_hi_file, src_dir = src_dir)
        md_lo = get_md.capella(img_lo_file, src_dir = src_dir)
        if out_dem_file is None:
            ext_hi = img_hi_file.find('.tif')
            out_dem_file = img_hi_file[0:ext_hi]+'_Capella_dem.tif'
        
    elif fmt.casefold() == "TDX".casefold(): 
        md_hi = get_md.tdx(img_hi_file, src_dir = src_dir)
        md_lo = get_md.tdx(img_lo_file, src_dir = src_dir)
        if out_dem_file is None:
            out_dem_file = img_hi_file+'_TDX_dem.tif'
    else:
        print("Error: unsupported SAR image format. Please select from the folllowing list: 'Capella', 'TDX'")
    if md_hi.inc < md_lo.inc:
        print("ERROR: file 'img_hi_file' has a lower incidence angle than file 'img_lo_file'. Please exchange order of arguments")
        return ''
    bound_box = None
    if fill: #if the fill keyword is set, the DEM should fill the entire window. define a bounding box as the output and expand the subplot area to guarantee the final window is full
        bound_box = np.asarray([ctr[0] - dim/2, ctr[1] -dim/2, ctr[0] + dim/2, ctr[1] + dim/2])
        dim *= 1.75
    ext_out = out_dem_file.find('.tif')
    sub_hi_file = out_dem_file[0:ext_out] + '_sub_hi.tif'
    sub_lo_file = out_dem_file[0:ext_out] + '_sub_lo.tif'
    if dem_file is None:
        #no_data needs to specified because the TDX data does not have it in the metadata, so the no_data value which is set on initialization is lost in this function
        sub_files = imgpro.subplot([md_hi.img_file, md_lo.img_file], ctr=ctr, dim=dim, in_dirs = [md_hi.img_dir, md_lo.img_dir], out_dirs = out_dir, no_datas = [md_hi.no_data, md_lo.no_data], x_res = min([md_hi.dx,md_lo.dx]), y_res = min([abs(md_hi.dy),abs(md_lo.dy)]), out_file_list = [sub_hi_file, sub_lo_file])
        if sub_files[0] == '':
            print('no subplot made for region for one or more files. It is possible one of the files was not found or the requested area was out of the image extent for one or more files.')
            return ''
    else:
        #no_data needs to specified because the TDX data does not have it in the metadata, so the no_data value which is set on initialization is lost in this function
        sub_dem_file = out_dem_file[0:ext_out] + '_sub_dem.tif'
        md_dem = get_md.GDAL_ds(dem_file, src_dir = src_dir)
        sub_files = imgpro.subplot([md_hi.img_file, md_lo.img_file, dem_file], ctr=ctr, dim=dim, in_dirs = [md_hi.img_dir, md_lo.img_dir, src_dir], out_dirs = out_dir, no_datas = [md_hi.no_data, md_lo.no_data, md_dem.no_data], x_res = min([md_hi.dx,md_lo.dx]), y_res = min([abs(md_hi.dy), abs(md_lo.dy)]), out_file_list = [sub_hi_file, sub_lo_file, sub_dem_file])
        if sub_files[0] == '':
            print('no subplot made for region for one or more files. It is possible one of the files was not found or the requested area was out of the image extent for one or more files.')
            return ''
        sub_dem_file = sub_files[2]
    if dem_file is not None:
        ds_dem = gdal.Open(os.path.join(out_dir,sub_dem_file))
        sub_dem = cv2.imread(sub_dem_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        md_dem = get_md.GDAL_ds(os.path.join(out_dir, sub_dem_file))
        sub_dem = ds_dem.ReadAsArray()
    sub_hi_file = sub_files[0]
    sub_lo_file = sub_files[1]
    #replace geotransform data with warped geotransform data, preserve other information
    md_hi = md_hi.add_GDAL_ds(get_md.GDAL_ds(os.path.join(out_dir, sub_hi_file)))
    md_lo = md_lo.add_GDAL_ds(get_md.GDAL_ds(os.path.join(out_dir, sub_lo_file)))
    #read in matrices
    ds_hi = gdal.Open(os.path.join(out_dir,sub_hi_file))
    ds_lo = gdal.Open(os.path.join(out_dir,sub_lo_file))
    if ds_hi is None:
        print('Cannot open image' + os.path.join(out_dir, sub_hi_file))
        exit()
    if ds_lo is None:
        print('Cannot open image' + os.path.join(out_dir, sub_lo_file))
        exit()
    sub_hi = ds_hi.ReadAsArray()
    sub_lo = ds_lo.ReadAsArray()
    #radiometrically scale images
    if np.argwhere(sub_hi == md_hi.no_data).size > sub_hi.size:
        print('many no data indices found. Exiting...')
        if cleanup:
            if os.path.isfile(sub_hi_file):
                os.remove(sub_hi_file)
            if os.path.isfile(sub_lo_file):
                os.remove(sub_lo_file)
            if dem_file is not None:
                if os.path.isfile(sub_dem_file):
                    os.remove(sub_dem_file)
        return ''
    if np.argwhere(sub_lo == md_lo.no_data).size > sub_lo.size:
        print('many no data indices found. Exiting...')
        if cleanup:
            if os.path.isfile(sub_hi_file):
                os.remove(sub_hi_file)
            if os.path.isfile(sub_lo_file):
                os.remove(sub_lo_file)
            if dem_file is not None:
                if os.path.isfile(sub_dem_file):
                    os.remove(sub_dem_file)
        return ''
    #multilook images if specified
    if nl_x != 1 or nl_y != 1:
        #note Python/numpy is row major, so everything is y, x
        sub_hi, md_hi = imgpro.multilook(sub_hi, md_hi, nl_x, nl_y)

        sub_lo, md_lo = imgpro.multilook(sub_lo, md_lo, nl_x, nl_y)
        if dem_file is not None:
            sub_dem, md_dem = imgpro.multilook(sub_dem, md_dem, nl_x, nl_y)
        
    sub_hi = np.float64(sub_hi)*md_hi.rad_scl
    sub_lo = np.float64(sub_lo)*md_lo.rad_scl
    #db scale image if option is selected
    if dB:
        sub_hi = 10*np.log10(np.float64(sub_hi)) 
        sub_lo = 10*np.log10(np.float64(sub_lo))
    #median filter images to reduce speckle
    if img_med > 1:
        print(f"Applying median filter size {img_med}")
        sub_hi = signal.medfilt2d(sub_hi, kernel_size = img_med)
        sub_lo = signal.medfilt2d(sub_lo, kernel_size = img_med)
    rect_arrs = rectify_imgs(sub_hi, sub_lo, md_hi, md_lo)
    rect_hi = rect_arrs['rect_hi']
    rect_lo = rect_arrs['rect_lo']
    #scale images to uint16 data type based on statistics
    ext = out_dem_file.find('.tif')
    sgm_dir = out_dem_file[0:ext]+'_sgm'
    if not os.path.exists(sgm_dir):
        os.mkdir(sgm_dir)
    else:
        print(f"Folder '{sgm_dir}' already exists.")
    int_hi = imgpro.int16_scale(rect_hi)
    int_lo = imgpro.int16_scale(rect_lo)
    rect_hi_file = os.path.join(out_dir, sgm_dir, 'rect_hi.png')
    rect_lo_file = os.path.join(out_dir, sgm_dir, 'rect_lo.png')
    cv2.imwrite(rect_hi_file, int_hi)
    cv2.imwrite(rect_lo_file, int_lo)
    left_disp_file, right_disp_file = sgm5.sgm(rect_lo_file, rect_hi_file, max_disparity = max_disp, min_disparity = min_disp, P1 = P1, P2 = P2, sgm_dir = os.path.join(out_dir,sgm_dir))
    if offset is None:
        #if no offset value has been provided, use the ellipsoid height of the high incidence angle image. The output will then be in WGS84 ellipsoid height
        if dem_file is not None:
            #if a dem has been provided, transform the offset height to the SRS of the dem
            md_dem = get_md.GDAL_ds(dem_file, src_dir = src_dir)
            target_srs = osr.SpatialReference()
            target_srs.ImportFromEPSG(md_dem.srs)        # get spatial reference from EPSG id
            src_srs = osr.SpatialReference()
            src_srs.ImportFromEPSG(md_hi.srs)        # get spatial reference from EPSG id
            transform = osr.CoordinateTransformation(src_srs, target_srs)
            (dem_x, dem_y, dem_ht) = transform.TransformPoint(md_hi.x0+md_hi.nx/2*md_hi.dx, md_hi.y0-md_hi.ny/2*md_hi.dy, md_hi.ht)
            offset = -dem_ht
        else:
            offset = -md_hi.ht
    radg_dem, dz = disp2ht(left_disp_file, right_disp_file, rect_arrs['phi'], rect_arrs['rotation_angle'], md_hi, md_lo, min_disp, max_disp, no_data_out)
    if dem_file is not None and msl is not None: #if a dem has been provided, filter the heights below the provided sea level
        radg_dem[np.nonzero(sub_dem < msl)] = no_data_out
    out_dem_file = geocode(radg_dem, md_lo, rect_arrs['angle_lo'], no_data_out, offset = offset, bound_box = bound_box, out_file = out_dem_file)
    if compare:
        dem_compare(out_dem_file, dem_file, test_dir = out_dir, ref_dir = src_dir, img_file = md_hi.img_file, img_dir = md_hi.img_dir, dB = True, min_ht = min_ht, max_ht = max_ht, min_cts = min_cts, dz = dz)
    #clean up extra files left over from processing
    if cleanup:
        if os.path.isfile(sub_hi_file):
            os.remove(sub_hi_file)
        if os.path.isfile(sub_lo_file):
            os.remove(sub_lo_file)
        if dem_file is not None:
            if os.path.isfile(sub_dem_file):
                os.remove(sub_dem_file)
        shutil.rmtree(sgm_dir, ignore_errors=True)
    return out_dem_file    
        
def disp2ht(left_disp_file, right_disp_file, phi, rotation_angle, md_hi, md_lo, min_disp, max_disp, no_data):
    """
    convert two disparity maps to a height map based on image geometry. Utilizes left-right consistency check to remove suspect points
    REQUIRED INPUTS:
       left_disp_file:    String: file name of left disparty image. The image is expected to be a 16 bit UINT
       right_disp_file:   String: file name of right disparty image. The image is expected to be a 16 bit UINT
       phi:               Float: separation angle of the two images
       md_hi:             Metadata_SAR object: high incidence angle image metadata
       md_lo:             Metadata_SAR object: low incidence angle image metadata
       min_disp:          Int: minimum disparity value
       max_disp:          Int: maximum disparity value
       no_data:           Float: output no data value
    OUTPUT
       ht_img:            np.float32 numpy array: containing rectified radargrammetry DEM
       dz:                FLOAT: height resolution of DEM
    """
    thold = 5 #threshold of disparity difference for pixel to be labeled
    thold_scl = np.float32(thold)/np.float32(max_disp-min_disp)*65535
    left_disp = cv2.imread(left_disp_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
    right_disp = cv2.imread(right_disp_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
    no_data_locs = np.nonzero(np.abs(left_disp-right_disp) > thold_scl)
    consistency = np.abs(left_disp-right_disp)
    left_disp[no_data_locs] = np.nan
    right_disp[no_data_locs] = np.nan
    #interpolate nans on grid before rotating and geocoding
    valid = np.nonzero(~np.isnan(left_disp))
    valid_array = ~np.isnan(left_disp)
    points = (valid[0], valid[1])
    interp_pts = (np.arange(left_disp.shape[0]).repeat(left_disp.shape[1]), np.arange(np.asarray(left_disp.shape).prod())%left_disp.shape[1])
    left_disp = griddata(points, left_disp[valid], interp_pts, method = 'linear', fill_value = min_disp-1).reshape(left_disp.shape)
    #rotate rectified images back to ground coordinates
    rot_disp = ndimage.rotate(left_disp, rotation_angle, mode = 'constant', cval = min_disp-1, reshape = False, order = 1)
    valid_array_rot = ndimage.rotate(valid_array, rotation_angle, mode = 'constant', cval = 0, reshape = False, order = 0)
    rot_disp[np.nonzero(rot_disp < min_disp)] = no_data
    inc_hi_rad = md_hi.inc*np.pi/180
    inc_lo_rad = md_lo.inc*np.pi/180
    phi_rad = phi*np.pi/180
    dz = md_hi.dx/(np.sqrt((1/np.tan(inc_lo_rad))**2+(1/np.tan(inc_hi_rad))**2-2*1/np.tan(inc_lo_rad)*1/np.tan(inc_hi_rad)*np.cos(phi_rad)))
    print(f"dz: {dz}")
    ht_img = (rot_disp*float(max_disp-min_disp)/65535+float(min_disp))*np.abs(md_hi.dx)/(np.sqrt((1/np.tan(inc_lo_rad))**2+(1/np.tan(inc_hi_rad))**2-2*1/np.tan(inc_lo_rad)*1/np.tan(inc_hi_rad)*np.cos(phi_rad))) #rescale disparity image and then scale to height using range information
    ht_img[np.nonzero(valid_array_rot == 0)] = no_data
    return ht_img, dz

def geocode(dem, md, gamma, no_data, offset = 0., bound_box = None, out_file = 'geocoded_dem.tif', out_res = None):
    """
    correct for heights above the target plane and write a point cloud in the reference plane of the provided metadata to CSV
    REQUIRED INPUTS:
       dem:        np.float32 numpy array: dem to be geocoded
       md:         Metadata_SAR object: metadata for the high incidence angle image used to generate the DEM
       gamma:      Float: angle between the high incidence angle image and the East(x) ground axis
       no_data:    Float: no data value for input and output data 
    OPTIONAL INPUTS
       offset:     (Optional) Float: offset of projection plane and target spatial reference system. Defualt: 0.
       bound_box:  (Optional) List-like of floats in format [x0, y0, x1, y1]: Bounding box of area of interest for final output
       out_file:   (Optional) String: output DEM file name. Default: "geocoded_dem.tif" 
       out_res:    (Optional) Float: specifying output resolution, in the coordinates of md
    OUTPUTS:
       out_file:   String: file name of geocoded DEM
    """
    if out_res is None:
        out_res = 3*md.dx
    inc_rad = md.inc*np.pi/180
    gamma_rad = gamma*np.pi/180
    x_disp = 1/np.tan(inc_rad)*(dem)*np.cos(gamma_rad) #calculate x and y displacement based on height
    y_disp = 1/np.tan(inc_rad)*(dem)*np.sin(gamma_rad)
    x_disp[np.nonzero((dem == no_data) & ~np.isnan(x_disp))] =  0.  
    y_disp[np.nonzero((dem == no_data) & ~np.isnan(y_disp))] =  0. 
    #x and y positions for each point assuming that the target is located on the projection plane
    x_pos = np.repeat(np.arange(md.nx).reshape(1,md.nx), md.ny, axis = 0)*md.dx+md.x0
    y_pos = md.y0 - np.repeat(np.arange(md.ny).reshape((md.ny,1)), md.nx, axis = 1)*np.abs(md.dy)
    #recalulate position based on estimated height
    x_pos -= x_disp
    y_pos -= y_disp 
    dem_cloud = np.stack((x_pos.reshape(md.nx*md.ny), y_pos.reshape(md.nx*md.ny), dem.reshape(md.nx*md.ny) + np.float64(offset)))
    dem_cloud_filt = dem_cloud[:,(np.nonzero(~(dem_cloud[2,:] == no_data) & ~np.isnan(dem_cloud[2,:])))[0]]
    #write the point cloud into a csv
    ext = out_file.find('.tif')
    csv_file = out_file[0:ext]+'.csv'
    with open(csv_file, 'w', newline = '') as csv_out:
        pt_writer = csv.writer(csv_out, delimiter = ',')
        pt_writer.writerow(["Easting", "Northing", "Height"])
        for p in range(dem_cloud_filt.shape[1]):
            pt_writer.writerow([dem_cloud_filt[0,p], dem_cloud_filt[1,p], dem_cloud_filt[2,p]])
    #and now make the VRT file which goes with the point cloud
    vrt_file = csv_file[0:csv_file.find('.csv')] + '.vrt'
    layer = csv_file[0:csv_file.find('.csv')]
    with open(vrt_file, 'w', newline = '') as vrt:
        vrt.write('<OGRVRTDataSource>\n') 
        vrt.write('\t<OGRVRTLayer name="'+layer+'">\n') 
        vrt.write('\t\t<SrcDataSource>'+csv_file+'</SrcDataSource>\n')
        vrt.write('\t\t <GeometryType>wkbPoint</GeometryType>\n')
        vrt.write('\t\t<GeometryField encoding="PointFromColumns" x="Easting" y="Northing" z="Height"/>\n')
        vrt.write('\t</OGRVRTLayer>\n')
        vrt.write('</OGRVRTDataSource>\n') 
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(md.srs)        # get spatial reference from EPSG id
    #find extents of output tif
    if bound_box is not None:
        x_min = bound_box[0]
        x_max = bound_box[2]
        y_min = bound_box[1]
        y_max = bound_box[3]
    else:
        x_min = md.x0
        x_max = md.x0+md.nx*md.dx
        y_min = md.y0-md.ny*np.abs(md.dy)
        y_max = md.y0
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(md.srs)        # get spatial reference from EPSG id
    width = int((x_max-x_min)/out_res)
    height = int((y_max-y_min)/out_res)
    grid_opt = gdal.GridOptions(outputSRS = srs, outputBounds = (x_min, y_min, x_max, y_max), algorithm = f'average:radius1=3.0:radius2=3.0:nodata={no_data}', noData = no_data, width = width, height = height)
    gdal.Grid(out_file, vrt_file, options=grid_opt)
    if os.path.isfile(csv_file):
        os.remove(csv_file)
    else:
    # If it fails, inform the user.
        print("Error: %s file not found: " + csv_file)
    return(out_file)


def dem_compare(test_dem_file, ref_dem_file, test_dir='.', ref_dir='.', img_file = None, img_dir = '.', max_ht = None, min_ht = None, dB = False, min_cts = 20, max_cts =  None, ctr = None, dim = None, msl = None, dz = None, class_file = None, class_dir = '.', diff_min = -20., diff_max = 20., img_min  = None, img_max = None, bound_box = None): 
#compare radargrammetry to a reference DEM
    """
    REQUIRED INPUTS:
        test_dem_file:  String: file name of the test DEM
        ref_dem_file:   String: file name of the reference DEM
    OPTIONAL INPUTS:
        test_dir:       (Optional) String: directory containing the test DEM. Default: current directory
        ref_dir:        (Optional) String: directory containing the reference DEM. Default: current directory
        img:            (Optional) String: file name of the reference image
        img_dir:        (Optional) String: directory containing the reference image. Default: current directory
        max_ht:         (Optional) Float: maximum height to display in the scatter plot. optional. Omitting this value set the maximum based on the mean + 2*stddev of the DEM heights
        min_ht:         (Optional) Float: minimum height to display in the scatter plot. optional. Omitting this value set the minimum based on the mean - 2*stddev of the DEM heights
        min_cts:        (Optional) Int: minimum count limit for display in 2D histogram. Default: 20
        max_cts:        (Optional) Int: maximum count limit for display in 2D histogram. Default: None
        ctr:            (Optional) Tuple-like of Floats (x, y): center coordinates of the sub area of interest, in the DEM coordinate system. By default, the entire overlapping area of test_dem_file  and ref_dem_file is displayed
        dim:            (Optional) Float: dimension of the sub area of interest, in the DEM coordinate system. By default, the entire overlapping area of test_dem_file and ref_dem_file is displayed
        msl:            (Optional) Float: "mean sea level" water level height in the reference DEM. Pixels below this value are flagged as no data. Default: None
        dz:             (Optional) Float: vertical resolution of the test DEM. Used to size the cells in the 2D histogram. Default: None
        class_file:     (Optional) String: file name of classification file (e.g. land cover). Currently supported: Binary classification file (0 or 1). Default: None. If a value is provided, scatter plots will be generated separately for each class
        class_dir:      (Optional) String: directory containing the classification file. Default: current directory
        diff_min:       (Optional) Float: minimum value for display of test DEM versus reference DEM difference map. Default: -20.
        diff_max:       (Optional) Float: maximum value for display of test DEM versus reference DEM difference map. Default: 20.
        dB:             (Optional) Boolean: True: dB scale the reference image if provided. Default: False
        dB_min:         (Optional) Float: minimum dB value for display of dB scaled image
        dB_max:         (Optional) Float: maximum dB value for display of dB scaled image
        img_min:        (Optional) Float: minimum value for display of image
        img_max:        (Optional) Float: minimum value for display of image
        bound_box:      (Optional) List-like of Floats [x_min,y_min,x_max,y_max]: bounding box specifying dimensions of sub area of interest. By default, the entire overlapping area of test_dem_file and ref_dem_file is displayed. Cannot be used at the same time as ctr or dim options.
    """

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    test_dem_path = os.path.join(test_dir, test_dem_file)
    ref_dem_path = os.path.join(ref_dir, ref_dem_file)
    test_md = get_md.GDAL_ds(test_dem_path) 
    ref_md = get_md.GDAL_ds(ref_dem_path)
    ref_srs = osr.SpatialReference()
    ref_srs.ImportFromEPSG(ref_md.srs)        # get spatial reference from EPSG id
    ref_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    test_srs = osr.SpatialReference()
    test_srs.ImportFromEPSG(test_md.srs)
    test_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(ref_srs, test_srs)
    nw_corner = transform.TransformPoint(ref_md.x0, ref_md.y0)
    se_corner = transform.TransformPoint(ref_md.x0 + ref_md.nx*ref_md.dx, ref_md.y0-abs(ref_md.ny*ref_md.dy))
    if bound_box is not None:
        if ctr is not None or dim is not None:
            print("Error: if bound_box is specified, ctr and dim must not be specified")
            return
        x_min = bound_box[0]
        y_min = bound_box[1]
        x_max = bound_box[2]
        y_max = bound_box[3]
        
    elif ctr is not None and dim  is  not None:
        x_max = ctr[0]+dim/2.
        x_min = ctr[0]-dim/2.
        y_max = ctr[1]+dim/2.
        y_min = ctr[1]-dim/2.
        bound_box = [x_min, y_min, x_max, y_max]
    else:
        x_max = np.min(np.asarray([test_md.x0 + test_md.nx*test_md.dx, se_corner[0]]))
        x_min = np.max(np.asarray([test_md.x0, nw_corner[0]]))
        y_max = np.min(np.asarray([test_md.y0, nw_corner[1]]))
        y_min = np.max(np.asarray([test_md.y0 - test_md.ny*np.abs(test_md.dy), se_corner[1]]))
        bound_box = [x_min, y_min, x_max, y_max]
    list_index = 0
    test_index = list_index
    list_index += 1
    ref_index = list_index
    list_index += 1
    subplot_file_list = [test_dem_file, ref_dem_file]
    
    subplot_dir_list = [test_dir, ref_dir]
    if class_file is not None:
        subplot_file_list.append(class_file)
        subplot_dir_list.append(class_dir)
        class_index = list_index
        list_index += 1
    if img_file is not None:
        subplot_file_list.append(img_file)
        subplot_dir_list.append(img_dir)     
        img_index = list_index
        list_index += 1
    sub_files = (imgpro.subplot(subplot_file_list, sub_bound_box = bound_box, in_dirs = subplot_dir_list, out_dirs = test_dir, x_res = test_md.dx, y_res = test_md.dy, srs = test_md.srs))
    if img_file is not None:
        img = cv2.imread(os.path.join(test_dir, sub_files[img_index]), cv2.IMREAD_LOAD_GDAL)
        if img.ndim == 3:
            if img.shape[2] > 3:
                img = img[...,0:3]
            img = img[...,::-1]
        if dB:
            img[np.nonzero(img == 0)] = np.nanmin(img[np.nonzero(img > 0)])
            img = 10*np.log10(np.float64(img))
    if class_file is not None:
        class_img = cv2.imread(os.path.join(test_dir, sub_files[class_index]), cv2.IMREAD_LOAD_GDAL)
    ref_subplot_file = sub_files[ref_index]
    test_subplot_file = sub_files[test_index]
    ref_subplot_path = os.path.join(test_dir, ref_subplot_file)
    test_subplot_path = os.path.join(test_dir, test_subplot_file)
    test_subplot_ds = gdal.Open(test_subplot_path)
    ref_subplot_ds = gdal.Open(ref_subplot_path)
    ref_dem = ref_subplot_ds.ReadAsArray()
    test_dem = test_subplot_ds.ReadAsArray()
    print(f"Max reference DEM ht: {np.max(ref_dem[np.nonzero((test_dem != test_md.no_data) & (ref_dem != ref_md.no_data))])}")
    print(f"Min reference DEM ht: {np.min(ref_dem[np.nonzero((test_dem != test_md.no_data) & (ref_dem != ref_md.no_data))])}")
    if max_ht is None:
        test_max = np.nanmean(test_dem[np.nonzero(test_dem != test_md.no_data)]) + 2*np.nanstd(test_dem[np.nonzero(test_dem != test_md.no_data)])
        ref_max = np.nanmean(ref_dem[np.nonzero(ref_dem != ref_md.no_data)]) + 2*np.nanstd(ref_dem[np.nonzero(ref_dem != ref_md.no_data)])
        max_ht = np.nanmax(np.asarray(test_max, ref_max))
    if min_ht is None:
        test_min = np.nanmean(test_dem[np.nonzero(test_dem != test_md.no_data)]) - 2*np.nanstd(test_dem[np.nonzero(test_dem != test_md.no_data)])
        ref_min = np.nanmean(ref_dem[np.nonzero(ref_dem != ref_md.no_data)]) - 2*np.nanstd(ref_dem[np.nonzero(ref_dem != ref_md.no_data)])
        min_ht = np.nanmin(np.asarray(test_min, ref_min))
    if msl is not None:
        test_dem[np.nonzero(ref_dem < msl)] = min_ht
        ref_dem[np.nonzero(ref_dem < msl)] = min_ht
        if class_file is not None:
            class_img[np.nonzero(ref_dem < msl)] = np.nan
            class_img[np.isnan(ref_dem)] = np.nan
    ref_dem[np.nonzero(ref_dem <= min_ht)] = np.nan
    test_dem[np.nonzero(test_dem <= min_ht)] = np.nan
    ref_dem[np.nonzero(ref_dem >= max_ht)] = np.nan
    test_dem[np.nonzero(test_dem >= max_ht)] = np.nan
    #display the test and reference DEMs
    num_ticks= 5
    x_tick_labels = np.int32(np.arange(0, x_max-x_min, (x_max-x_min)/num_ticks))
    y_tick_labels = np.int32(np.arange(y_max-y_min, 0, (y_min-y_max)/num_ticks))
    fig, axes = plt.subplots(2,2, squeeze = False)
    fig.set_size_inches(12,12)
    divider = make_axes_locatable(axes[0,0])
    ax_cb = divider.append_axes("right", size = "5%", pad = 0.05)
    fig.add_axes(ax_cb)

    test_im = axes[0,0].imshow(test_dem, vmin = min_ht, vmax = max_ht, cmap = 'turbo', interpolation = 'none')
    fig.colorbar(test_im, label = 'Height (m)', cax = ax_cb)
    axes[0,0].set_title('Radargrammetry derived DEM')

    axes[0,0].set_ylabel('Northing (m)')
    axes[0,0].set_xlabel('Easting (m)')
    axes[0,0].set_xticks(np.arange(0,test_dem.shape[1],test_dem.shape[1]/num_ticks), x_tick_labels)
    axes[0,0].set_yticks(np.arange(0,test_dem.shape[0],test_dem.shape[0]/num_ticks), y_tick_labels)

    
    divider = make_axes_locatable(axes[0,1])
    ax_cb = divider.append_axes("right", size = "5%", pad = 0.05)
    fig.add_axes(ax_cb)

    ref_im = axes[0,1].imshow(ref_dem, vmin = min_ht, vmax = max_ht, cmap = 'turbo', interpolation = 'none')
    fig.colorbar(ref_im, label = 'Height(m)', cax = ax_cb)
    axes[0,1].set_title('Reference DEM')
    axes[0,1].set_ylabel('Northing (m)')
    axes[0,1].set_xlabel('Easting (m)')
    axes[0,1].set_xticks(np.arange(0,ref_dem.shape[1],ref_dem.shape[1]/num_ticks), x_tick_labels)
    axes[0,1].set_yticks(np.arange(0,ref_dem.shape[0],ref_dem.shape[0]/num_ticks), y_tick_labels)

    valid = np.nonzero((ref_dem != ref_md.no_data) & (ref_dem > min_ht) & (ref_dem < max_ht) & (test_dem != test_md.no_data) & (test_dem > min_ht) & (test_dem < max_ht) & ~np.isnan(ref_dem) & ~np.isnan(test_dem))
    bias = np.nanmean(test_dem[valid]-ref_dem[valid])
    if class_file is not None:
        axes[1,0].imshow(class_img, vmin = np.nanmin(class_img), vmax = np.nanmax(class_img), interpolation = 'none', cmap = 'PiYG')
        axes[1,0].set_title('Classification Map')
        axes[1,0].set_xticks(np.arange(0,class_img.shape[1],class_img.shape[1]/num_ticks), x_tick_labels)
        axes[1,0].set_yticks(np.arange(0,class_img.shape[0],class_img.shape[0]/num_ticks), y_tick_labels)
    elif img_file is not None:
        if img_min is None:
            img_min = np.nanmin(img)
        if img_max is None:
            img_max = np.nanmax(img)
        axes[1,0].imshow(img, vmin = img_min, vmax = img_max, cmap = 'gray')
        axes[1,0].set_title('Image')
        axes[1,0].set_xticks(np.arange(0,img.shape[1],img.shape[1]/num_ticks), x_tick_labels)
        axes[1,0].set_yticks(np.arange(0,img.shape[0],img.shape[0]/num_ticks), y_tick_labels)
    else:
        test_no_bias = test_dem-bias
        #display bias-corrected test dem
        divider = make_axes_locatable(axes[1,0])
        ax_cb = divider.append_axes("right", size = "5%", pad = 0.05)
        fig.add_axes(ax_cb)
        test_no_bias_im = axes[1,0].imshow(test_no_bias, vmin = min_ht, vmax = max_ht, cmap = 'turbo', interpolation = 'none')
        fig.colorbar(test_no_bias_im, label = 'Height(m)', cax = ax_cb)
        axes[1,0].set_title('Bias-corrected Test DEM')
        axes[1,0].set_xticks(np.arange(0,test_no_bias.shape[1], test_no_bias.shape[1]/num_ticks), x_tick_labels)
        axes[1,0].set_yticks(np.arange(0,test_no_bias.shape[0], test_no_bias.shape[0]/num_ticks), y_tick_labels)
    axes[1,0].set_ylabel('Northing (m)')
    axes[1,0].set_xlabel('Easting (m)')
        
    divider = make_axes_locatable(axes[1,1])
    ax_cb = divider.append_axes("right", size = "5%", pad = 0.05)
    fig.add_axes(ax_cb)
    diff = test_dem-ref_dem-bias
    no_data_diff = np.nonzero((test_dem == test_md.no_data) | (ref_dem == ref_md.no_data))
    diff[no_data_diff] = test_md.no_data
    valid_diff = np.nonzero(diff != test_md.no_data)
    diff[np.nonzero(diff > diff_max)] = diff_max
    diff[np.nonzero(diff < diff_min)] = diff_min
    diff_im = axes[1,1].imshow(diff, vmin = diff_min, vmax = diff_max, cmap = 'RdBu_r', interpolation = 'none')
    axes[1,1].set_title('Test DEM Error')
    axes[1,1].set_ylabel('Northing (m)')
    axes[1,1].set_xlabel('Easting (m)')
    axes[1,1].set_xticks(np.arange(0, diff.shape[1], diff.shape[1]/num_ticks), x_tick_labels)
    axes[1,1].set_yticks(np.arange(0, diff.shape[0], diff.shape[0]/num_ticks), y_tick_labels)
    fig.colorbar(diff_im, label = 'Height(m)', cax = ax_cb)
    
    plt.tight_layout()
    plt.show()
    plt.pause(1)
    plt.close()
    #dsplay a scatter plot of the difference
    if class_file is not None:
        imgpro.scatter(ref_dem[np.nonzero(class_img == 1)], test_dem[np.nonzero(class_img == 1)], min_value = min_ht, max_value = max_ht, no_data_x = ref_md.no_data, no_data_y = test_md.no_data, title = "Radargrammetry vs Reference DEM Height (Forested)", title_x = 'Reference DEM Height (m)', title_y = 'Radargrammetry DEM Height (m)', min_cts = min_cts, plot_res = dz, max_cts = max_cts)
        imgpro.scatter(ref_dem[np.nonzero(class_img == 0)], test_dem[np.nonzero(class_img == 0)], min_value = min_ht, max_value = max_ht, no_data_x = ref_md.no_data, no_data_y = test_md.no_data, title = "Radargrammetry vs Reference DEM Height (Unforested)", title_x = 'Reference DEM Height (m)', title_y = 'Radargrammetry DEM Height (m)', min_cts = min_cts, plot_res = dz, max_cts = max_cts)
        
    else:
        imgpro.scatter(ref_dem, test_dem, min_value = min_ht, max_value = max_ht, no_data_x = ref_md.no_data, no_data_y = test_md.no_data, title_x = 'Reference DEM Height (m)', title_y = 'Radargrammetry DEM Height (m)', min_cts = min_cts, plot_res = dz, max_cts = max_cts)

def manual_offset(in_file, out_file, offset, in_dir = ".", out_dir = "."):
    """
    manually offset dem by offset value and write to a new file
    REQUIRED INPUTS:
       in_file:    String: input DEM file name
       out_file:   String: output DEM file name
       offset:     Float: offset value to apply
    OPTIONAL INPUTS:
       in_dir:     String: input DEM file directory. Default: current directory
       out_dir:     String: output DEM file directory. Default: current directory
    """
    in_path = os.path.join(in_dir, in_file)
    out_path = os.path.join(out_dir, out_file)
    ds = gdal.Open(in_path)
    band = ds.GetRasterBand(1)
    dem_in = band.ReadAsArray()
    dem_out = dem_in + offset
    driver = gdal.GetDriverByName("GTiff")
    data_out = driver.Create(out_path, dem_in.shape[1], dem_in.shape[0], 1, band.DataType)
    data_out.SetGeoTransform(ds.GetGeoTransform())
    data_out.SetProjection(ds.GetProjection())
    data_out.GetRasterBand(1).WriteArray(dem_out)
    data_out.GetRasterBand(1).SetNoDataValue(band.GetNoDataValue())
    data_out.FlushCache()
    data_out = None
    band = None
    ds = None



def rectify_imgs (img_hi, img_lo, md_hi, md_lo):
    """
    find rotation angle off of east exits for each satellite
    REQUIRED INPUTS:
       img_hi:     numpy array: high incidence angle image array
       img_lo:     numpy array: high incidence angle image array
       md_hi:      Metadata_SAR object: metadata associated with img_hi
       md_lo:      Metadata_SAR object: metadata associated with img_lo
    OUTPUT
       A dictionary contianing the following keys:
           rect_hi:        numpy array of same data type as img_hi: rectified high incidence angle image
           rect_lo:        numpy array of same data type as img_lo: rectified lo incidence angle image - primary
           phi:            Float: angle (in degrees) between the two images
           rotation_angle: Float: angle (in degrees) the images were rotated
           angle_lo:       Float: angle between low incidence angle image and East
    """
    angle_hi = enu_angle(md_hi)
    angle_lo = enu_angle(md_lo)
    phi = angle_hi - angle_lo
    beta_func = lambda b: beta_function(b, phi, md_lo.inc, md_hi.inc)
    beta = optimize.root_scalar(beta_func, method = 'newton', x0 = 0.).root
    rotation_angle = (angle_lo-beta)

    #difference between low and high incidence angle image ellipsoid heights. need to use this to correct imagery
    projection_error = md_lo.ht-md_hi.ht
    disparity_error = 1/np.tan(md_hi.inc*np.pi/180)*projection_error
    shift_x = -np.int32(np.round(np.cos(rotation_angle*np.pi/180)*disparity_error/md_hi.dx))
    shift_y = -np.int32(np.round(np.sin(rotation_angle*np.pi/180)*disparity_error/md_hi.dy))
    #shift high incidence image to align with coordinates of low incidence image ellipsoid 
    img_hi = np.roll(img_hi, shift_y, axis = 0)
    img_hi = np.roll(img_hi, shift_x, axis = 1)
    img_hi[np.min([shift_y,0]):np.max([-1,shift_y]),:] = md_hi.no_data
    img_hi[:,np.min([0,shift_x]):np.max([-1,shift_x])] = md_hi.no_data
    
    rect_hi = ndimage.rotate(img_hi, -rotation_angle, mode = 'constant', cval = md_hi.no_data, reshape = False)
    rect_lo = ndimage.rotate(img_lo, -rotation_angle, mode = 'constant', cval = md_lo.no_data, reshape = False)
    return {'rect_hi':rect_hi, 'rect_lo':rect_lo, 'phi':phi, 'rotation_angle':rotation_angle, 'angle_lo':angle_lo}

def enu_angle(md_sar):
    """
    calculate angle in degrees in the East-North plane of the pointing vector for the provided SAR metadata
    """
    ang = 180/np.pi*np.arctan2(md_sar.pt_vec[1]/np.sqrt(np.sum(md_sar.pt_vec[0:1]**2)), md_sar.pt_vec[0]/np.sqrt(np.sum(md_sar.pt_vec[0:1]**2)))
    if ang < 0:
        ang += 360    
    return ang

def beta_function(b, phi, inc_lo, inc_hi):
    """
    function to be wrapped for optimization to solve for disparity angle beta. Accepts angles in degrees, returns angle in degrees
    """
    phi_rad = phi/180*np.pi
    b_rad = b/180*np.pi
    inc_lo_rad = inc_lo/180*np.pi
    inc_hi_rad = inc_hi/180*np.pi
    return (np.sin(np.pi-b_rad-phi_rad)*np.tan(inc_lo_rad)-np.tan(inc_hi_rad)*np.sin(b_rad))
