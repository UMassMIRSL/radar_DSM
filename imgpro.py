#imgpro.py
#image processing related functions for the radg package
from radg import get_md
from osgeo import gdal, osr
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
def subplot (file_list, x_res = None, y_res = None, in_dirs = '.', out_dirs = '.', out_file_list = None, srs = None, no_datas = None, ctr = None, dim = None, sub_bound_box = None):
    """
    create subplots of the files in the provided list. The subplots will have the same ground resolution and area extent, and number of pixels.
    REQUIRED INPUTS:
        file_list:  list of Strings: files to process subplots from
    OPTIONAL INPUTS:
        NOTE: both ctr and dim must be set, or sub_bound_box must be set
        ctr:           Tuple of Floats (x_ctr, y_ctr): the x and y coordinates of the center of the subplot. It is assumed that ctr is in the coordinate system of srs. If this parameter is specified, dim must also be specified and sub_bound_box cannot be specified.
        dim:           Float: dimension of one side of the subplot. The subplot will be a square of side dimension x dimension, centered on ctr. It is assumed that dim has the same units as the pixel size. If this parameter is specified, dim must also be specified and sub_bound_box cannot be specified.
        x_res:         Float: output x resolution of files. Defaults to coarsest x resolution of files
        y_res:         Float: output y resolution of files. Defaults to coarsest y resolution of files
        in_dirs:       List of Strings: directories input files are located in. If only one directory is supplied, then it is applied to all files
        out_dirs:      List of Strings: directories output files are located in. If only one directory is supplied, then it is applied to all files
        srs:           Int: spatial reference system to output files in. Accepted format: EPSG code. Default: srs of first file in list
        no_datas:      List of Floats: list of no_data values for the files in file_list. Note this list must be the same length as file_list if it is included. If it is not included, the file metadata is used to select the no_data values.
        sub_bound_box: List of Floats [xmin, xmax, ymin, ymax]:  the extents of the subplot bounding box. If this keyword is supplied, the ctr and dim keywords cannot be provided
    OUTPUT
        List of Strings: the file names of the subplots
    """
    if sub_bound_box is not None and (dim is not None or ctr is not None):
        print('Error: if sub_bound_box is defined than ctr and dim cannot be defined. Exiting...')
        return([''])
    if (ctr is not None) ^ (dim is not None):
        print('Error: both ctr and dim must be defined, or sub_bound_box must be defined. Exiting...')
        return([''])
    #subplot bounding box in format (left, bottom, right, top)
    if sub_bound_box is None and ctr is not None and dim is not None:
        x_ctr = np.float64(ctr[0])
        y_ctr = np.float64(ctr[1])
        sub_bound_box = [x_ctr - dim/2, y_ctr - dim/2, x_ctr + dim/2, y_ctr + dim/2]
    dx_max = np.float64(0)
    dy_max = np.float64(0)
    dx_min = np.float64(0)
    dy_min = np.float64(0)
    md_files = list()
    if out_file_list is None:
        use_default_filenames = True
        out_file_list = []
    else:
        use_default_filenames = False
    #if only one file provided treat it as a list
    if not isinstance(file_list, list):
        file_list = [file_list]
    #if only one input or output directory is provided us it for all files
    if not isinstance(in_dirs,list):
        in_dirs = [in_dirs] * len(file_list)
    if len(in_dirs) == 1:
        in_dirs = in_dirs * len(file_list)
    if not isinstance(out_dirs,list):
        out_dirs = [out_dirs] * len(file_list)
    if len(out_dirs) == 1:
        out_dirs = out_dirs * len(file_list)
    
    #verify subplot box is in the extent of all files and find the minimum and maximum resolution values
    for f in range(len(file_list)):
        img_file = file_list[f]
        md_files.append(get_md.GDAL_ds(os.path.join(in_dirs[f],img_file)))
        if srs is None: 
            srs = md_files[f].srs
        file_srs = osr.SpatialReference()
        file_srs.ImportFromEPSG(md_files[f].srs)        # get spatial reference from EPSG id
        file_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(srs)
        target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osr.CoordinateTransformation(file_srs, target_srs)
        nw_corner = transform.TransformPoint(md_files[f].x0, md_files[f].y0)
        se_corner = transform.TransformPoint(md_files[f].x0 + md_files[f].nx*md_files[f].dx, md_files[f].y0-abs(md_files[f].ny*md_files[f].dy))
        bound_box = [nw_corner[0], se_corner[0], se_corner[1], nw_corner[1]] #find bounding box for source image

        if not(bound_box[0] <= sub_bound_box[0] <= sub_bound_box[2] <= bound_box[2] and bound_box[1] <= sub_bound_box[1] <= sub_bound_box[3] <= bound_box[3]):  
            #if the bounding box is outside the source image, return an error and blank strings
            print("specified subplot outside of image")
            return ['']
        #establish minimum and maximium resolutions. By default, all images will be resampled to the larger resolution through nearest neighbor sampling.
        if np.abs(md_files[f].dx) > dx_max: 
            dx_max = md_files[f].dx
        if np.abs(md_files[f].dy) > dy_max:
            dy_max = md_files[f].dy
        if np.abs(md_files[f].dx) < dx_min: 
            dx_min = md_files[f].dx
        if np.abs(md_files[f].dy) < dy_min:
            dy_min = md_files[f].dy
        ext = img_file.find('.tif')
        if use_default_filenames:
            out_file = img_file[0:ext] + '_subplot.tif'
            out_file_list.append(out_file) 
    #after the files have been validated, construct gdalwarp command arguments
    if x_res is None:
        x_res = dx_max
    if y_res is None:
        y_res = dy_max
    width = int((sub_bound_box[2] - sub_bound_box[0])/abs(x_res))
    height = int((sub_bound_box[3] - sub_bound_box[1])/abs(y_res))
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(srs)        # get spatial reference from EPSG id
    for f in range(len(file_list)):
        if no_datas is not None:
            no_data = no_datas[f]
        else:
            no_data = md_files[f].no_data
        warp = gdal.WarpOptions(width = width, height = height, dstSRS = target_srs, outputBounds = sub_bound_box, resampleAlg = "bilinear", srcNodata = no_data, dstNodata = no_data)
        gdal.Warp(os.path.join(out_dirs[f], out_file_list[f]), os.path.join(in_dirs[f], file_list[f]), options = warp)
    return out_file_list


def byte_scale(arr, min_value = None, max_value = None):
    """
    scale an array from 0-255 based on the array minimum and maximum or predefined values
    REQUIRED INPUTS:
       arr:        numpy array: data to be scaled
    OPTIONAL INPUTS:
       min_value:  (Optional) data type of arr: minimum scaling value. defaults to minimum of arr
       max_value:  (Optional) data type of arr: maximum scaling value. defaults to maximum of arr
    OUTPUT:
       numpy uint8 array: scaled data
    """
    if min_value is None:
        min_value = np.min(arr)
    if max_value is None:
        max_value = np.max(arr)
    for pt in np.argwhere(arr < min_value):
        arr[pt] = min_value
    for pt in np.argwhere(arr > max_value):
        arr[pt] = max_value
    byte_arr = np.uint8((arr - min_value)*255/(max_value-min_value))
    return byte_arr

def int16_scale(arr, min_value = None, max_value = None):
    """
    scale an array from 0-65535 based on the array minimum and maximum or predefined values
    REQUIRED INPUTS:
       arr: numpy array of data to be scaled
    OPTIONAL INPUTS:
       min_value: (Optional) minimum scaling value. defaults to minimum of arr
       max_value: (Optional) maximum scaling value. defaults to maximum of arr
    OUTPUT:
       numpy uint16 array: scaled data
    """
    if min_value is None:
        min_value = np.min(arr)
    if max_value is None:
        max_value = np.max(arr)
    for pt in np.argwhere(arr < min_value):
        arr[pt] = min_value
    for pt in np.argwhere(arr > max_value):
        arr[pt] = max_value
    int16_arr = np.uint16((arr - min_value)/(max_value-min_value)*65535)
    return int16_arr

def scatter(x_data, y_data, min_value = None, max_value = None, no_data_x = np.nan, no_data_y = np.nan, plot_res = 0.1, title_x = 'x', title_y = 'y', min_cts = 20, max_cts = None, title = None):
    """
    display data in a scatter plot/2D histogram
    REQUIRED INPUTS:
        x_data:     Numpy array: input data to be plotted on the x-axis
        y_data:     Numpy array: input data to be plotted on the y-axis
    OPTIONAL INPUTS:
        min_value:  Numeric of same type as x_data and y_data: minimum data value to be plotted on axis. Default: None
        max_value:  Numeric of same type as x_data and y_data: maximum data value to be plotted on axis. Default: None
        no_data_x:  Numeric of same type as x_data: no data value in x data set. This value will be excluded from scatter plot. Default: np.nan
        no_data_y:  Numeric of same type as y_data: no data value in y data set. This value will be excluded from scatter plot. Default: np.nan
        plot_res:   Float: cell resultion of scatter plot. Default: 0.1
        title_x:    String: Title of x-axis of plot. Default: 'x'
        title_y:    String: Title of y-axis of plot. Default: 'y'
        min_cts:    Int: minimum number of counts in a cell to display the cell value. Default: 20
        max_cts:    Int: maximum number of counts in color scale. Default: None
        title:      String: Title of scatter plot
    """
    x = x_data.flatten() 
    y = y_data.flatten()
    if title is None:
        title = f"{title_y} vs {title_x}"
    if min_value is None:
        min_value = np.nanmin(np.asarray([np.nanmin(x[np.nonzero(x != no_data_x)]), np.nanmin(y[np.nonzero(y != no_data_y)])]))
    if max_value is None:
        max_value = np.nanmax(np.asarray([np.nanmax(x[np.nonzero(x != no_data_x)]), np.nanmax(y[np.nonzero(y != no_data_y)])]))
    valid = np.nonzero((x != no_data_x) & (x > min_value) & (x < max_value) & (y != no_data_y) & (y > min_value) & (y < max_value) & ~np.isnan(x) & ~np.isnan(y))
    fig = plt.figure()
    bias = np.nanmean(y[valid]-x[valid])
    rms = np.sqrt(np.nanmean((y[valid]-x[valid])**2)-np.nanmean(y[valid]-x[valid])**2)
    hist = plt.hist2d(x[valid], y[valid], bins = int((max_value-min_value)/plot_res), cmap = 'turbo', cmin = min_cts, cmax = max_cts, range = [[min_value, max_value],[min_value, max_value]])
    plt.plot(np.arange(min_value, max_value,(max_value-min_value)/100), np.arange(min_value, max_value,(max_value-min_value)/100), ls ='--', lw = 0.75, c = 'k')
    plt.title(title)
    plt.xlabel(title_x, size = 'x-large')
    plt.ylabel(title_y, size = 'x-large')
    plt.text(min_value,max_value, f'\n  Bias (m): {bias:.3f}\n  RMS Error (m): {rms:.3f}', verticalalignment = 'top', size = 'x-large')
    fig.set_size_inches(8,6)
    fig.colorbar(hist[3], ax = fig.get_axes(), label = 'Counts')
    plt.show()
    plt.pause(1)
    plt.close()

def multilook(arr, md, nl_x, nl_y):
    """
    multilook GeoTIFF and update associated metadata
    REQUIRED INPUTS:
       arr:      numpy array: image array to be multilooked
       md:     Metadata_SAR object: metadata associated with a
       nl_x:   Int: multilooking factor in x direction
       nl_y:   Int: multilooking factor in y direction
    OUTPUT:
       tuple containing:
           arr:    numpy array of same type as arr: multilooked array
           md:     Metadata_SAR object: adjusted metadata based on multilooking
    """
    md.dx *= nl_x
    md.dy *= nl_y
    nx = np.uint32(md.nx//nl_x) * nl_x
    ny = np.uint32(md.ny//nl_y) * nl_y
    arr = arr[0:ny, 0:nx]
    arr = rebin(arr, (ny//nl_y, nx//nl_x))
    md.nx = nx//nl_x
    md.ny = ny//nl_y
    return (arr, md)

def rebin(a, in_shape):
    """
    Downsample 2D array a to shape by using nearest neighbor averaging
    REQUIRED INPUTS:
       a:       numpy array: array to be resampled
       shape:   Int array-like containing the desired output dimensions. Note that the dimensions of a must be integer factors of the dimensions of shape, but need not be the same integer factor.
    OUTPUT
       resampled numpy array of the same data type as a
    """
    if a.shape[0] % in_shape[0] != 0:
        print("warning: input array x dimension not integer factor of output x shape")
        return
    if a.shape[0] % in_shape[0] != 0:
        print("warning: input array y dimension not integer factor of output y shape")
        return
    x_factor = a.shape[0] // in_shape[0]
    y_factor = a.shape[1] // in_shape[1]
    out_shape = (in_shape[0], x_factor, in_shape[1], y_factor )
    return np.nanmean(np.nanmean(a.reshape(out_shape), axis = -1), axis = 1)

def rotate_nan(img, rotation_angle):
    """
    perform rotation of image and nan mask to preserve nan values 
    REQUIRED INPUTS:
       img:                numpy array: image array to be rotated
       rotation_angle:     Float: angle to rotate image by
    OUTPUT:
       rot_img:            numpy array of same type as img: rotated image array
    """
    valid = np.nonzero(~np.isnan(img))
    valid_array = ~np.isnan(img)
    points = (valid[0], valid[1])
    interp_pts = (np.arange(img.shape[0]).repeat(img.shape[1]), np.arange(np.asarray(img.shape).prod())%img.shape[1])
    img = griddata(points, img[valid], interp_pts, method = 'nearest', fill_value = np.min(img)).reshape(img.shape)
    #rotate rectified images back to ground coordinates
    rot_img = ndimage.rotate(img, rotation_angle, mode = 'constant', cval = np.min(img), reshape = False, order = 1)
    valid_array_rot = ndimage.rotate(valid_array, rotation_angle, mode = 'constant', cval = 0, reshape = False, order = 0)
    rot_img[np.nonzero(valid_array_rot == 0)] = np.nan
    return rot_img
