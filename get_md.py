#get_md.py
#ingest metadata from files for further processing. Multiple methods exist for different file types
import json
import os
import re
import datetime
import numpy as np
from datetime import timezone
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import xml.etree.ElementTree as et
from numpy.polynomial import Polynomial

def tdx(img_dir, src_dir = '.'):
    """
    REQUIRED INPUTS: 
        img_dir: String: TanDEM-X (TDX) SAR image directory structure
    OPTIONAL INPUTS:
        src_dir: String: directory location of img_dir. Default: current directory
    OUTPUT: 
       Metadata_SAR object containing the metadata from the TDX file
    """
    xml_file = img_dir + '.xml'
    xml_path = os.path.join(src_dir, img_dir, xml_file)
    tree = et.parse(xml_path)
    root = tree.getroot()
    img_path_xml = root.find('productComponents').find('imageData').find('file').find('location').find('path')
    img_file_xml = root.find('productComponents').find('imageData').find('file').find('location').find('filename')
    img_path = os.path.join(src_dir, img_dir, img_path_xml.text)
    md = Metadata_SAR(GDAL_ds(img_file_xml.text, src_dir = img_path))
    #get center frequency
    ctr_freq = np.float64(root.find('instrument').find('radarParameters').find('centerFrequency').text)
    #get center coordinate and average scene height in latitude, longitude, & altitude coordinates
    ht = np.float32(root.find('productInfo').find('sceneInfo').find('sceneAverageHeight').text)
    ctr_xml = root.find('productSpecific').find('geocodedImageInfo').find('geoParameter').find('sceneCenterCoordsGeographic')
    ctr_lon = np.float32(ctr_xml.find('centerCoordLongitude').text)
    ctr_lat = np.float32(ctr_xml.find('centerCoordLatitude').text)
    #get satellite track vector (in ITRF2014 data frame). I think this is pretty close to the "ECEF" data provided by Capella, so we should be able to use the same transform functions from here on out. ITRF is *an* ECEF frame, so it should work I think
    state_vec_xml = root.find('platform').find('orbit').findall('stateVec')
    sat_x_list = []
    sat_y_list = []
    sat_z_list = []
    secs_list = []
    for state in state_vec_xml:
        sat_x_list.append(np.float64(state.find('posX').text))
        sat_y_list.append(np.float64(state.find('posY').text))
        sat_z_list.append(np.float64(state.find('posZ').text))
        secs_list.append(np.float64(state.find('timeGPS').text))
    secs = np.asarray(secs_list)
    #find a quadratic function that fits the data torugh the least squares method
    sat_x_poly = Polynomial.fit(secs, np.asarray(sat_x_list), 2)
    sat_y_poly = Polynomial.fit(secs, np.asarray(sat_y_list), 2)
    sat_z_poly = Polynomial.fit(secs, np.asarray(sat_z_list), 2)
    total_time = (secs[-1]-secs[0])
    #find points of the function along this curve. Choose an interval of 0.5 seconds to start
    interval = 0.5
    N = int(np.float32(total_time)/interval)
    sat_x = np.asarray(sat_x_poly.linspace(n = N)[1])
    sat_y = np.asarray(sat_y_poly.linspace(n = N)[1])
    sat_z = np.asarray(sat_z_poly.linspace(n = N)[1])
    time = np.arange(secs[0], secs[-1], interval)
    
    #find ellipsoid parameters for GCPs to convert center to ECEF from LLA: https://gssc.esa.int/navipedia/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion
    lat_rad = ctr_lat*np.pi/180
    lon_rad = ctr_lon*np.pi/180
    annots = root.find('productComponents').findall('annotation')
    for note in annots:
        if note.find('type').text == 'GEOREF':
            geo_file = note.find('file').find('location').find('filename').text
            geo_path = os.path.join(src_dir, img_dir, note.find('file').find('location').find('path').text, geo_file)

    geo_tree = et.parse(geo_path) 
    geo_root = geo_tree.getroot()
    #a = semi-major axis of ellipse, b = semi-minor axis, C = radius of curvature (N in provided link)
    a = np.float64(geo_root.find('referenceFrames').find('sphere').find('semiMajorAxis').text)
    b = np.float64(geo_root.find('referenceFrames').find('sphere').find('semiMinorAxis').text)
    e2 = (a**2-b**2)/(a**2)
    C = a/np.sqrt(1-e2*(np.sin(lat_rad)**2))
    ctr_x = (C+ht)*np.cos(lat_rad)*np.cos(lon_rad)
    ctr_y = (C+ht)*np.cos(lat_rad)*np.sin(lon_rad)
    ctr_z = ((1-e2)*C+ht)*np.sin(lat_rad)
    ctr = np.asarray((ctr_x,ctr_y,ctr_z))
    md.pt_vec = ecef2enu(dopp_ctr_vec(sat_x, sat_y, sat_z, time, ctr, ctr_freq), ctr_lat, ctr_lon)
    md.rad_scl = np.float64(root.find('calibration').find('calibrationConstant').find('calFactor').text)
    md.inc = np.float64(root.find('productInfo').find('sceneInfo').find('sceneCenterCoord').find('incidenceAngle').text)
    md.ht = np.float64(ht)
    md.no_data = 0
    return md

def capella(in_file, src_dir = '.'):
    """
    NOTE: this function requires the base Capella GEC TIFF file, and the _extended.json files to be located in the same location
    REQUIRED INPUTS: 
        in_file: Capella GEC GeoTIFF file name
    OPTIONAL INPUTS:
        src_dir: String: directory location of in_file. Default: current directory
    OUTPUT: 
       Metadata_SAR object containing the metadata from the capella file
    """
    ext = in_file.find('.tif')
    ext_json_file = in_file[0:ext] + '_extended.json'
    ext_json_path = os.path.join(src_dir, ext_json_file)
    img_path = os.path.join(src_dir, in_file)
    img_file = in_file
    md = Metadata_SAR(GDAL_ds(in_file, src_dir = src_dir))
    #get Metadata from _extended.json file
    json_obj = open(ext_json_path,'r')
    md_json = json.load(json_obj)
    #incidence angle (degrees)
    md.inc = np.float64(md_json['collect']['image']['center_pixel']['incidence_angle'])
    #radiometric scale factor
    md.rad_scl = np.float64(md_json['collect']['image']['scale_factor'])
    #get ellipsoid projection height
    md.ht = np.float64(re.split("[\[\]]", md_json['collect']['image']['terrain_models']['reprojection']['name'])[1])
    #center pixel information (in ECEF)
    ctr = np.asarray(md_json['collect']['image']['center_pixel']['target_position'], dtype = np.double)
    #x, y, and z satellite track coordinates (in ECEF)
    vector = md_json['collect']['state']['state_vectors']
    sat_x = np.asarray([pt['position'][0] for pt in vector], dtype = np.double)
    sat_y = np.asarray([pt['position'][1] for pt in vector], dtype = np.double)
    sat_z = np.asarray([pt['position'][2] for pt in vector], dtype = np.double)

    #get timestamps and convert them to Unix
    times = [pt['time'] for pt in vector]
    time_arr = [re.split("[\.T:-]", t.rstrip('Z')) for t in times]
    secs = np.asarray([datetime.datetime(int(t[0]),int(t[1]),int(t[2]),hour=int(t[3]),minute=int(t[4]),second=int(t[5]), microsecond = int(float(t[6])/1e3), tzinfo=timezone.utc).timestamp() for t in time_arr], dtype = np.double)
 
    #get radar center frequnecy, used to calculate wavelength
    ctr_freq = float(md_json['collect']['radar']['center_frequency'])
    
    #get latitude and longitude
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(md.srs)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)     # WGS84/Geographic
    transform = osr.CoordinateTransformation(src_srs, target_srs)
    (lat_deg, lon_deg, alt) = transform.TransformPoint(md.x0+md.nx/2*md.dx, md.y0-md.ny/2*md.dy)
    #get unit vector from satellite to scene center at minimum doppler frequency
    pt_vec = ecef2enu(dopp_ctr_vec(sat_x, sat_y, sat_z, secs, ctr, ctr_freq), lat_deg, lon_deg)
    md.pt_vec  = pt_vec
    return md   

def dopp_ctr_vec(sat_x, sat_y, sat_z, secs, ctr, ctr_freq):
    """
    create a unit vector poining from the satellite position at the doppler center frequency to the scene center
    REQUIRED INPUTS: 
      sat_x:       np.double numpy array:   satellite x postions
      sat_y:       np.double numpy array:   satellite y postions
      sat_z:       np.double numpy array:   satellite z postions
      secs:        np.double numpy array:   POSIX/UNIX position timestamps
      ctr:         np.double numpy array:   scene center position. Must be in the same coordinate system as sat_x, sat_y, and sat_z
      ctr_freq:    np.double:               satellite center frequency
    OUTPUT:
       pt_vec:     np.double numpy array:   3D satellite pointing vector at minimum Doppler frequency (center of acquisition) in ECEF coordinates 
    """
#calculate the index of the Doppler Center frequency
    #range to center point
    ctr_range = np.sqrt((sat_x-ctr[0])**2+(sat_y-ctr[1])**2+(sat_z-ctr[2])**2)
    #radial velocity
    n = np.prod(secs.shape)
    rad_vel = ctr_range
    rad_vel[0:n-2] = (ctr_range[1:n-1]-ctr_range[0:n-2])/(secs[1]-secs[0])
    rad_vel[n-1] = 2*rad_vel[n-2] - rad_vel[n-3]
    
    #doppler frequency
    c = 3e8
    wavelen = c/ctr_freq
    dopp_freq = -2*rad_vel/wavelen
 
    #doppler frequency center index. used to determine the pointing direction of the satellite
    dopp_ctr_ind = np.argmin(np.abs(dopp_freq))
    #interpolation to determine the pointing direction more accurately
    itp_start = np.max([0, dopp_ctr_ind-2])
    itp_end = np.min([dopp_ctr_ind+2, np.prod(dopp_freq.shape)-1])
    N = 50
    itp_pts = np.arange(secs[itp_start],secs[itp_end],(secs[itp_end]-secs[itp_start])/N)
    dopp_freq_itp = np.interp(itp_pts, secs[itp_start:itp_end], dopp_freq[itp_start:itp_end])
    itp_ctr = np.argmin(dopp_freq_itp)
    #interpolate satelite positions on same scale as doppler shift
    sat_x_itp =  np.interp(itp_pts, secs[itp_start:itp_end], sat_x[itp_start:itp_end])
    sat_y_itp =  np.interp(itp_pts, secs[itp_start:itp_end], sat_y[itp_start:itp_end])
    sat_z_itp =  np.interp(itp_pts, secs[itp_start:itp_end], sat_z[itp_start:itp_end])
    #calcualte range from doppler center
    ctr_range_itp = np.sqrt((sat_x_itp[itp_ctr]-ctr[0])**2+(sat_y_itp[itp_ctr]-ctr[1])**2+(sat_z_itp[itp_ctr]-ctr[2])**2)
    #generate unit vector at doppler center
    pt_vec = np.asarray([(sat_x_itp[itp_ctr]-ctr[0]),(sat_y_itp[itp_ctr]-ctr[1]),(sat_z_itp[itp_ctr]-ctr[2])], dtype = np.double)/ctr_range_itp
    return(pt_vec)


def GDAL_ds(in_file, src_dir = '.'):
    """
    get relevant metadata from a GDALdataset object
    REQUIRED INPUTS: 
       in_file: GeoTIFF file name
    OPTIONAL INPUTS:
       src_dir: directory containing in_file
    OUTPUT: Metadata object containing the information from the GeoTIFF:
    """
    md = Metadata()
    #get Metadata from TIFF file
    ds = gdal.Open(os.path.join(src_dir,in_file))
    #get EPSG code for spatial reference
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    md.srs = int(proj.GetAttrValue('AUTHORITY',1))
    #get raster origin and pixel spacing
    geotransform = np.asarray(ds.GetGeoTransform(), dtype = np.double)
    md.x0 = geotransform[0]
    md.y0 = geotransform[3]
    md.dx = geotransform[1]
    md.dy = geotransform[5]
    md.nx = ds.RasterXSize
    md.ny = ds.RasterYSize
    md.num_bands = ds.RasterCount
    md.no_data = ds.GetRasterBand(1).GetNoDataValue()
    md.img_file  = in_file
    md.img_path = os.path.join(src_dir, in_file)
    md.img_dir =src_dir
    return md

def ecef2enu (ecef_vec, lat_deg, lon_deg):
    """
    convert a unit vector in ECEF units to a unit vector in ENU units using the transfromation matrix available at https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates
    REQUIRED INPUTS:
       ecef_vec: np.double numpy array:  unit vector in ECEF coordinates
       lat_deg:  np.double: latitude of scene center, in degrees
       lon_deg:  np.double: longitude of scene center, in degrees
    OUTPUT:
       np.double numpy array: 3D unit vector in ENU coordinates
    """
    lat_rad = lat_deg*np.pi/180
    lon_rad = lon_deg*np.pi/180
    conv_mtx  = np.array([[-np.sin(lon_rad), np.cos(lon_rad), 0], [-np.cos(lon_rad)*np.sin(lat_rad), -np.sin(lon_rad)*np.sin(lat_rad), np.cos(lat_rad)], [np.cos(lon_rad)*np.cos(lat_rad), np.sin(lon_rad)*np.cos(lat_rad), np.sin(lat_rad)]], np.double)
    enu_vec = np.matmul(conv_mtx, ecef_vec)
    return enu_vec

class Metadata:
    """
    basic class to hold the most used geotiff metadata in this package
    FIELDS
       srs:        Int: EPSG code of coordinate system
       x0:         np.double: x coordinate of raster origin
       y0:         np.double: y coordinate of raster origin
       dx:         np.double: pixel width
       dy:         np.double: pixel height
       nx:         Int: number of pixel columns
       ny:         Int: number of pixel rows
       num_bands:  Int: number of raster bands
       no_data:    np.float: no data value of first raster band
       img_file:   String: file name of image
       img_path:   String: path to image
       img_dir:    String: directory containing image
    """

    def __init__(self):
        self.img_path = '.'
        self.img_file = ''
        self.img_dir = '.'
        self.nx = np.float32(0)
        self.ny = np.float32(0)
        self.x0 = np.float32(0)
        self.y0 = np.float32(0)
        self.dx = np.nan
        self.dy = np.nan
        self.srs = 0
        self.num_bands = 0
        self.no_data = np.nan
    def print(self):
        print('Metadata Object')
        print(f'\timg_path: {self.img_path}')
        print(f'\timg_dir: {self.img_path}')
        print(f'\timg_file: {self.img_file}')
        print(f'\tnx: {self.nx}')
        print(f'\tny: {self.ny}')
        print(f'\tx0: {self.x0}')
        print(f'\ty0: {self.y0}')
        print(f'\tdx: {self.dx}')
        print(f'\tdy: {self.dy}')
        print(f'\tsrs: {self.srs}')
        print(f'\tnum_bands: {self.num_bands}')
        print(f'\tno_data: {self.no_data}')

class Metadata_SAR(Metadata):
    """
    child of the metadata class which holds additional material related specifically to SAR files
    ADDITIONAL FIELDS:
        inc:     incidence angle (deg)
        rad_scl:  radiometric scale factor 
        ht:      WGS84 ellipsoid projection height (m)
        pt_vec:   satellite pointing vector in ENU coordinates at zero Doppler frequency
        img_path: path within directory to location of image
        img_file: file name of image
    """
    def __init__(self, md = None):
        self.inc = np.nan
        self.rad_scl = np.nan
        self.ht = np.nan
        self.pt_vec = np.array([np.nan, np.nan, np.nan])
        if md is not None:
            self.nx = md.nx
            self.ny = md.ny
            self.x0 = md.x0
            self.y0 = md.y0
            self.dx = md.dx
            self.dy = md.dy
            self.srs = md.srs
            self.num_bands = md.num_bands
            self.no_data = md.no_data
            self.img_path = md.img_path
            self.img_file = md.img_file
            self.img_dir = md.img_dir
    def add_GDAL_ds(self,md):
    #add fields from an input Metadata object
        self.nx = md.nx
        self.ny = md.ny
        self.x0 = md.x0
        self.y0 = md.y0
        self.dx = md.dx
        self.dy = md.dy
        self.srs = md.srs
        self.num_bands = md.num_bands
        self.no_data = md.no_data
        self.img_path = md.img_path
        self.img_file = md.img_file
        self.img_dir = md.img_dir
        return self   
    def print(self):
        print('SAR Metadata Object')
        print(f'\timg_path: {self.img_path}')
        print(f'\timg_dir: {self.img_path}')
        print(f'\timg_file: {self.img_file}')
        print(f'\tnx: {self.nx}')
        print(f'\tny: {self.ny}')
        print(f'\tx0: {self.x0}')
        print(f'\ty0: {self.y0}')
        print(f'\tdx: {self.dx}')
        print(f'\tdy: {self.dy}')
        print(f'\tsrs: {self.srs}')
        print(f'\tnum_bands: {self.num_bands}')
        print(f'\tno_data: {self.no_data}')
        print(f'\tinc: {self.inc}')
        print(f'\trad_scl: {self.rad_scl}')
        print(f'\tht: {self.ht}')
        print(f'\tpt_vec: {self.pt_vec}')
        
