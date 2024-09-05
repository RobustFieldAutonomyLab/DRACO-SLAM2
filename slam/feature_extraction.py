#!/usr/bin/env python
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import PointCloud2, Image
import cv_bridge
import ros_numpy
import yaml
from bruce_slam.utils.io import *
from bruce_slam.utils.topics import *
from bruce_slam.utils.conversions import *
from bruce_slam.utils.visualization import apply_custom_colormap
#from bruce_slam.feature import FeatureExtraction
from bruce_slam import pcl
import matplotlib.pyplot as plt
from sonar_oculus.msg import OculusPing, OculusPingUncompressed
from scipy.interpolate import interp1d
# from holoocean import packagemanager

from slam.utils import *
# from slam.sonar import *

from bruce_slam.CFAR import CFAR


#from bruce_slam.bruce_slam import sonar

class FeatureExtraction:
    '''Class to handle extracting features from Sonar images using CFAR
    subsribes to the sonar driver and publishes a point cloud
    '''

    def __init__(self, config_path):
        '''Class constructor, no args required all read from yaml file
        '''
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        #default parameters for CFAR
        self.Ntc = config['CFAR']['Ntc']
        self.Ngc = config['CFAR']['Ngc']
        self.Pfa = config['CFAR']['Pfa']
        self.rank = config['CFAR']['rank']
        self.alg = config['CFAR']['alg']
        self.detector = None
        self.threshold = config['filter']['threshold']
        # self.cimg = None

        #default parameters for point cloud 
        self.colormap = "RdBu_r"
        self.pub_rect = True
        self.resolution = 0.5
        self.outlier_filter_radius = config['filter']['radius']
        self.outlier_filter_min_points = config['filter']['min_points']
        self.skip = config['filter']['skip']

        # for offline visualization
        self.feature_img = None

        #for remapping from polar to cartisian
        self.res = config['filter']['resolution']
        self.height = None
        self.rows = None
        self.width = None
        self.cols = None
        self.map_x = None
        self.map_y = None
        self.f_bearings = None
        self.to_rad = lambda bearing: bearing * np.pi / 18000
        self.REVERSE_Z = 1
        self.maxRange = None

        #which vehicle is being used
        self.compressed_images = config['compressed_images']

        self.BridgeInstance = cv_bridge.CvBridge()

        # place holder for the multi-robot system
        self.rov_id = ""
        # packagemanager.installed_packages()
        # self.sonar_config = packagemanager.get_scenario("rfal_land_single_sonar")

        self.sonar_config = config['sonar_config']

        self.detector = CFAR(self.Ntc, self.Ngc, self.Pfa, None)
        self.convert = False

        # generate a mesh grid mapping from polar to cartisian
        self.generate_map(self.sonar_config)


    def generate_map(self, config: dict):
        """Generate a map to convert a holoocean sonar image from 
        polar to cartisian coords. Note this does not apply the remapping
        it generates the tools to apply cv2.remap later. 

        Args:
            config (dict): the holoocean config file. Read in using
            holoocean.packagemanager.get_scenario(scenario)

        """
        # parse out the parameters
        config = config['agents'][0]['sensors'][-1]["configuration"]
        horizontal_fov = float(config['Azimuth'])
        range_resolution = (config['RangeMax'] - config['RangeMin']) / 512
        height = range_resolution * config['RangeBins']
        rows = config['RangeBins']
        width = np.sin(np.radians(horizontal_fov / 2)) * height * 2
        cols = int(np.ceil(width / range_resolution))
        bearings = np.radians(np.linspace(-horizontal_fov / 2, horizontal_fov / 2, config['AzimuthBins']))

        self.height = height
        self.rows = rows
        self.width = width
        self.cols = cols

        # create an interpolation object for bearing angle
        f_bearings = interp1d(
            bearings,
            range(len(bearings)),
            kind='linear',
            bounds_error=False,
            fill_value=-1,
            assume_sorted=True)
        self.f_bearings = f_bearings

        #build the meshgrid
        XX, YY = np.meshgrid(range(cols), range(rows))
        x = range_resolution * (rows - YY)
        y = range_resolution * (-cols / 2.0 + XX + 0.5)
        b = np.arctan2(y, x) * -1
        r = np.sqrt(np.square(x) + np.square(y))
        map_y = np.asarray(r / range_resolution, dtype=np.float32)
        map_x = np.asarray(f_bearings(b), dtype=np.float32)

        self.map_y = map_y
        self.map_x = map_x

    #@add_lock
    def extract_features(self, image):
        # decode the compressed image
        img = image

        # Detect targets and check against threshold using CFAR (in polar coordinates)
        peaks = self.detector.detect(img, self.alg)
        peaks &= img > self.threshold

        vis_img = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)
        # convert from image coords to meters
        vis_img_copy = np.array(vis_img)
        vis_img_copy = cv2.applyColorMap(vis_img, 2)

        # convert the detected feature points into cartesian coordinate
        peaks = cv2.remap(peaks, self.map_x, self.map_y, cv2.INTER_LINEAR)
        locs = np.c_[np.nonzero(peaks)]
        for loc in locs:
            cv2.circle(vis_img_copy, (loc[1], loc[0]), 5, (0, 0, 255), -1)

        if self.convert:
            x = locs[:, 1] - self.cols / 2.
            x = (-1 * ((x / float(self.cols / 2.)) * (self.width / 2.)))  #+ self.width
            y = (-1 * (locs[:, 0] / float(self.rows)) * self.height) + self.height
            points = np.column_stack((y, x))
        else:
            points = locs

        # filter the cloud using PCL
        if len(points) and self.resolution > 0:
            points = pcl.downsample(points, self.resolution)

        # remove some outliers
        '''if self.outlier_filter_min_points > 1 and len(points) > 0:
            # points = pcl.density_filter(points, 5, self.min_density, 1000)
            points = pcl.remove_outlier(
                points, self.outlier_filter_radius, self.outlier_filter_min_points
            )'''

        return points, vis_img, vis_img_copy





