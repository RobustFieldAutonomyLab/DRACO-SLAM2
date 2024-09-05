import numpy as np
import cv2
import cv_bridge
from tensorflow import keras
from sklearn.cluster import DBSCAN
from scipy.interpolate import interp1d
from sensor_msgs.msg import Image, PointCloud2, PointField, CompressedImage
# bruce_slam
from bruce_slam.CFAR import CFAR
from bruce_slam import pcl
import matplotlib.pyplot as plt
from slam.feature_extraction import FeatureExtraction


class Keyframe():
    # class to be used as a sub class for 3D mapping

    def __init__(self, pose_, image_, fusedCloud_):
        self.pose = pose_
        self.image = image_
        self.segInfo = None
        self.segCloud = None
        self.ID = None
        self.width = float(image_.shape[1])
        self.height = float(image_.shape[0])
        self.maxRange = 30.
        self.FOV = 130.
        self.xRange = self.maxRange * np.cos(np.radians(90. - self.FOV / 2))
        self.fusedCloud = fusedCloud_
        self.fusedCloudDiscret = self.real2pix(fusedCloud_)
        self.matchImage = None
        self.rot = None
        self.rerun = True
        self.constructedCloud = None
        self.rerun = [True, True]
        self.containsPoints = False
        self.segcontainsPoints = False
        self.fusedCloudReg = None
        self.constructedCloudReg = None
        self.segCloudReg = None

    def real2pix(self, points):
        '''convert from meters to pixels
        '''
        x = (- self.width / 2 * (points[:, 1] / self.xRange)) + self.width / 2
        y = self.height - (self.height * (points[:, 0] / self.maxRange))

        return np.column_stack((x, y))


class ObjectDetection:
    def __init__(self, params):
        # set up cv_bridge
        self.CVbridge = cv_bridge.CvBridge()

        self.feature_extractor = FeatureExtraction(params["feature_extraction"])

        # pose history
        self.poses = None

        # build the sonar image mask
        self.blank = np.zeros((600, 1106))
        self.buildMask()

        # set up the CFAR detector
        self.detector = CFAR(20, 10, 0.5, None)
        self.thresholdCFAR = 35

        # define the classes
        self.classes = [0, 1]

        # define a container for the grid regressions
        self.grids = {0: None,
                      1: None}

        self.guassianGrids = {0: None,
                              1: None}

        # define grid res
        self.gridRes = .1

        # define the number of required keyframes to declare an object simple
        self.minFrames = 1

        # define some image parameters
        self.width = 1106.
        self.height = 600.
        self.maxRange = 30.
        self.FOV = 130.
        self.xRange = self.maxRange * np.cos(np.radians(90. - self.FOV / 2))

        # define some parameters for baysian update
        self.maxDepth = 6.
        self.minDepth = -2.
        self.baysianRes = 30 / 600.
        self.baysianBins = int((self.maxDepth - self.minDepth) / self.baysianRes)
        self.baysianX = np.linspace(self.minDepth, self.maxDepth, self.baysianBins)
        self.measSigma = .08

        # ICP for object registration
        self.icp = pcl.ICP()
        self.icp.loadFromYaml(params["object_icp"])

        # define laser fields for fused point cloud
        self.laserFields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        # define laser fields for segmented point cloud
        self.segFields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # for remapping from polar to cartisian
        self.res = self.feature_extractor.res
        self.height = self.feature_extractor.height
        self.rows = self.feature_extractor.rows
        self.width = self.feature_extractor.width
        self.cols = self.feature_extractor.cols
        self.REVERSE_Z = 1
        # self.maxRange = None
        self.predCount = 0

        # ablation study params, for logging purposes only
        self.scene = None
        self.keyframe_translation = None
        self.keyframe_rotation = None
        self.time_log = []
        self.vis_3D = True

        self.model = keras.models.load_model(params["model"])
        self.model.make_predict_function()

    def buildMask(self):
        # Build an image mask to determine if a point is outside the sonar block
        self.blank = cv2.circle(self.blank, (553, 600), 600, (255, 255, 255), -1)
        pts = np.array([(553, 600), (1106, 343), (1106, 600)], np.int32)
        pts = pts.reshape((-1, 1, 2))
        self.blank = cv2.fillPoly(self.blank, [pts], (0, 0, 0))
        pts = np.array([(553, 600), (0, 343), (0, 600)], np.int32)
        pts = pts.reshape((-1, 1, 2))
        self.blank = cv2.fillPoly(self.blank, [pts], (0, 0, 0))

    def getLeadingEdge(self, points):
        '''Takes a set of points in polar coords and returns the closest point in each scan line
            points: 2D points from sonar, must be integers (bin numbers)
            returns: the closest point in each scan line
        '''

        #init output points
        outputPoints = np.ones((1, 1))

        #get the possible bearings in the points
        bearings = list(set(points[:, 0]))

        #loop over the bearings and get the closest range return
        for b in bearings:

            #get the points from this scan line
            points_i = points[points[:, 0] == b]

            #get the min range
            r = np.min(points_i[:, 1])

            #log the points
            if outputPoints.shape == (1, 1):
                outputPoints = np.array([b, r])
            else:
                outputPoints = np.row_stack((outputPoints, np.array([b, r])))

        return outputPoints

    def guassianRegressObject(self, objectPoints, keyframe, box, classNum):
        '''Estimate object geometry using bayssian estimation
            keyframe: the frame to be analyzed, contains pointclouds from sensor fusion
            box: bounding box of object in qeuestion
            classNum: class label for object
            returns: nothing, updates class varibles
        '''

        #convert object points to range and bearing
        #objectX = (-1 * ((objectPoints[:,1] / self.width) * (self.xRange * 2.))) + self.xRange
        #objectY = (-1*(objectPoints[:,0] / self.height) * self.maxRange) + self.maxRange

        objectX = objectPoints[:, 1] - self.cols / 2.
        objectX = (-1 * ((objectX / float(self.cols / 2.)) * (self.width / 2.)))
        objectY = (-1 * (objectPoints[:, 0] / float(self.rows)) * self.height) + self.height

        rangeObject = np.sqrt(objectX ** 2 + objectY ** 2)
        bearingObject = np.degrees(np.arctan2(objectY, objectX))

        #convert to image local
        bearingObject = 256 * (bearingObject / 65.)
        bearingObject += 256
        bearingObject -= np.min(bearingObject)

        #convert spherical ranges to local image coordinates
        rangeObject = 600. * (rangeObject / 30.)
        rangeObject -= np.min(rangeObject)

        xMins = np.array([box[0][0], box[1][0]])
        yMins = np.array([box[0][1], box[1][1]])

        xMins = xMins - self.cols / 2.
        xMins = (-1 * ((xMins / float(self.cols / 2.)) * (self.width / 2.)))
        yMins = (-1 * (yMins / float(self.rows)) * self.height) + self.height

        minX = xMins[0]
        maxX = xMins[1]
        minY = yMins[0]
        maxY = yMins[1]

        '''#get the box corners
        minY = box[0][1]
        minX = box[0][0]
        maxY = box[1][1]
        maxX = box[1][0]

        #convert box corners to meters
        minX = (-1 * ((minX / self.width) * (self.xRange * 2.))) + self.xRange
        minY = (-1*(minY / self.height) * self.maxRange) + self.maxRange
        maxX = (-1 * ((maxX / self.width) * (self.xRange * 2.))) + self.xRange
        maxY = (-1*(maxY / self.height) * self.maxRange) + self.maxRange'''

        # pull the point cloud from the keyframe
        x = keyframe.fusedCloud[:, 0]
        z = keyframe.fusedCloud[:, 1]
        y = keyframe.fusedCloud[:, 1]

        # filter the 3D points based on the bounding box
        x = x[(keyframe.fusedCloud[:, 0] < minY) & (keyframe.fusedCloud[:, 0] > maxY) & (
                keyframe.fusedCloud[:, 2] < minX) & (keyframe.fusedCloud[:, 2] > maxX)]
        y = y[(keyframe.fusedCloud[:, 0] < minY) & (keyframe.fusedCloud[:, 0] > maxY) & (
                keyframe.fusedCloud[:, 2] < minX) & (keyframe.fusedCloud[:, 2] > maxX)]
        z = z[(keyframe.fusedCloud[:, 0] < minY) & (keyframe.fusedCloud[:, 0] > maxY) & (
                keyframe.fusedCloud[:, 2] < minX) & (keyframe.fusedCloud[:, 2] > maxX)]

        # Requires 3D points and 2D points
        if len(x) != 0 and len(bearingObject) != 0:

            # get the spherical range of each point
            rangeSpeherical = np.sqrt(x ** 2 + y ** 2 + z ** 2)

            # get the bearing of each point
            bearingPolar = np.degrees(np.arctan(y / x))

            # convert the bearings to local image coordinates
            bearingPolar = 256 * (bearingPolar / 65.)
            bearingPolar += 256
            bearingPolar -= np.min(bearingPolar)

            # convert spherical ranges to local image coordinates
            rangeSpeherical = 600. * (rangeSpeherical / 30.)
            rangeSpeherical -= np.min(rangeSpeherical)

            # check if we have not set up a grid before
            if self.guassianGrids.get(classNum) == None:

                # set up brand new grids
                gridProb = np.ones((int(np.max(rangeSpeherical)) + 1, int(np.max(bearingPolar)) + 1, 160)) / 160.
                gridImg = np.zeros((int(np.max(rangeSpeherical)) + 1, int(np.max(bearingPolar)) + 1))

                # get the first set of source points
                sourcePoints = np.column_stack((np.array(bearingObject).astype(int), np.array(rangeObject).astype(int)))
                sourcePoints = self.getLeadingEdge(sourcePoints)

            # if we have set up the grid before, pull the class objects
            else:

                # pull the grids
                gridProb = self.guassianGrids.get(classNum)[0]
                gridImg = self.guassianGrids.get(classNum)[1]

                # pull the source points
                sourcePoints = self.guassianGrids.get(classNum)[2]

                # set up target points, the newest set of observations
                targetPoints = np.column_stack((np.array(bearingObject).astype(int), np.array(rangeObject).astype(int)))
                targetPoints = self.getLeadingEdge(targetPoints)

                # compute ICP
                icpRes = self.icp.compute(targetPoints, sourcePoints, np.identity(3))
                icpStatus = icpRes[0]
                icpRes = icpRes[1]

                # call out the rotation matrix
                Rmtx = np.array([[icpRes[0][0], icpRes[0][1]],
                                 [icpRes[1][0], icpRes[1][1]]])

                # get the rotation angle
                #eulerAngle = np.degrees(np.arctan(icpRes[0][0] / icpRes[1][0]))

                # register the new observations
                targetPoints = targetPoints.dot(Rmtx.T)
                targetPoints += np.array([icpRes[0][2], icpRes[1][2]])

                # concanate the point clouds for future timesteps
                sourcePoints = np.row_stack((sourcePoints, targetPoints))

                # transform the 3D measurnments
                targetPoints = np.column_stack((bearingPolar, rangeSpeherical)).dot(Rmtx.T) + np.array(
                    [icpRes[0][2], icpRes[1][2]])
                bearingPolar = targetPoints[:, 0]
                rangeSpeherical = targetPoints[:, 1]

                # grow the grids as required

                # if the data has more range than the grid
                if gridProb.shape[0] <= np.max(rangeSpeherical):
                    # grow the grid image
                    growth = np.zeros((int(np.max(rangeSpeherical)) + 1, gridProb.shape[1]))
                    gridImg = np.row_stack((gridImg, growth))

                    # grow the probability grid
                    growth = np.ones((int(np.max(rangeSpeherical)) + 1, gridProb.shape[1], 160)) / 160.
                    gridProb = np.row_stack((gridProb, growth))

                # if the data has a lower range than the grid
                '''if int(np.min(rangeSpeherical)) < 0:

                    #grow the grid image
                    growth = np.zeros(( int(abs(np.min(rangeSpeherical))), gridProb.shape[1]))
                    gridImg = np.row_stack((growth, gridImg))

                    #grow the probability grid
                    growth = np.ones(( int(abs(np.min(rangeSpeherical))), gridProb.shape[1], 160)) / 160.
                    gridProb = np.row_stack((growth, gridProb))

                    #shift the range observations as needed
                    rangeSpeherical -= growth.shape[0]'''

                # if the data has more bearing than the grid
                if gridProb.shape[1] <= np.max(bearingPolar):
                    # grow the grid image
                    growth = np.zeros((gridProb.shape[0], int(np.max(bearingPolar)) + 1))
                    gridImg = np.column_stack((gridImg, growth))

                    # grow the probability grid
                    growth = np.ones((gridProb.shape[0], int(np.max(bearingPolar)) + 1, 160)) / 160.
                    gridProb = np.column_stack((gridProb, growth))

                # if the data has a lower bearing than the grid
                '''if int(np.min(bearingPolar)) < 0:

                    #grow the grid image
                    growth = np.zeros(( gridProb.shape[0], abs(int(np.min(bearingPolar))) ))
                    gridImg = np.column_stack((growth, gridImg))

                    #grow the probability grid
                    growth = np.ones(( gridProb.shape[0], abs(int(np.min(bearingPolar))), 160)) / 160.
                    gridProb = np.column_stack(( growth, gridProb))

                    #shift the bearings as needed
                    #bearingPolar -= abs(int(np.min(bearingPolar)))'''

            # loop over the bearings
            for bearing in list(set(bearingPolar)):

                # get the measurnments from this scan line
                z_at_bearing = z[bearingPolar == bearing]
                range_at_bearing = rangeSpeherical[bearingPolar == bearing]
                bearing_at_bearing = bearingPolar[bearingPolar == bearing]

                # loop over ranges
                for range_i in list(set(range_at_bearing)):

                    # get the measurnments for this range bin
                    z_at_range_bearing = z_at_bearing[range_at_bearing == range_i]
                    bearing_at_range_bearing = bearing_at_bearing[range_at_bearing == range_i]
                    range_at_range_bearing = range_at_bearing[range_at_bearing == range_i]

                    # dummy varible
                    p_m_z = np.ones((1, 1))

                    # loop over all the z values here to make a max mixture
                    for zVal in z_at_range_bearing:

                        # generate the prob distrubtion for this single measurnment
                        p_m_z_i = np.exp(- np.power(self.baysianX - zVal, 2.) / (2 * np.power(self.measSigma, 2.)))

                        # if more than one measurnement at this bin record that
                        if p_m_z.shape == (1, 1):
                            p_m_z = p_m_z_i
                        else:
                            p_m_z = np.column_stack((p_m_z, p_m_z_i))

                    # if there was more than one measunment take the max at that bin, a max mixture
                    if p_m_z.shape != (160,):
                        p_m_z = np.max(p_m_z, axis=1)

                    if int(range_at_range_bearing[0]) >= 0:
                        # pull out the prior distrubtuion
                        prior = gridProb[int(range_at_range_bearing[0])][int(bearing_at_range_bearing[0])]

                        # get the post distrubtion
                        posterior = p_m_z * prior
                        posterior /= np.sum(posterior)

                        # update the grids
                        gridProb[int(range_at_range_bearing[0])][int(bearing_at_range_bearing[0])] = posterior
                        gridImg[int(range_at_range_bearing[0])][int(bearing_at_range_bearing[0])] = 255

            # push the new grids
            self.guassianGrids[classNum] = [gridProb, gridImg, sourcePoints]

    def segmentImage(self, keyframe):
        '''Segment a sonar image into classes
            sonarImg: greyscale sonar image from a keyframe
            matchImg: matches from orthoganal fusion as a sonar elevation image (float64)
        '''

        # get some info from the keyframe
        sonarImg = keyframe.image
        # get the CFAR points
        points, sonarImg, visualize_img = self.feature_extractor.extract_features(sonarImg)

        # output container
        segPoints = np.array([0, 0])
        segColors = [0]

        # outputs for downstream code
        boundingBoxes = []
        pointsBoxes = []
        probs = []

        # protect for an empty frame
        if len(points) > 0:

            # cluster the CFAR points
            clustering = DBSCAN(eps=5, min_samples=2).fit(points)
            labels = clustering.labels_

            # loop over the labels from the clusters
            for label in list(set(labels)):

                self.predCount += 1
                # print self.predCount

                # if this label is not noise
                if label != -1:

                    # get the points in this cluster
                    pointsSubset = points[labels == label]
                    pointsSubset = np.array(pointsSubset, dtype=np.int32)
                    # get a bounding box
                    [x, y, w, h] = cv2.boundingRect(pointsSubset)

                    # convert the bounding box to tuples
                    refPt = [(y - 5, x - 5), (y + h + 5, x + w + 5)]

                    # get the query image
                    roi = sonarImg[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

                    # resize and package
                    roi = cv2.resize(roi, (40, 40))
                    roi = np.array(roi).astype('uint8')

                    # Segment image for network query
                    blur = cv2.GaussianBlur(roi, (15, 15), 0)
                    ret3, roi = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # normalize image
                    roi = roi / 255.

                    # check the blank mask
                    mask = self.blank[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

                    # if there are more than 30 CFAR points and the bounding box is not near an
                    # print("pointsSubset: ", len(pointsSubset))
                    if len(pointsSubset) > 5:

                        # copy the image 10 times
                        queries = []
                        for j in range(20):
                            queries.append(roi)

                        # package and predict
                        queries = np.expand_dims(queries, axis=3)
                        predictions = self.model.predict(queries)

                        # get the mean and variance of the predictions
                        avg = np.mean(predictions, axis=0)
                        var = np.var(predictions, axis=0)

                        # print avg
                        # print var

                        # if the confidence in the prediction is below 99% do nothing
                        if np.max(avg) > .99:

                            # record the points
                            segPoints = np.row_stack((segPoints, pointsSubset))

                            # get the class
                            pred = np.argmax(avg)

                            # color the points
                            if pred == 0:

                                # log colors for cloud
                                segColors += list(100 * np.ones(len(pointsSubset)))

                                # update class zero
                                # self.guassianRegressObject(pointsSubset, keyframe, refPt, 0)

                                # log for downstream code
                                boundingBoxes.append(refPt)
                                pointsBoxes.append(pointsSubset)
                                probs.append(0)

                                # plt.scatter(pointsSubset[:,1], pointsSubset[:,0], c = "r")

                            elif pred == 1:

                                # log colors for cloud
                                segColors += list(255 * np.ones(len(pointsSubset)))

                                # update class 1
                                #self.guassianRegressObject( pointsSubset, keyframe, refPt, 1)

                                # log for downstream code
                                boundingBoxes.append(refPt)
                                pointsBoxes.append(pointsSubset)
                                probs.append(1)

                                # plt.scatter(pointsSubset[:,1], pointsSubset[:,0], c = "b")

                            else:

                                # log colors for cloud
                                segColors += list(0 * np.ones(len(pointsSubset)))
                                boundingBoxes.append(refPt)
                                pointsBoxes.append(pointsSubset)
                                probs.append(2)

                                # plt.scatter(pointsSubset[:,1], pointsSubset[:,0], c = "g")

                    else:

                        # record the points
                        segPoints = np.row_stack((segPoints, pointsSubset))
                        segColors += list(0 * np.ones(len(pointsSubset)))

        dict_result = {'image': visualize_img,
                       'segPoints': segPoints,
                       'segColors': segColors,
                       'pose': keyframe.pose}
        if probs != []:
            new_boxes = []
            for box0, box1 in boundingBoxes:
                new_boxes.append(([box0[1], box0[0]], [box1[1], box1[0]]))
            dict_result['boundingBoxes'] = new_boxes
            dict_result['pointsBoxes'] = pointsBoxes
            dict_result['probs'] = probs
            dict_result['detected'] = True
        else:
            dict_result['detected'] = False

        return dict_result
