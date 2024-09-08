import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from slam.object_detection import *


# Everything related to the object map
def transform_points(points, pose):
    # transform the points to the global frame
    R = np.array([[np.cos(pose[2]), -np.sin(pose[2])],
                  [np.sin(pose[2]), np.cos(pose[2])]])
    T = pose[:2]
    return np.dot(R, points.T).T + T


def transform_point(point, pose):
    # transform the points to the global frame
    R = np.array([[np.cos(pose[2]), -np.sin(pose[2])],
                  [np.sin(pose[2]), np.cos(pose[2])]])
    T = pose[:2]
    return np.dot(R, point) + T


class Object:
    def __init__(self, points, pose, bounding_box, obj_type):
        # list[ndarray]:untransformed points of the object
        self.id = -1
        self.points_raw = [points]
        # list[ndarray]: poses where the object is detected
        self.pose = [pose]
        # list[ndarray]: untransformed bounding box of the object
        self.bounding_box_raw = [np.array([bounding_box[0, :],
                                           [bounding_box[1, 0], bounding_box[0, 1]],
                                           bounding_box[1, :],
                                           [bounding_box[0, 0], bounding_box[1, 1]],
                                           bounding_box[0, :]])]
        # int: object type
        self.object_type = obj_type
        # ndarray: transformed points of the object
        self.points_transformed = transform_points(points, pose)
        # ndarray: transformed bounding box of the object
        self.bounding_box_transformed = transform_points(self.bounding_box_raw[0], pose)
        # ndarray: center of the object
        self.center = np.mean(self.points_transformed, axis=0)

    def __lt__(self, other: 'Object'):
        return self.id < other.id

    def compare_points(self, other: 'Object'):
        # Build KD-Trees for fast nearest-neighbor search
        tree1 = cKDTree(self.points_transformed)
        tree2 = cKDTree(other.points_transformed)

        # Compute the minimum distance from each point in cloud1 to the nearest point in cloud2
        min_dist1, _ = tree1.query(other.points_transformed)

        # Compute the minimum distance from each point in cloud2 to the nearest point in cloud1
        min_dist2, _ = tree2.query(self.points_transformed)

        # Return the smallest distance from either query
        return min(np.min(min_dist1), np.min(min_dist2))

    def compare_bounding_boxes(self, other: 'Object'):
        pass

    def update(self, other: 'Object'):
        self.points_raw = self.points_raw + other.points_raw
        self.pose = self.pose + other.pose
        self.bounding_box_raw = self.bounding_box_raw + other.bounding_box_raw

        self.points_transformed = np.concatenate([self.points_transformed, other.points_transformed], axis=0)
        bounding_box = np.array([np.min(self.points_transformed, axis=0),
                                 np.max(self.points_transformed, axis=0)])
        self.bounding_box_transformed = np.array([bounding_box[0, :],
                                                  [bounding_box[1, 0], bounding_box[0, 1]],
                                                  bounding_box[1, :],
                                                  [bounding_box[0, 0], bounding_box[1, 1]],
                                                  bounding_box[0, :]])
        self.center = np.mean(self.points_transformed, axis=0)


def calculate_center_distance(object0: Object, object1: Object):
    # TODO: maybe try using distance between clouds?
    return np.linalg.norm(object0.center - object1.center)


class ObjectMapper:
    def __init__(self, robot_ns, config):
        self.robot_ns = robot_ns
        self.config = config
        self.object_detector = ObjectDetection(config)

        self.cols = self.object_detector.feature_extractor.cols
        self.rows = self.object_detector.feature_extractor.rows
        self.width = self.object_detector.feature_extractor.width
        self.height = self.object_detector.feature_extractor.height

        self.objects = {}
        self.edges = []
        self.points = np.empty((0, 2))
        self.poses = np.empty((0, 3))
        self.object_counter = 0

        # graph construction and matching parameters
        with open(config["graph_matching"], 'r') as file:
            config_graph = yaml.safe_load(file)

        self.max_edge_distance = config_graph["edge"]["max_distance"]
        self.min_node_distance = config_graph["node"]["min_distance_accept"]

    def add_object(self, keyframe: Keyframe):
        dict_result = self.object_detector.segmentImage(keyframe)

        self.poses = np.concatenate([self.poses, np.array([keyframe.pose])], axis=0)
        try:
            points = self.pixel2meter(dict_result['segPoints'])
            self.points = np.concatenate([self.points,
                                          transform_points(points, keyframe.pose)],
                                         axis=0)
        except:
            print(dict_result['segPoints'])

        if dict_result['detected']:
            # add objects into map
            for i, bounding_box in enumerate(dict_result['boundingBoxes']):
                obj_new = Object(self.pixel2meter(dict_result['pointsBoxes'][i]),
                                 keyframe.pose,
                                 self.pixel2meter(np.array(bounding_box)),
                                 dict_result['probs'][i])
                merged = False
                for obj_old in self.objects.values():
                    if obj_old.compare_points(obj_new) < self.min_node_distance:
                        obj_old.update(obj_new)
                        merged = True
                        break
                if not merged:
                    obj_new.id = self.object_counter
                    self.objects[self.object_counter] = obj_new
                    self.object_counter += 1
                    self.reconstruct_edges()
                    print(self.edges)
        self.plot_figure()

    def pixel2meter(self, locs):
        x = locs[:, 1] - self.cols / 2.
        x = (-1 * ((x / float(self.cols / 2.)) * (self.width / 2.)))  #+ self.width
        y = (-1 * (locs[:, 0] / float(self.rows)) * self.height) + self.height
        points = np.column_stack((y, -x))
        return points

    def reconstruct_edges(self):
        self.edges.clear()
        nodes_list = list(self.objects.values())
        for i, node0 in enumerate(nodes_list):
            for j in range(i + 1, len(nodes_list)):
                node1 = nodes_list[j]
                dist = calculate_center_distance(node0, node1)
                if dist < self.max_edge_distance:
                    self.edges.append((node0.id, node1.id, dist))

    def plot_figure(self):

        plt.figure(figsize=(8, 8), dpi=100)
        plt.scatter(self.points[:, 0], self.points[:, 1], s=1, c='k')
        plt.plot(self.poses[:, 0], self.poses[:, 1], 'go')
        # cloud = keyframe.fusedCloud
        #
        # plt.scatter(cloud[:, 0], cloud[:, 1], s=1, c='g')

        for obj in self.objects.values():
            plt.plot(obj.points_transformed[:, 0], obj.points_transformed[:, 1], 'lightblue')
            if obj.object_type == 0:
                color = 'r'
            elif obj.object_type == 1:
                color = 'b'
            else:
                color = 'g'
            plt.plot(obj.bounding_box_transformed[:, 0],
                     obj.bounding_box_transformed[:, 1],
                     color)

        plt.xlim([-80, 80])
        plt.ylim([-80, 80])
        plt.savefig('test_data/' + str(self.robot_ns) + '/' + str(self.poses.shape[0]) + '.png')
        plt.close()
