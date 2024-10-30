import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from slam.object_detection import *
from scipy.optimize import linear_sum_assignment
from slam.feature_extraction import FeatureExtraction
import cv2
import time
from scipy.sparse.linalg import eigs
import copy


# Everything related to the object map
def transform_points(points, pose):
    # transform the points to the global frame
    if pose.ndim == 1:
        R = np.array([[np.cos(pose[2]), -np.sin(pose[2])],
                      [np.sin(pose[2]), np.cos(pose[2])]])
        T = pose[:2]
    else:
        R = pose[:2, :2]
        T = pose[:2, 2]
    return np.dot(R, points.T).T + T


def state_to_matrix(state):
    return np.array([[np.cos(state[2]), -np.sin(state[2]), state[0]],
                     [np.sin(state[2]), np.cos(state[2]), state[1]],
                     [0, 0, 1]])


def matrix_to_state(transformation):
    rad = np.arctan2(transformation[1, 0], transformation[0, 0])
    return np.array([transformation[0, 2], transformation[1, 2], rad])


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
        self.keys = set()
        self.pose = [pose]
        self.dimension = (1000, 1000)
        # list[ndarray]: untransformed bounding box of the object
        if len(bounding_box) == 2:
            self.bounding_box_raw = [np.array([bounding_box[0, :],
                                               [bounding_box[1, 0], bounding_box[0, 1]],
                                               bounding_box[1, :],
                                               [bounding_box[0, 0], bounding_box[1, 1]],
                                               bounding_box[0, :]])]
        else:
            self.bounding_box_raw = [np.concatenate((bounding_box, bounding_box[0:1]), axis=0)]
        # int: object type
        self.object_type = obj_type
        # ndarray: transformed points of the object
        self.points_transformed = transform_points(points, pose)
        # ndarray: transformed bounding box of the object
        self.bounding_box_transformed = transform_points(self.bounding_box_raw[0], pose)
        # ndarray: center of the object
        self.center = np.mean(self.points_transformed, axis=0)
        # for logging graph matching results
        self.associated_object_id = -1

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

    def compare_object_type(self, other: 'Object'):
        if self.object_type == other.object_type:
            return True
        else:
            return False

    def compare_object_shape(self, other: 'Object', mu_shape):
        return np.exp(-mu_shape * (abs(max(self.dimension) - max(other.dimension))
                                   + abs(min(self.dimension) - min(other.dimension))))

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


def get_principal_eigen_vector(mat):
    # Compute eigenvalues and eigenvectors
    # eigenvalues, eigenvectors = np.linalg.eig(mat)
    #
    # # Find the index of the largest eigenvalue
    # principal_eigenvalue_index = np.argmax(eigenvalues)
    #
    # # Get the principal eigenvector
    # principal_eigenvector = eigenvectors[:, principal_eigenvalue_index]
    # return np.abs(principal_eigenvector)

    eigenvalue, eigenvector = eigs(mat, k=1, which='LM')  # 'LM' means Largest Magnitude
    return np.abs(eigenvector[:, 0])


def calculate_bounding_box(points):
    # min_x, min_y = np.min(points, axis=0)
    # max_x, max_y = np.max(points, axis=0)

    # Compute the minimum area bounding rectangle
    points = np.array(points, dtype=np.float32)
    rect = cv2.minAreaRect(points)

    # Get the rectangle's corner points
    box = cv2.boxPoints(rect)

    return box, rect[1]


class ObjectMapper:
    def __init__(self, robot_ns, config):
        self.robot_ns = robot_ns
        self.config = config
        self.feature_extractor = FeatureExtraction(config["feature_extraction"])

        self.cols = self.feature_extractor.cols
        self.rows = self.feature_extractor.rows
        self.width = self.feature_extractor.width
        self.height = self.feature_extractor.height

        self.objects = {}
        self.edges = []
        self.points = np.empty((0, 2))
        self.ids = np.empty((0, 1), dtype=int)
        self.poses = np.empty((0, 3))
        self.object_counter = 0

        self.graphs_neighbor = {}
        self.transformations_neighbor = {}

        # graph construction and matching parameters
        with open(config["graph_matching"], 'r') as file:
            config_graph = yaml.safe_load(file)

        # parameters for graph matching
        self.object_detection_method = "DBSCAN"
        self.max_edge_distance = config_graph["edge"]["max_distance"]
        self.min_num_edge = config_graph["edge"]["min_number"]

        self.min_node_distance = config_graph["node"]["min_distance_accept"]
        self.min_num_node = config_graph["node"]["min_number"]
        self.dbscan_eps = config_graph["node"]["dbscan_eps"]
        self.dbscan_min_sample = config_graph["node"]["dbscan_min_sample"]

        self.match_use_type = config_graph["matching"]["use_type"]
        self.match_use_shape = config_graph["matching"]["use_shape"]
        self.mu = config_graph["matching"]["mu"]
        self.mu_shape = config_graph["matching"]["mu_shape"]
        self.min_eigenvector_accept = config_graph["matching"]["min_eigenvector_accept"]
        self.min_matched_nodes = config_graph["matching"]["min_number"]
        self.min_inliers = config_graph["matching"]["min_inliers"]

        if self.object_detection_method == "Gaussian":
            self.object_detector = ObjectDetection(config)

    def add_object(self, keyframe: Keyframe):
        if self.object_detection_method == "Gaussian":
            self.add_object_Gaussian(keyframe)
        elif self.object_detection_method == "DBSCAN":
            self.add_object_dbscan(keyframe)

    def add_object_dbscan(self, keyframe: Keyframe):
        if keyframe.image is not None:
            points, sonarImg, visualize_img = self.feature_extractor.extract_features(keyframe.image)
            points = self.pixel2meter(points)
        else:
            points = keyframe.fusedCloud
        ids = np.full((points.shape[0], 1), keyframe.ID)
        self.poses = np.concatenate([self.poses, np.array([keyframe.pose])], axis=0)
        self.points = np.concatenate([self.points, transform_points(points, keyframe.pose)], axis=0)
        self.ids = np.concatenate([self.ids, ids], axis=0)

        # dbscan = DBSCAN(eps=1.2, min_samples=10)
        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_sample)
        labels = dbscan.fit_predict(self.points)

        unique_labels = set(labels)  # Get unique cluster labels (-1 for noise)

        self.objects = {}
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = self.points[labels == label]  # Points belonging to a cluster
            box, rect = calculate_bounding_box(cluster_points)
            obj = Object(cluster_points,
                         np.array([0, 0, 0]),
                         box,
                         1)
            obj.keys = set(self.ids[labels == label].flatten())
            ratio = max(rect) / (min(rect) + 0.01)
            obj.dimension = rect
            if ratio > 4 and max(rect) > 5:
                obj.object_type = 1
            else:
                obj.object_type = 2
            self.objects[label] = obj
            self.objects[label].id = label
        self.reconstruct_edges()

    def add_object_Gaussian(self, keyframe: Keyframe):
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
                    # print(self.edges)

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

    def compare_all_neighbor_graph(self):
        self_dict = {}
        request_dict = {}
        for robot_ns_neighbor, robot_graph_neighbor in self.graphs_neighbor.items():
            time0 = time.time()
            # ids_pair: each row: 0: self object id; 1: neighbor object id
            # points_pair: each row: 0,1: x & y from self graph; 2 & 3: x & y from neighbor graph
            # step 1: compare two graphs
            matched, ids_pair, points_pair = self.compare_graphs(
                (self.objects, self.edges), robot_graph_neighbor)
            if not matched or points_pair.shape[0] < self.min_matched_nodes:
                continue
            # step 2: calculate relative transformation
            transformation_estimation_success, transformation, object_inlier_ids = (
                self.calculate_relative_transformation(ids_pair, points_pair))
            if not transformation_estimation_success:
                continue
            self.transformations_neighbor[robot_ns_neighbor] = transformation
            # step 3: generate request keyframe from neighbor robot
            request_dict[robot_ns_neighbor] = self.get_request_keyframe_id(robot_graph_neighbor[0], ids_pair[:, 1])
            self_dict[robot_ns_neighbor] = self.get_request_keyframe_id(self.objects, ids_pair[:, 0])
            time1 = time.time()
            print(f"Compare one graph time: {time1 - time0:.3f}")
        return self_dict, request_dict

    def get_request_keyframe_id(self, robot_nodes, object_ids):
        # request keyframe from neighbor robot
        keyframe_id_to_request = set()
        for object_id in object_ids:
            object_this = robot_nodes[object_id]
            keyframe_id_to_request = keyframe_id_to_request.union(object_this.keys)
            # request keyframe from neighbor robot
            # object.request_keyframe_from_neighbor()
        return keyframe_id_to_request

    def calculate_relative_transformation(self, ids_pair: np.ndarray, points_pair: np.ndarray):
        self_pts = np.array(points_pair[:, :2], dtype=np.float32)
        neighbor_pts = np.array(points_pair[:, 2:], dtype=np.float32)
        affine_matrix, inliers = cv2.estimateAffinePartial2D(neighbor_pts,
                                                             self_pts,
                                                             method=cv2.RANSAC,
                                                             ransacReprojThreshold=5.0)
        if inliers.sum() < self.min_inliers:
            return False, np.eye(2), np.zeros([2])

        R = affine_matrix[:, :2]
        scale = np.linalg.norm(R[:, 0])  # Get scale factor from the first column
        if scale != 0:
            R /= scale  # Normalize to get rid of scale
        t = affine_matrix[:, 2]
        return True, (R, t), ids_pair[inliers.flatten().astype(dtype=bool)]

    def compare_graphs(self, graph_self: tuple, graph_neighbor: tuple):
        nodes_self, edges_self = graph_self
        nodes_neighbor, edges_neighbor = graph_neighbor
        if len(nodes_self) < self.min_num_node or len(nodes_neighbor) < self.min_num_node:
            return False, None, None
        if len(edges_self) < self.min_num_edge or len(edges_neighbor) < self.min_num_edge:
            return False, None, None
        nodes_self_id_sorted = sorted(nodes_self.keys())
        nodes_neighbor_id_sorted = sorted(nodes_neighbor.keys())
        N = len(nodes_neighbor)
        M = len(nodes_self)

        # perform graph matching
        # construct the edge weight assignment matrix
        S = self.construct_edge_similarity_matrix_batch(graph_self=graph_self, graph_neighbor=graph_neighbor)
        # S = self.construct_edge_similarity_matrix(graph_self=graph_self, graph_neighbor=graph_neighbor)
        # solve NP-hard question using principal eigen vector of edge_similarity_mat S
        A_star = get_principal_eigen_vector(S)
        A_star_matrix = A_star.reshape(N, M)
        node_matching = self.linear_assignment(A_star_matrix)
        if len(node_matching) == 0:
            return False, None, None
        ids_pair = np.zeros([len(node_matching), 2], dtype=int)
        points_pair = np.zeros([len(node_matching), 4])
        for i, (node_idx_self, node_idx_neighbor) in enumerate(node_matching.items()):
            node_id_neighbor = nodes_neighbor_id_sorted[node_idx_neighbor]
            graph_neighbor[0][node_id_neighbor].associated_object_id = nodes_self_id_sorted[node_idx_self]
            ids_pair[i, :] = [nodes_self_id_sorted[node_idx_self], node_id_neighbor]
            points_pair[i, :2] = nodes_self[nodes_self_id_sorted[node_idx_self]].center
            points_pair[i, 2:] = nodes_neighbor[node_id_neighbor].center

        # first node from self, then node from neighbor
        return True, ids_pair, points_pair

    def linear_assignment(self, L: np.ndarray):
        # solve max_A\sum_{j=1}^N\sum_{k=1}^M trace(-A^TL)
        # with A(j,k)\in {0,1}
        # \sum^N_{j=1}A(j,k)<=1
        # \sum^M_{k=1} A(j,k) <=1
        # return the corresponding assignment for each index in the matrix
        node_indices_neighbor, node_indices_self = linear_sum_assignment(-L)
        assignment_result = {}
        for i, node_idx_self in enumerate(node_indices_self):
            node_idx_neighbor = node_indices_neighbor[i]
            if L[node_idx_neighbor, node_idx_self] >= self.min_eigenvector_accept:
                assignment_result[node_idx_self] = node_idx_neighbor
        # print("node_indices_neighbor", node_indices_neighbor)
        # print("node_indices_self", node_indices_self)
        return assignment_result

    def construct_edge_similarity_matrix_batch(self, graph_self: tuple, graph_neighbor: tuple):
        # Accelerate the process by using numpy array operations
        time0 = time.time()
        nodes_self, edges_self = graph_self
        nodes_neighbor, edges_neighbor = graph_neighbor

        nodes_self_id_sorted = np.array(sorted(nodes_self.keys()))
        nodes_neighbor_id_sorted = np.array(sorted(nodes_neighbor.keys()))

        size_s = len(nodes_self) * len(nodes_neighbor)

        # edge utility
        # row: number of edges self, col: number of edges neighbor
        e_self_mat = np.full((len(nodes_self), len(nodes_self)), np.nan)
        e_neighbor_mat = np.full((len(nodes_neighbor), len(nodes_neighbor)), np.nan)

        # Create index arrays using searchsorted for better performance
        j0_list, j1_list, ej01_list = zip(*edges_self)  # size m*m
        j0_indices = np.searchsorted(nodes_self_id_sorted, j0_list)
        j1_indices = np.searchsorted(nodes_self_id_sorted, j1_list)

        # Assign the corresponding ej01 values to the e_self_mat matrix using advanced indexing
        e_self_mat[j0_indices, j1_indices] = ej01_list
        e_self_mat[j1_indices, j0_indices] = ej01_list

        # Process edges_neighbor similarly
        j0_list, j1_list, ej01_list = zip(*edges_neighbor)  # size n*n
        j0_indices = np.searchsorted(nodes_neighbor_id_sorted, j0_list)
        j1_indices = np.searchsorted(nodes_neighbor_id_sorted, j1_list)

        e_neighbor_mat[j0_indices, j1_indices] = ej01_list
        e_neighbor_mat[j1_indices, j0_indices] = ej01_list

        # Calculate e_s_mat
        e_s_mat = -self.mu * np.abs(e_self_mat[np.newaxis, :, np.newaxis, :] -
                                    e_neighbor_mat[:, np.newaxis, :, np.newaxis])

        # Replace NaNs with 0 and apply exp only on non-zero values
        e_s_mat = np.nan_to_num(e_s_mat, nan=0)
        e_s_mat[e_s_mat != 0] = np.exp(e_s_mat[e_s_mat != 0])
        e_s_mat = np.reshape(e_s_mat, (size_s, size_s))

        # Node utility
        if self.match_use_type:
            d_self_mat = np.full((1, len(nodes_self)), np.nan)
            d_neighbor_mat = np.full((len(nodes_neighbor), 1), np.nan)
            for i, node_id in enumerate(nodes_self_id_sorted):
                d_self_mat[0, i] = nodes_self[node_id].object_type
            for i, node_id in enumerate(nodes_neighbor_id_sorted):
                d_neighbor_mat[i, 0] = nodes_neighbor[node_id].object_type
            d_t_mat = d_self_mat == d_neighbor_mat
            d_t_mat = np.array([d_t_mat.flatten()])
            d_t_mat = d_t_mat * d_t_mat.T
            e_s_mat = e_s_mat * d_t_mat
        if self.match_use_shape:
            d_self_mat = np.full((2, 1, len(nodes_self)), np.nan)
            d_neighbor_mat = np.full((2, len(nodes_neighbor), 1), np.nan)

            for i, node_id in enumerate(nodes_self_id_sorted):
                d_self_mat[0, 0, i] = max(nodes_self[node_id].dimension)
                d_self_mat[1, 0, i] = min(nodes_self[node_id].dimension)
            for i, node_id in enumerate(nodes_neighbor_id_sorted):
                d_neighbor_mat[0, i, 0] = max(nodes_neighbor[node_id].dimension)
                d_neighbor_mat[1, i, 0] = min(nodes_neighbor[node_id].dimension)
            d_t_mat = np.exp(-self.mu_shape * np.abs(d_self_mat - d_neighbor_mat))
            d_t_mat = d_t_mat[0, :, :] * d_t_mat[1, :, :]
            d_t_mat = np.array([d_t_mat.flatten()])
            d_t_mat = d_t_mat * d_t_mat.T
            e_s_mat = e_s_mat * d_t_mat

        time1 = time.time()
        print("time calculate batch: ", time1 - time0)

        return e_s_mat

    def construct_edge_similarity_matrix(self, graph_self: tuple, graph_neighbor: tuple):
        nodes_self, edges_self = graph_self
        nodes_neighbor, edges_neighbor = graph_neighbor
        nodes_self_id_sorted = sorted(nodes_self.keys())
        nodes_neighbor_id_sorted = sorted(nodes_neighbor.keys())
        size_s = len(nodes_self) * len(nodes_neighbor)
        node_self_index = {node_id: idx for idx, node_id in enumerate(nodes_self_id_sorted)}
        node_neighbor_index = {node_id: idx for idx, node_id in enumerate(nodes_neighbor_id_sorted)}

        # row: number of edges self, col: number of edges neighbor
        e_s_mat = np.zeros([size_s, size_s])
        for j0, j1, ej01 in edges_neighbor:
            for k0, k1, ek01 in edges_self:
                # if not using object type for edge matching
                d_e_jk = np.exp(-self.mu * abs(ej01 - ek01))
                # d_e_jk = ej01 - ek01

                case_1 = True
                case_2 = True
                if self.match_use_type:
                    case_1 = (nodes_neighbor[j0].compare_object_type(nodes_self[k0])
                              and nodes_neighbor[j1].compare_object_type(nodes_self[k1]))
                    case_2 = (nodes_neighbor[j1].compare_object_type(nodes_self[k0])
                              and nodes_neighbor[j0].compare_object_type(nodes_self[k1]))
                if self.match_use_shape:
                    d_e_jk1 = (d_e_jk * nodes_neighbor[j0].compare_object_shape(nodes_self[k0], self.mu_shape)
                               * nodes_neighbor[j1].compare_object_shape(nodes_self[k1], self.mu_shape))
                    d_e_jk2 = (d_e_jk * nodes_neighbor[j1].compare_object_shape(nodes_self[k0], self.mu_shape)
                               * nodes_neighbor[j0].compare_object_shape(nodes_self[k1], self.mu_shape))
                else:
                    d_e_jk1 = d_e_jk
                    d_e_jk2 = d_e_jk

                if case_1:
                    # case 1
                    s0 = node_neighbor_index[j0] * len(nodes_self) + node_self_index[k0]
                    s1 = node_neighbor_index[j1] * len(nodes_self) + node_self_index[k1]
                    e_s_mat[s0, s1] = d_e_jk1
                    e_s_mat[s1, s0] = d_e_jk1
                if case_2:
                    # case 2
                    s0 = node_neighbor_index[j0] * len(nodes_self) + node_self_index[k1]
                    s1 = node_neighbor_index[j1] * len(nodes_self) + node_self_index[k0]
                    e_s_mat[s0, s1] = d_e_jk2
                    e_s_mat[s1, s0] = d_e_jk2
        return e_s_mat

    def get_graph(self):
        return (copy.deepcopy(self.objects), copy.deepcopy(self.edges))

    def plot_figure(self):

        plt.figure(figsize=(8, 8), dpi=150)
        plt.scatter(self.points[:, 0], self.points[:, 1], s=1, c='k')
        plt.plot(self.poses[:, 0], self.poses[:, 1], 'co')
        # cloud = keyframe.fusedCloud
        #
        # plt.scatter(cloud[:, 0], cloud[:, 1], s=1, c='g')

        for k, (neighbor_objects, _) in self.graphs_neighbor.items():
            if k not in self.transformations_neighbor:
                R = np.eye(2)
                t = np.zeros([2])
            else:
                (R, t) = self.transformations_neighbor[k]
            if k == 1:
                self.plot_objects(neighbor_objects, R, t, color_pt='lightblue', color_obj='blue')
            if k == 2:
                self.plot_objects(neighbor_objects, R, t, color_pt='lightpink', color_obj='red')
            if k == 3:
                self.plot_objects(neighbor_objects, R, t, color_pt='lightcoral', color_obj='orange')

        self.plot_objects(self.objects, np.eye(2), np.zeros([2]), color_pt='lightgreen', color_obj='green')

        plt.xlim([-80, 80])
        plt.ylim([-80, 80])
        # plt.axis('equal')
        plt.savefig('test_data/' + str(self.robot_ns) + '/' + str(self.poses.shape[0]) + '.png')
        plt.close()

    def plot_objects(self, objects, R, t, color_pt='lightblue', color_obj='r'):
        if t.sum() != 0:
            print("transformation: ", R, t)
        for obj in objects.values():
            # plt.plot(obj.points_transformed[:, 0], obj.points_transformed[:, 1], color_pt)
            if obj.object_type == 0:
                color = 'green'
            elif obj.object_type == 1:
                color = 'blue'
            else:
                color = 'orange'
            bounding_box = np.dot(R, obj.bounding_box_transformed.T).T + t
            center = np.dot(R, obj.center.T).T + t
            plt.plot(bounding_box[:, 0],
                     bounding_box[:, 1],
                     color_obj, alpha=1)
            if obj.associated_object_id != -1:
                plt.plot([center[0], self.objects[obj.associated_object_id].center[0]],
                         [center[1], self.objects[obj.associated_object_id].center[1]],
                         color_obj, alpha=1)
