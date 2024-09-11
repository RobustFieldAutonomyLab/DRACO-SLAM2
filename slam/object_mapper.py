import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from slam.object_detection import *
from scipy.optimize import linear_sum_assignment


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
    eigenvalues, eigenvectors = np.linalg.eig(mat)

    # Find the index of the largest eigenvalue
    principal_eigenvalue_index = np.argmax(eigenvalues)

    # Get the principal eigenvector
    principal_eigenvector = eigenvectors[:, principal_eigenvalue_index]
    return np.abs(principal_eigenvector)


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

        self.graphs_neighbor = {}

        # graph construction and matching parameters
        with open(config["graph_matching"], 'r') as file:
            config_graph = yaml.safe_load(file)

        self.max_edge_distance = config_graph["edge"]["max_distance"]
        self.min_node_distance = config_graph["node"]["min_distance_accept"]
        self.min_num_node = config_graph["node"]["min_number"]
        self.min_num_edge = config_graph["edge"]["min_number"]
        self.match_use_type = config_graph["matching"]["use_type"]
        self.mu = config_graph["matching"]["mu"]
        self.min_eigenvector_accept = config_graph["matching"]["min_eigenvector_accept"]

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
        for robot_ns_neighbor, robot_graph_neighbor in self.graphs_neighbor.items():
            self.compare_graphs((self.objects, self.edges), robot_graph_neighbor)

        self.plot_figure()

    def compare_graphs(self, graph_self: tuple, graph_neighbor: tuple):
        nodes_self, edges_self = graph_self
        nodes_neighbor, edges_neighbor = graph_neighbor
        if len(nodes_self) < self.min_num_node or len(nodes_neighbor) < self.min_num_node:
            return
        if len(edges_self) < self.min_num_edge or len(edges_neighbor) < self.min_num_edge:
            return
        nodes_self_id_sorted = sorted(nodes_self.keys())
        nodes_neighbor_id_sorted = sorted(nodes_neighbor.keys())
        N = len(nodes_neighbor)
        M = len(nodes_self)

        # perform graph matching
        # construct the edge weight assignment matrix
        S = self.construct_edge_similarity_matrix(graph_self=graph_self, graph_neighbor=graph_neighbor)
        # solve NP-hard question using principal eigen vector of edge_similarity_mat S
        A_star = get_principal_eigen_vector(S)
        A_star_matrix = A_star.reshape(N, M)
        node_matching = self.linear_assignment(A_star_matrix, N, M)
        for node_idx_self, node_idx_neighbor in node_matching.items():
            node_id_neighbor = nodes_neighbor_id_sorted[node_idx_neighbor]
            graph_neighbor[0][node_id_neighbor].associated_object_id = nodes_self_id_sorted[node_idx_self]

    def linear_assignment(self, L: np.ndarray, N: int, M: int):
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
        print("node_indices_neighbor", node_indices_neighbor)
        print("node_indices_self", node_indices_self)
        return assignment_result

    def construct_edge_similarity_matrix(self, graph_self: tuple, graph_neighbor: tuple):
        nodes_self, edges_self = graph_self
        nodes_neighbor, edges_neighbor = graph_neighbor
        nodes_self_id_sorted = sorted(nodes_self.keys())
        nodes_neighbor_id_sorted = sorted(nodes_neighbor.keys())
        size_s = len(nodes_self) * len(nodes_neighbor)

        # row: number of edges self, col: number of edges neighbor
        e_s_mat = np.zeros([size_s, size_s])
        for j0, j1, ej01 in edges_neighbor:
            for k0, k1, ek01 in edges_self:
                # if not using object type for edge matching
                d_e_jk = np.exp(-self.mu * abs(ej01 - ek01))
                case_1 = (nodes_neighbor[j0].compare_object_type(nodes_self[k0])
                          and nodes_neighbor[j1].compare_object_type(nodes_self[k1]))
                case_2 = (nodes_neighbor[j1].compare_object_type(nodes_self[k0])
                          and nodes_neighbor[j0].compare_object_type(nodes_self[k1]))
                if not self.match_use_type or case_1:
                    # case 1
                    s0 = nodes_neighbor_id_sorted.index(j0) * len(nodes_self) + nodes_self_id_sorted.index(k0)
                    s1 = nodes_neighbor_id_sorted.index(j1) * len(nodes_self) + nodes_self_id_sorted.index(k1)
                    e_s_mat[s0, s1] = d_e_jk
                    e_s_mat[s1, s0] = d_e_jk
                if not self.match_use_type or case_2:
                    # case 2
                    s0 = nodes_neighbor_id_sorted.index(j0) * len(nodes_self) + nodes_self_id_sorted.index(k1)
                    s1 = nodes_neighbor_id_sorted.index(j1) * len(nodes_self) + nodes_self_id_sorted.index(k0)
                    e_s_mat[s0, s1] = d_e_jk
                    e_s_mat[s1, s0] = d_e_jk
        return e_s_mat

    def plot_figure(self):

        plt.figure(figsize=(8, 8), dpi=150)
        # plt.scatter(self.points[:, 0], self.points[:, 1], s=1, c='k')
        plt.plot(self.poses[:, 0], self.poses[:, 1], 'ro')
        # cloud = keyframe.fusedCloud
        #
        # plt.scatter(cloud[:, 0], cloud[:, 1], s=1, c='g')

        for k, (neighbor_objects, _) in self.graphs_neighbor.items():
            if k == 1:
                self.plot_objects(neighbor_objects, color_pt='lightblue', color_obj='blue')
            if k == 2:
                self.plot_objects(neighbor_objects, color_pt='lightpink', color_obj='red')
            if k == 3:
                self.plot_objects(neighbor_objects, color_pt='lightcoral', color_obj='orange')

        self.plot_objects(self.objects, color_pt='lightgreen', color_obj='green')

        plt.xlim([-80, 80])
        plt.ylim([-80, 80])
        # plt.axis('equal')
        plt.savefig('test_data/' + str(self.robot_ns) + '/' + str(self.poses.shape[0]) + '.png')
        plt.close()

    def plot_objects(self, objects, color_pt='lightblue', color_obj='r'):
        for obj in objects.values():
            # plt.plot(obj.points_transformed[:, 0], obj.points_transformed[:, 1], color_pt)
            if obj.object_type == 0:
                color = color_obj
            elif obj.object_type == 1:
                color = color_obj
            else:
                color = 'g'
            plt.plot(obj.bounding_box_transformed[:, 0],
                     obj.bounding_box_transformed[:, 1],
                     color, alpha=.5)
            if obj.associated_object_id != -1:
                plt.plot([obj.center[0], self.objects[obj.associated_object_id].center[0]],
                         [obj.center[1], self.objects[obj.associated_object_id].center[1]],
                         'black', alpha=.5)
