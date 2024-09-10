import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from slam.object_detection import *
import cvxpy as cp


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
    return principal_eigenvector


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

    def compare_all_neighbor_graph(self):
        for robot_ns_neighbor, robot_graph_neighbor in self.graphs_neighbor.items():
            self.compare_graphs((self.objects, self.edges), robot_graph_neighbor)

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

    # solve min_A\sum_{j=1}^N\sum_{k=1}^M trace(-A^TL)
    # with A(j,k)\in {0,1}
    # \sum^N_{j=1}A(j,k)<=1
    # \sum^M_{k=1} A(j,k) <=1
    def linear_assignment(self, L: np.ndarray, N: int, M: int):
        A = cp.Variable((N, M), boolean=True)

        # Objective function: sum of traces of A^T L
        objective = cp.Minimize(-cp.sum(cp.trace(A.T @ L)))

        # Constraints:
        # Each column sum should be <= 1
        column_constraints = [cp.sum(A[:, k]) <= 1 for k in range(M)]

        # Each row sum should be <= 1
        row_constraints = [cp.sum(A[j, :]) <= 1 for j in range(N)]

        # Combine constraints
        constraints = column_constraints + row_constraints

        # Define and solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve()

        a = A.value
        A_star = np.array(A.value)
        non_zero_indices = np.nonzero(A_star)


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
