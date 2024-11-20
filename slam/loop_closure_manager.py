import numpy as np
from bruce_slam import pcl
from slam.object_mapper import transform_points, state_to_matrix, matrix_to_state
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from slam.utils import find_cliques


class RobotMessage:
    def __init__(self, robot_id, keyframe_id, pose, covariance, pose_truth=None):
        self.robot_id = robot_id
        self.id = keyframe_id
        self.pose = pose
        self.pose_truth = pose_truth
        self.covariance = covariance

        # pose transformed into the local robot coordinate frame
        self.pose_transformed = None
        # covariance transformed into the local robot coordinate frame
        self.covariance_transformed = None
        # keyframe ids matched with the keyframe id of the local robot
        self.matched_keyframe_id = None


class ScanMessage:
    def __init__(self, robot_id, keyframe_id, points):
        self.robot_id = robot_id
        self.id = keyframe_id
        self.points = points


class LoopClosureMessage:
    def __init__(self, source_keyframe_id, target_keyframe_id, robot_id):
        self.source_keyframe_id = source_keyframe_id
        self.target_keyframe_id = target_keyframe_id
        self.robot_id = robot_id

        self.fitness_score = -1
        # from source keyframe to target keyframe
        # dT \dot T_source = T_target
        self.between_pose = None

        # for evaluation purpose
        self.between_pose_truth = None

        # covariance matrix based on the fitness score
        self.cov = None
        self.added = False

    def set_measurement(self, pose, cov, fitness_score):
        self.between_pose = pose
        self.cov = cov
        self.fitness_score = fitness_score

    def __hash__(self):
        # Define a hash based on the attributes that make this class unique
        return hash((self.source_keyframe_id, self.target_keyframe_id, self.robot_id))

    def __eq__(self, other):
        # Define equality based on the unique attribute
        if isinstance(other, LoopClosureMessage):
            return (self.source_keyframe_id == other.source_keyframe_id and
                    self.target_keyframe_id == other.target_keyframe_id and
                    self.robot_id == other.robot_id)
        return False


class LoopClosureManager:
    def __init__(self, robot_ns, config):
        self.robot_ns = robot_ns
        # loop closure parameters
        self.config = config
        self.max_dist = config['max_dist']
        self.max_angle = config['max_angle'] * (np.pi / 180)
        self.use_ringkey = config['use_ringkey']
        self.min_overlap = config['min_overlap']
        self.cov_scale = config['cov_scale']

        self.icp_config = config['icp']
        self.window_size = config['window_size']

        self.visualize = config['visualize']

        self.min_pcm_value = config['min_pcm_value']
        self.pcm_queue_size = config['pcm_queue_size']
        self.min_num_points = config['min_num_points']

        # save history scan from robot self for loop closure detection
        self.historical_scans = []

        # ground truth pose for evaluation
        self.ground_truth_self = None

        # poses_neighbor dictionary:
        # key: neighbor robot_id
        # value: dictionary:
        #           key: neighbor robot keyframe id
        #           value: RobotMessage (neighbor robot keyframe poses and covariances)
        self.poses_neighbor = {}

        # points_neighbor dictionary:
        # key: neighbor robot_id
        # value: dictionary:
        #           key: neighbor robot keyframe id
        #           value: neighbor robot keyframe point clouds
        self.points_neighbor = {}

        # poses_id_neighbor dictionary:
        # key: neighbor robot_id
        # value: set of neighbor robot keyframe ids for poses received
        self.poses_id_neighbor = {}

        # points_id_neighbor dictionary:
        # key: neighbor robot_id
        # value: set of neighbor robot keyframe ids for point clouds received
        self.points_id_neighbor = {}

        # transformations_neighbor dictionary:
        # key: neighbor robot_id
        # value: ndarray transformation from neighbor robot to self robot`
        self.transformations_neighbor = {}

        # history loop
        self.loops = set()

        self.loops_added = set()

        self.robot_key_added = set()

        # ICP cloud registration for refine loop closures
        self.icp = pcl.ICP()
        self.icp.loadFromYaml(self.icp_config)

    def perform_icp(self, robot_id, latest_states_self):
        if robot_id not in self.points_id_neighbor:
            return

        for id in self.points_id_neighbor[robot_id]:
            source_keyframe = self.poses_neighbor[robot_id][id]

            # if no potential candidate, do nothing
            if source_keyframe.matched_keyframe_id is None:
                continue

            source_pose = source_keyframe.pose_transformed
            source_cloud = transform_points(self.points_neighbor[robot_id][id].points, source_pose)

            for target_keyframe_id in source_keyframe.matched_keyframe_id:
                loop_this = LoopClosureMessage(source_keyframe_id=id,
                                               target_keyframe_id=target_keyframe_id,
                                               robot_id=robot_id)
                # TODO: This is only meaningful if the inter-robot map merging is correct, or this would cause problem.
                # TODO: Try to fix this by introducing the map merging result also into consideration.
                # if loop already been processed before, do not process it again
                if loop_this in self.loops:
                    continue
                target_pose = latest_states_self[target_keyframe_id]
                # target_cloud = T_icp \cdot T_frame \cdot T_{neighbor_state} \cdot source_cloud
                target_cloud = transform_points(self.historical_scans[target_keyframe_id], target_pose)
                # using window strategy to add more points for registration
                if self.window_size != 0:
                    for i in range(target_keyframe_id - self.window_size, target_keyframe_id + self.window_size):
                        if i < 0 or i >= len(self.historical_scans):
                            continue
                        target_cloud = np.vstack((target_cloud,
                                                  transform_points(self.historical_scans[i], latest_states_self[i])))
                icp_status, icp_transform = self.icp.compute(source_cloud, target_cloud, np.eye(3))
                if not icp_status:
                    # if fall to register, continue
                    continue
                # check the quality of register
                source_cloud_transformed = transform_points(source_cloud, icp_transform)
                # get fitness score for the current loop
                idx, distances = pcl.match(target_cloud, source_cloud_transformed, 1, 0.5)
                overlap = len(idx[idx != -1]) / len(idx[0])
                # if not enough overlap, skip this loop candidate
                if overlap < self.min_overlap or len(source_cloud_transformed) < self.min_num_points:
                    continue
                # finally, the loop closure is a satisfying one
                source_pose_new = icp_transform @ state_to_matrix(source_pose)
                between_pose = np.linalg.inv(source_pose_new) @ state_to_matrix(target_pose)
                fitness_score = np.mean(distances[distances < float("inf")])
                cov = np.eye(3) * fitness_score * self.cov_scale
                loop_this.set_measurement(pose=matrix_to_state(between_pose), cov=cov, fitness_score=fitness_score)

                if self.ground_truth_self is not None:
                    # evaluate the loop closure
                    source_truth = source_keyframe.pose_truth
                    target_truth = self.ground_truth_self[target_keyframe_id]
                    between_truth = source_truth.between(target_truth)
                    loop_this.between_pose_truth = between_truth
                    if self.visualize:
                        source_truth = matrix_to_state(
                            state_to_matrix(target_pose) @ np.linalg.inv(
                                state_to_matrix(np.array([between_truth.x(),
                                                          between_truth.y(),
                                                          between_truth.theta()]))))
                        source_cloud_transformed3 = transform_points(self.points_neighbor[robot_id][id].points,
                                                                     source_truth)
                        plt.plot(source_cloud_transformed3[:, 0], source_cloud_transformed3[:, 1], 'y.')

                if self.visualize:
                    plt.plot(source_cloud[:, 0], source_cloud[:, 1], 'r.')
                    plt.plot(target_cloud[:, 0], target_cloud[:, 1], 'b.')
                    source_cloud_transformed2 = transform_points(self.points_neighbor[robot_id][id].points,
                                                                 source_pose_new)
                    plt.plot(source_cloud_transformed2[:, 0], source_cloud_transformed2[:, 1], 'g.')
                    plt.savefig(f"animate/loop/loop_"
                                f"{self.robot_ns}_{target_keyframe_id}_{robot_id}_{id}:{overlap:.1f}.png")
                    plt.close()

                self.loops.add(loop_this)

    def add_scan(self, scan):
        self.historical_scans.append(scan)

    def add_keyframes_neighbor(self, robot_id, ids, keyframes):
        if robot_id in self.poses_neighbor:
            self.poses_neighbor[robot_id].update(keyframes)
            self.poses_id_neighbor[robot_id].union(ids)
        else:
            self.poses_neighbor[robot_id] = keyframes
            self.poses_id_neighbor[robot_id] = ids

    def add_scans_neighbor(self, robot_id, ids, scans):
        if robot_id in self.points_neighbor:
            self.points_neighbor[robot_id].update(scans)
            self.points_id_neighbor[robot_id].union(ids)
        else:
            self.points_neighbor[robot_id] = scans
            self.points_id_neighbor[robot_id] = ids

    def get_ids_pose_not_received(self, robot_id, ids):
        if robot_id not in self.poses_id_neighbor:
            return ids
        return ids.difference(self.poses_id_neighbor[robot_id])

    def update_keyframe_poses_transformed(self, robot_id, latest_states_self):
        if robot_id not in self.poses_neighbor:
            return
        R, t = self.transformations_neighbor[robot_id]
        transformation = np.eye(3)
        transformation[:2, :2] = R
        transformation[:2, 2] = t
        neighbor_keyframe_id_matched = set()
        for keyframe_id, keyframe in self.poses_neighbor[robot_id].items():
            keyframe.pose_transformed = matrix_to_state(np.dot(transformation, state_to_matrix(keyframe.pose)))
            keyframe.covariance_transformed = np.dot(transformation, np.dot(keyframe.covariance, transformation.T))
            if self.use_ringkey:
                # TODO: implement ringkey
                pass
            else:
                dist = np.linalg.norm(latest_states_self[:, :2] - keyframe.pose_transformed.reshape(-1, 3)[:, :2],
                                      axis=1)
                angular_dist = (latest_states_self[:, 2] - keyframe.pose_transformed[2]) % (2 * np.pi)
                angular_dist[angular_dist > np.pi] -= 2 * np.pi

                indices_angle = np.where(np.abs(angular_dist) <= self.max_angle)[0]
                indices_dist = np.where(dist < self.max_dist)[0]

                indices = np.intersect1d(indices_angle, indices_dist)

                if len(indices) > 0:
                    keyframe.matched_keyframe_id = indices
                    # if no nearby robot scan, do not request scan from neighbor
                    if robot_id in self.points_id_neighbor:
                        if keyframe.id not in self.points_id_neighbor[robot_id]:
                            neighbor_keyframe_id_matched.add(keyframe.id)
                    else:
                        neighbor_keyframe_id_matched.add(keyframe_id)
        return neighbor_keyframe_id_matched

    def pcm(self, robot_id):
        pass

    def gcm(self, latest_states_self):
        if len(self.loops) < self.min_pcm_value:
            return []
        G = defaultdict(list)
        list_loop = list(self.loops)
        for (id_ik, ret_ik), (id_jl, ret_jl) in combinations(zip(range(len(list_loop)), list_loop), 2):
            # x_{ak}
            x_ak = state_to_matrix(latest_states_self[ret_ik.target_keyframe_id, :])
            # x_{bi}
            x_bi = self.poses_neighbor[ret_ik.robot_id][ret_ik.source_keyframe_id]
            x_a0bi = state_to_matrix(x_bi.pose_transformed)

            # x_{al}
            x_al = state_to_matrix(latest_states_self[ret_jl.target_keyframe_id, :])
            x_cj = self.poses_neighbor[ret_jl.robot_id][ret_jl.source_keyframe_id]
            x_a0cj = state_to_matrix(x_cj.pose_transformed)

            # from b_i to a_k
            z_ik = state_to_matrix(ret_ik.between_pose)
            # from c_j to a_l
            z_jl = state_to_matrix(ret_jl.between_pose)

            x_bicj = np.linalg.inv(x_a0bi) @ x_a0cj
            x_alak = np.linalg.inv(x_al) @ x_ak
            z_bi_ak = x_bicj @ z_jl @ x_alak
            e = matrix_to_state(np.linalg.inv(z_ik) @ z_bi_ak)
            try:
                md = e @ np.linalg.inv(ret_ik.cov) @ e
                # chi2.ppf(0.99, 3) = 11.34

                if md < 11.34:  # this is not a magic number
                    G[id_ik].append(id_jl)
                    G[id_jl].append(id_ik)
            except:
                pass

        # find the sets of consistent loops
        maximal_cliques = list(find_cliques(G))

        # if we got nothing, return nothing
        if not maximal_cliques:
            return []

        # sort and return only the largest set, also checking that the set is large enough
        maximum_clique = sorted(maximal_cliques, key=len, reverse=True)[0]
        if len(maximum_clique) < self.min_pcm_value:
            return []

        valid_loops = []
        for i in maximum_clique:
            if list_loop[i] in self.loops_added:
                continue
            valid_loops.append(list_loop[i])
            self.loops_added.add(list_loop[i])

        while len(self.loops) > self.pcm_queue_size:
            self.loops.pop()

        return valid_loops
