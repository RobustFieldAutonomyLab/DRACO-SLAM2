import copy
import time

import gtsam
import numpy as np
from bruce_slam import pcl
from slam.object_mapper import transform_points, state_to_matrix, matrix_to_state
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from slam.utils import find_cliques
import yaml
import seaborn as sns

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
    def __init__(self, robot_id, keyframe_id, points, context):
        self.robot_id = robot_id
        self.id = keyframe_id
        self.context = context
        self.points = points

    def get_size(self):
        return (self.points.shape[0] * self.points.shape[1]) * 16 + 8 * 2


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

        self.overlap = -1.0

        # covariance matrix based on the fitness score
        self.cov = None
        self.checked = False

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


def loop_per_robot(loops, key_func):
    classified = defaultdict(set)
    for loop in loops:
        classified[key_func(loop)].add(loop)
    return classified


class LoopClosureManager:
    def __init__(self, robot_ns, config_path):
        self.robot_ns = robot_ns
        # loop closure parameters
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.config = config
        self.max_dist = config['max_dist']
        self.max_angle = config['max_angle'] * (np.pi / 180)
        self.use_ringkey = config['use_ringkey']
        self.min_overlap = config['min_overlap']
        self.cov_scale = config['cov_scale']

        self.icp_config = config['icp']
        self.window_size = config['window_size']
        self.icp_count = 0
        self.icp_time = []

        self.visualize = config['visualize']

        self.min_pcm_value = config['min_pcm_value']
        self.pcm_queue_size = config['pcm_queue_size']
        self.min_num_points = config['min_num_points']
        self.num_scan_one_time = config['num_scan_one_time']

        self.inter_factor_type = config['inter_factor_type']
        self.noise_model_type = config['noise_model_type']
        self.pcm_type = config['pcm_type']
        self.exchange_inter_robot_factor = config['exchange_inter_robot_factor']
        self.use_best_loop = config['use_best_loop']
        self.use_context = config['use_context']
        if self.use_context:
            self.context_difference = config['context_diff']

        # save history scan from robot self for loop closure detection
        self.historical_scans = []
        self.historical_contexts = []

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
        self.transformations_graph = gtsam.NonlinearFactorGraph()
        self.transformations_initial = gtsam.Values()

        # fix the transformation for robot itself
        self.transformations_graph.add(gtsam.PriorFactorPose2(self.robot_ns,
                                                              gtsam.Pose2(0.0, 0.0, 0.0),
                                                              gtsam.noiseModel.Isotropic.Sigma(3, 1e-6)))
        self.transformations_initial.insert(self.robot_ns, gtsam.Pose2(0.0, 0.0, 0.0))

        self.transformations_neighbor_optimized = {}

        # history loop
        # key: tuple(robot_id, source_keyframe_id, target_keyframe_id)
        # value: LoopClosureMessage
        self.loops = {}

        self.loops_added = set()

        self.robot_key_added = set()

        # keyframe ids to be published to the neighbor robots
        self.keyframe_id_to_publish = {}

        # ICP cloud registration for refine loop closures
        self.icp = pcl.ICP()
        self.icp.loadFromYaml(self.icp_config)

        # Neighbor factors to add into the graph once inter-robot loop closure is detected
        self.factors_neighbor = {}
        self.neighbor_added = set()

        self.neighbor_ringkeys = {}

        # list of inter-robot loop closures to be added
        self.inter_loops_to_add = []

    def compare_context(self, source_context, target_context):
        context_diff = np.sum(abs(source_context - target_context))
        if context_diff != -1:
            if np.sum(abs(source_context - target_context)) > self.context_difference:
                return False
        return True

    def perform_icp(self, robot_id, latest_states_self, comm_link = None):
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
                if self.use_context:
                    source_context = self.points_neighbor[robot_id][id].context
                    target_context = self.historical_contexts[target_keyframe_id]
                    if not self.compare_context(source_context, target_context):
                        continue

                time0 = time.time()
                loop_this = LoopClosureMessage(source_keyframe_id=id,
                                               target_keyframe_id=target_keyframe_id,
                                               robot_id=robot_id)
                # TODO: This is only meaningful if the inter-robot map merging is correct, or this would cause problem.
                # TODO: Try to fix this by introducing the map merging result also into consideration.
                # if loop already been processed before, do not process it again
                loop_key = (robot_id, id, target_keyframe_id)
                if loop_key in self.loops.keys():
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
                if len(self.historical_scans[target_keyframe_id]) == 0:
                    continue
                icp_status, icp_transform = self.icp.compute(source_cloud, target_cloud, np.eye(3))
                time1 = time.time()
                if comm_link is not None:
                    comm_link.log_time(time1 - time0, 'icp')
                self.icp_count += 1
                self.icp_time.append(time1 - time0)
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
                # cov[2, 2] = cov[2, 2] * 0.1
                loop_this.set_measurement(pose=matrix_to_state(between_pose), cov=cov, fitness_score=fitness_score)
                loop_this.overlap = overlap

                source_cloud_transformed3 = None
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
                        # plt.plot(source_cloud_transformed3[:, 0], source_cloud_transformed3[:, 1], 'y.')

                if self.visualize:
                    fig = plt.figure(figsize=(10, 6), dpi = 150)
                    # fig = plt.figure(figsize=(12, 6), dpi = 150)
                    sns.set_theme(style="white")
                    spec = fig.add_gridspec(5, 2)
                    co_pa = sns.color_palette('bright')
                    from matplotlib.font_manager import FontProperties
                    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
                    font = FontProperties(family='serif', fname=font_path, size=20)
                    axis_grid_0 = fig.add_subplot(spec[:4, 0])
                    axis_grid_1 = fig.add_subplot(spec[:4, 1])
                    axis_grid_0.scatter(source_cloud[:, 0], source_cloud[:, 1], s=10,
                                        color=co_pa[1], label = 'Source Cloud', zorder=5)
                    axis_grid_0.scatter(target_cloud[:, 0], target_cloud[:, 1], color='black', s=10,
                                        label = 'Target Cloud')
                    axis_grid_1.scatter(target_cloud[:, 0], target_cloud[:, 1], color='black', s=10)
                    axis_grid_0.set_aspect('equal')
                    axis_grid_1.set_aspect('equal')
                    source_cloud_transformed2 = transform_points(self.points_neighbor[robot_id][id].points,
                                                                 source_pose_new)
                    axis_grid_1.scatter(source_cloud_transformed2[:, 0],
                                        source_cloud_transformed2[:, 1], color=co_pa[2], s=10,
                                        label='Source Cloud (Registered)', zorder=5)
                    if source_cloud_transformed3 is not None:
                        axis_grid_1.scatter(source_cloud_transformed3[:, 0],
                                            source_cloud_transformed3[:, 1], color=co_pa[3], s=10,
                                            label='Source Cloud (GT)', zorder=5)
                    fig.legend(prop=font, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.05))
                    axis_grid_0.set_xlabel('x [m]', fontproperties=font)
                    axis_grid_0.set_ylabel('y [m]', fontproperties=font)
                    axis_grid_1.set_xlabel('x [m]', fontproperties=font)
                    axis_grid_1.set_ylabel('y [m]', fontproperties=font)
                    for label in axis_grid_0.get_xticklabels():
                        label.set_fontproperties(font)
                    for label in axis_grid_0.get_yticklabels():
                        label.set_fontproperties(font)
                    for label in axis_grid_1.get_xticklabels():
                        label.set_fontproperties(font)
                    for label in axis_grid_1.get_yticklabels():
                        label.set_fontproperties(font)
                    axis_grid_0.set_xticks(range(int(axis_grid_0.get_xticks()[0]),
                                                 int(axis_grid_0.get_xticks()[-1])+1, 10))
                    axis_grid_0.set_yticks(range(int(axis_grid_0.get_yticks()[0]),
                                                 int(axis_grid_0.get_yticks()[-1])+1, 10))
                    axis_grid_1.set_xticks(range(int(axis_grid_0.get_xticks()[0]),
                                                 int(axis_grid_0.get_xticks()[-1])+1, 10))
                    axis_grid_1.set_yticks(range(int(axis_grid_0.get_yticks()[0]),
                                                 int(axis_grid_0.get_yticks()[-1])+1, 10))
                    plt.tight_layout()
                    plt.savefig(f"animate/loop/loop_"
                                f"{self.robot_ns}_{target_keyframe_id}_{robot_id}_{id}:{overlap:.1f}.png")

                    plt.close()

                self.loops[loop_key] = loop_this

    def add_scan(self, scan):
        self.historical_scans.append(scan)

    def add_context(self, context):
        self.historical_contexts.append(context)

    def add_keyframes_neighbor(self, robot_id, ids, keyframes):
        if robot_id in self.poses_neighbor:
            self.poses_neighbor[robot_id].update(keyframes)
            self.poses_id_neighbor[robot_id] = self.poses_id_neighbor[robot_id].union(ids)
        else:
            self.poses_neighbor[robot_id] = keyframes
            self.poses_id_neighbor[robot_id] = ids

    def add_scans_neighbor(self, robot_id, ids, scans):
        if robot_id in self.points_neighbor:
            self.points_neighbor[robot_id].update(scans)
            self.points_id_neighbor[robot_id] = self.points_id_neighbor[robot_id].union(ids)
        else:
            self.points_neighbor[robot_id] = scans
            self.points_id_neighbor[robot_id] = ids

    def get_ids_pose_not_received(self, robot_id, ids):
        if robot_id not in self.poses_id_neighbor:
            return ids
        return ids.difference(self.poses_id_neighbor[robot_id])

    def update_keyframe_poses_transformed(self, robot_id, latest_states_self, tree):
        if robot_id not in self.poses_neighbor:
            return
        R, t = self.transformations_neighbor[robot_id]
        transformation = np.eye(3)
        transformation[:2, :2] = R
        transformation[:2, 2] = t
        neighbor_keyframe_id_matched = set()

        if self.use_ringkey:
            for keyframe_id, keyframe in self.poses_neighbor[robot_id].items():
                keyframe.pose_transformed = matrix_to_state(np.dot(transformation, state_to_matrix(keyframe.pose)))
                keyframe.covariance_transformed = np.dot(transformation, np.dot(keyframe.covariance, transformation.T))

            ringkeys_left = {}
            while self.neighbor_ringkeys[robot_id]:
                keyframe_id, neighbor_key = self.neighbor_ringkeys[robot_id].popitem()
                if keyframe_id not in self.poses_neighbor[robot_id].keys():
                    ringkeys_left[keyframe_id] = neighbor_key
                    continue
                keyframe = self.poses_neighbor[robot_id][keyframe_id]
                # get the tree we want to search against
                distances, indices = tree.query(neighbor_key,k=5,distance_upper_bound=20) # search
                indices = indices[distances <= 20]
                if len(indices) > 0:
                    keyframe.matched_keyframe_id = indices
                    # if no nearby robot scan, do not request scan from neighbor
                    if robot_id in self.points_id_neighbor:
                        if keyframe.id not in self.points_id_neighbor[robot_id]:
                            neighbor_keyframe_id_matched.add(keyframe.id)
                    else:
                        neighbor_keyframe_id_matched.add(keyframe_id)
            self.neighbor_ringkeys[robot_id] = ringkeys_left
        else:
            for keyframe_id, keyframe in self.poses_neighbor[robot_id].items():
                keyframe.pose_transformed = matrix_to_state(np.dot(transformation, state_to_matrix(keyframe.pose)))
                keyframe.covariance_transformed = np.dot(transformation, np.dot(keyframe.covariance, transformation.T))

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

    def pcm(self, latest_states_self):
        if len(self.loops) < self.min_pcm_value:
            return []
        graph_dict = loop_per_robot(self.loops.values(), lambda r: r.checked)
        graph_dict = loop_per_robot(graph_dict[False], lambda r: r.robot_id)
        loops_accepted = set()
        for robot_id, subset in graph_dict.items():
            loops_accepted.update(self.pcm_one_robot(subset, latest_states_self))
        return list(loops_accepted)

    def pcm_one_robot(self, loops, latest_states_self):
        if len(loops) < self.min_pcm_value:
            return []
        G = defaultdict(list)
        list_loop = list(loops)
        for (id_ik, ret_ik), (id_jl, ret_jl) in combinations(zip(range(len(list_loop)), list_loop), 2):
            # x_{ak}
            x_ak = state_to_matrix(latest_states_self[ret_ik.target_keyframe_id, :])
            # x_{bi}
            x_bi = self.poses_neighbor[ret_ik.robot_id][ret_ik.source_keyframe_id]
            x_a0bi = state_to_matrix(x_bi.pose)

            # x_{al}
            x_al = state_to_matrix(latest_states_self[ret_jl.target_keyframe_id, :])
            x_bj = self.poses_neighbor[ret_jl.robot_id][ret_jl.source_keyframe_id]
            x_a0bj = state_to_matrix(x_bj.pose)

            # from b_i to a_k
            z_ik = state_to_matrix(ret_ik.between_pose)
            # from c_j to a_l
            z_jl = state_to_matrix(ret_jl.between_pose)

            x_bibj = np.linalg.inv(x_a0bi) @ x_a0bj
            x_alak = np.linalg.inv(x_al) @ x_ak
            z_bi_ak = x_bibj @ z_jl @ x_alak
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
            valid_loops.append(copy.deepcopy(list_loop[i]))
            self.loops_added.add(copy.deepcopy(list_loop[i]))
            self.neighbor_added.add(copy.deepcopy(list_loop[i].robot_id))
        graph = loop_per_robot(self.loops.values(), lambda r: r.checked)
        print(f"valid loops: {len(loops)} {len(graph[False])}")
        while len(loops) > self.pcm_queue_size:
            loop_to_delete = loops.pop()
            loop_key = (loop_to_delete.robot_id, loop_to_delete.source_keyframe_id, loop_to_delete.target_keyframe_id)
            self.loops[loop_key].checked = True
        graph = loop_per_robot(self.loops.values(), lambda r: r.checked)
        print(f"After valid loops: {len(loops)} {len(graph[False])}")

        return valid_loops

    def perform_pcm(self, latest_states_self):
        if self.pcm_type == 0:
            graph_dict = loop_per_robot(self.loops.values(), lambda r: r.checked)
            loops = graph_dict[False]
            for loop in loops:
                loop_key = (loop.robot_id, loop.source_keyframe_id, loop.target_keyframe_id)
                self.loops[loop_key].checked = True
        elif self.pcm_type == 1:
            loops = self.pcm(latest_states_self)
        else:
            loops = self.gcm(latest_states_self)
        # Add transformation from valid loop to the transformation optimization factor graph
        self.add_valid_loops_to_transformations_graph(loops, latest_states_self)
        return loops

    def gcm(self, latest_states_self):
        if len(self.loops) < self.min_pcm_value:
            return []
        graph_dict = loop_per_robot(self.loops.values(), lambda r: r.checked)
        G = defaultdict(list)
        list_loop = list(graph_dict[False])
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
            valid_loops.append(copy.deepcopy(list_loop[i]))
            self.loops_added.add(copy.deepcopy(list_loop[i]))
            self.neighbor_added.add(copy.deepcopy(list_loop[i].robot_id))

        graph = loop_per_robot(self.loops.values(), lambda r: r.checked)
        print(f"valid loops: {len(list_loop)} {len(graph[False])}")
        while len(list_loop) > self.pcm_queue_size:
            loop_to_delete = list_loop.pop(0)
            loop_key = (loop_to_delete.robot_id, loop_to_delete.source_keyframe_id, loop_to_delete.target_keyframe_id)
            self.loops[loop_key].checked = True
        graph = loop_per_robot(self.loops.values(), lambda r: r.checked)
        print(f"After valid loops: {len(list_loop)} {len(graph[False])}")

        return valid_loops

    def add_valid_loops_to_transformations_graph(self, loops, latest_states_self):
        for loop in loops:
            id_0 = loop.source_keyframe_id
            ch_0 = loop.robot_id
            id_1 = loop.target_keyframe_id
            ch_1 = self.robot_ns
            matrix_neighbor = state_to_matrix(self.poses_neighbor[ch_0][id_0].pose)
            matrix_target = state_to_matrix(latest_states_self[id_1, :])
            matrix_between = state_to_matrix(loop.between_pose)
            matrix_self = matrix_target @ np.linalg.inv(matrix_between)
            pose = matrix_to_state(matrix_self @ np.linalg.inv(matrix_neighbor))
            pose_btwn = gtsam.Pose2(pose[0], pose[1], pose[2])
            noise_model = gtsam.noiseModel.Gaussian.Covariance(loop.cov)
            if loop.overlap < 0.9:
                robust = gtsam.noiseModel.mEstimator.Cauchy.Create(1.0)
                noise_model = gtsam.noiseModel.Robust.Create(robust, noise_model)

            factor = gtsam.BetweenFactorPose2(ch_1, ch_0, pose_btwn, noise_model)
            if ch_0 not in self.transformations_initial.keys():
                self.transformations_initial.insert(ch_0, pose_btwn)
            self.transformations_graph.add(factor)
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.transformations_graph, self.transformations_initial)
        self.transformations_initial = optimizer.optimize()
        # update the optimized transformation
        for robot_id in self.transformations_initial.keys():
            if robot_id == self.robot_ns:
                continue
            pose = self.transformations_initial.atPose2(robot_id)
            mat = state_to_matrix(np.array([pose.x(), pose.y(), pose.theta()]))
            # self.transformations_neighbor_optimized[robot_id] = (mat[:2, :2], mat[:2, 2])
            # print("transformation_optimized: ", mat)

    def get_keyframes(self, robot_id_to, keyframe_id_set):
        msgs = {}
        msgs_id_set = set()
        if robot_id_to in self.keyframe_id_to_publish.keys():
            self.keyframe_id_to_publish[robot_id_to] = self.keyframe_id_to_publish[robot_id_to].union(keyframe_id_set)
        else:
            self.keyframe_id_to_publish[robot_id_to] = keyframe_id_set
        while self.keyframe_id_to_publish[robot_id_to]:
            keyframe_id = self.keyframe_id_to_publish[robot_id_to].pop()
            if len(self.historical_scans[keyframe_id]) < self.min_num_points:
                continue
            msgs[keyframe_id] = ScanMessage(robot_id=self.robot_ns,
                                            keyframe_id=keyframe_id,
                                            points=self.historical_scans[keyframe_id],
                                            context=self.historical_contexts[keyframe_id])
            msgs_id_set.add(keyframe_id)
            if len(msgs) >= self.num_scan_one_time:
                break
        return msgs_id_set, msgs

    def prepare_loops_to_send(self, loops):
        if not self.exchange_inter_robot_factor:
            return []
        msgs = []
        for loop in loops:
            msg = {
                'r0': loop.robot_id,
                'key0': loop.source_keyframe_id,
                'r1': self.robot_ns,
                'key1': loop.target_keyframe_id,
                'pose': loop.between_pose,
                'cov': loop.cov}
            msgs.append(msg)

        return msgs

    def get_transformation(self, robot_id):
        if robot_id in self.transformations_neighbor_optimized.keys():
            return self.transformations_neighbor_optimized[robot_id]
        elif robot_id in self.transformations_neighbor.keys():
            return self.transformations_neighbor[robot_id]
        else:
            return None

    def record_all_loops(self, path):
        with open(path, "w") as file:
            file.write(f"robot0, id0, robot1, id1, dx, dy, dtheta, "
                       f"dx_truth, dy_truth, dtheta_truth, num_points, overlap\n")
            for loop in self.loops.values():
                file.write(f"{loop.robot_id}, {loop.source_keyframe_id}, "
                           f"{self.robot_ns}, {loop.target_keyframe_id}, "
                           f"{loop.between_pose[0]:.3f}, {loop.between_pose[1]:.3f}, {loop.between_pose[2]:.3f}, "
                           f"{loop.between_pose_truth.x():.3f}, "
                           f"{loop.between_pose_truth.y():.3f}, "
                           f"{loop.between_pose_truth.theta():.3f}, "
                           f"{len(self.points_neighbor[loop.robot_id][loop.source_keyframe_id].points)}, "
                           f"{loop.overlap}\n")


def compute_covariance(T1, T2, cov1, cov2):
    # Adjoint matrix of T (SE(2) transformation)
    def adjoint(T):
        R = T[:2, :2]
        t = T[:2, 2]
        adj = np.zeros((3, 3))
        adj[:2, :2] = R.T
        adj[:2, 2] = -R.T @ t
        adj[2, 2] = 1
        return adj

    # Compute the adjoint of T1^{-1} (the transformation in SE(2))
    Ad_T1_inv = adjoint(np.linalg.inv(T1))

    # Compute Jacobians:
    # For T1^{-1}, we use the adjoint of T1^{-1}
    # For T2, the Jacobian is the identity matrix in SE(2)
    J1 = Ad_T1_inv

    # Propagate covariance: cov12 = Ad_T1_inv @ cov2 @ Ad_T1_inv.T + cov1
    cov12 = J1 @ cov2 @ J1.T + cov1
    return cov12
