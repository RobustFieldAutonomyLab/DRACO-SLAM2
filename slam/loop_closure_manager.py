import numpy as np
import yaml
from bruce_slam import pcl
from slam.object_mapper import transform_points, state_to_matrix, matrix_to_state


class LoopClosureMessage:
    def __init__(self, source_keyframe_id, target_keyframe_id, robot_id):
        self.source_keyframe_id = source_keyframe_id
        self.target_keyframe_id = target_keyframe_id
        self.robot_id = robot_id

        self.accept = False
        self.fitness_score = -1
        # from source keyframe to target keyframe
        self.between_pose = np.empty([3, 1])
        self.cov = np.empty([3, 3])

    def set_measurement(self, pose, cov, fitness_score):
        self.accept = True
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

        # save history scan from robot self for loop closure detection
        self.historical_scans = []

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
                icp_status, icp_transform = self.icp.compute(source_cloud, target_cloud, np.eye(3))
                if not icp_status:
                    # if fall to register, continue
                    continue
                # check the quality of register
                source_cloud_transformed = transform_points(source_cloud, icp_transform)
                # get fitness score for the current loop
                idx, distances = pcl.match(source_cloud_transformed, target_cloud, 1, 0.5)
                overlap = len(idx[idx != -1]) / len(idx[0])
                # if not enough overlap, skip this loop candidate
                if overlap < self.min_overlap:
                    self.loops.add(loop_this)
                    continue
                # finally, the loop closure is a satisfying one
                source_pose = icp_transform @ state_to_matrix(source_pose)
                between_pose = 
                fitness_score = np.mean(distances[distances < float("inf")])
                cov = np.eye(3) * fitness_score * 20.0
                loop_this.set_measurement(pose=icp_transform, cov=cov, fitness_score=fitness_score)

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
            keyframe.pose_transformed = np.dot(transformation, keyframe.pose)
            keyframe.covariance_transformed = np.dot(transformation, np.dot(keyframe.covariance, transformation.T))
            if self.use_ringkey:
                # TODO: implement ringkey
                pass
            else:
                dist = np.linalg.norm(latest_states_self[:, :2] - keyframe.pose_transformed.reshape(-1, 3)[:, :2],
                                      axis=1)
                angular_dist = (latest_states_self[:, 2] - keyframe.pose_transformed[2]) % (2 * np.pi)
                angular_dist[angular_dist > np.pi] -= 2 * np.pi

                indices_angle = np.where(np.abs(angular_dist) < self.max_angle)[0]
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
