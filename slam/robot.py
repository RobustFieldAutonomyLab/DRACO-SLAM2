from scipy.spatial import KDTree
from typing import Tuple
import numpy as np
import gtsam
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge
import pickle
import os
import time
import copy
from slam.loop_closure_manager import compute_covariance

from slam.utils import create_full_cloud, get_all_context, get_points, verify_pcm, X, robot_to_symbol, numpy_to_gtsam
from slam.utils import transform_points as transform_points_gtsam

from slam.loop_closure import LoopClosure
from slam.object_detection import Keyframe
from slam.object_mapper import ObjectMapper, transform_points, matrix_to_state, state_to_matrix
from slam.loop_closure_manager import LoopClosureManager, RobotMessage, ScanMessage


def write_trajectory(path, states):
    with open(path, "w") as file:
        for i, state in enumerate(states):
            # timestamp tx ty tz qx qy qz qw
            if isinstance(state, np.ndarray):
                x = state[0]
                y = state[1]
                theta = state[2]
            elif isinstance(state, gtsam.Pose2):
                x = state.x()
                y = state.y()
                theta = state.theta()
            half_theta = theta/2.0
            file.write(f"{i:.2f} {x} {y} 0.0 0.0 0.0 {np.sin(half_theta)} {np.cos(half_theta)}\n")


class Robot():
    def __init__(self, robot_id: int, data: dict, run_config: dict, sc_config: dict):

        self.robot_id = robot_id  # unique ID number

        self.slam_step = 0  # we want to track how far along the mission is, what SLAM step are we on
        self.total_steps = len(data["poses"])

        self.poses = data["poses"]  # poses as numpy arrays
        self.poses_g = data["poses_g"]  # poses from above but as gtsam.Pose2
        self.points = data["points"]  # raw points at each pose
        self.points_t = data["points_t"]  # transformed points at each pose
        self.truth = data["truth"]  # ground truth for each pose
        self.factors = data["factors"]  # the factors in the graph
        self.images = data["images"]  # the images at each pose

        self.partner_truth = {}  # the partner robots ground truth

        self.points_partner = {}

        # get the scan context images and ring keys
        self.keys, self.context = get_all_context(self.poses, self.points, sc_config)

        self.submap_size = sc_config['submap_size']
        self.pcm_dict = {}
        self.pcm_queue_size = 5
        self.min_pcm = 2

        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.values_added = {}
        self.isam_params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(self.isam_params)
        self.isam_combined = None
        self.state_estimate = []  # my own state estimate
        self.covariance = []
        self.partner_robot_state_estimates = {}  # my estimates about the other robots
        self.partner_robot_trajectories = {}  # the trajectory each robot sends to me, they figure this out
        self.partner_reference_frames = {}  # my estimate of where I think the this other robot started
        self.multi_robot_frames = {}  # the frames I have added as loop closures from other robots
        self.partner_robot_covariance = {}
        self.partner_robot_exchange_costs = {}
        self.my_exchange_costs = None

        self.prior_sigmas = [0.2, 0.2, 0.02]
        self.prior_model = self.create_noise_model(self.prior_sigmas)
        self.partner_robot_model = self.create_noise_model([0.05, 0.05, 0.005])

        self.inter_robot_loop_closures = []

        self.point_clouds_received = {}  # log when we have gotten a point cloud. Format: robot_id_number, keyframe_id

        self.possible_loops = {}
        self.best_possible_loops = {}
        self.tested_loops = {}

        self.mse = None
        self.rmse = None
        self.team_uncertainty = []

        self.loops_tested = {}

        self.dummy_cloud = create_full_cloud()
        self.poses_needing_loops = {}
        self.icp_count = 0
        self.icp_success_count = 0
        self.merged = {}

        self.pcm_run_time = []
        self.alcs_run_time = []
        self.alcs_reg_time = []
        self.draco_reg_time = []

        self.is_shutdown = False
        self.mission = run_config['mission']
        self.mode = run_config['mode']
        self.min_uncertainty = run_config['min_uncertainty']

        # object_detection_params = {"object_icp": "config/object_icp.yaml",
        #                            "model": "models/sim_model.h5",
        #                            "feature_extraction": "config/feature.yaml"}
        self.object_detection = ObjectMapper(robot_id, run_config['object_detection'])
        self.loop_closure_manager = LoopClosureManager(robot_id, run_config['loop_closure'])
        if self.truth is not None:
            self.loop_closure_manager.ground_truth_self = self.truth
        self.update_poses = run_config['update_poses']
        self.source_key_before = {}

    def create_noise_model(self, *sigmas: list) -> gtsam.noiseModel.Diagonal:
        """Create a noise model from a list of sigmas, treated like a diagnal matrix.

        Returns:
            gtsam.noiseModel.Diagonal: gtsam version of input
        """
        return gtsam.noiseModel.Diagonal.Sigmas(np.r_[sigmas])

    def create_full_noise_model(self, cov: np.array) -> gtsam.noiseModel.Gaussian.Covariance:
        """Create a noise model from a numpy array

        Args:
            cov (np.array): numpy array of the covariance matrix.

        Returns:
            gtsam.noiseModel.Gaussian.Covariance: gtsam version of the input
        """

        return gtsam.noiseModel.Gaussian.Covariance(cov)

    def create_robust_noise_model(self, cov: np.array):
        """Create a noise model from a numpy array

        Args:
            cov (np.array): numpy array of the covariance matrix.

        Returns:
            gtsam.noiseModel_Robust: gtsam version of the input
        """

        cov = np.eye(3) * cov
        model = gtsam.noiseModel.Gaussian.Covariance(cov)
        robust = gtsam.noiseModel.mEstimator.Cauchy.Create(1.0)
        return gtsam.noiseModel.Robust.Create(robust, model)

    def update_merge_log(self, robot: int) -> None:
        """Update which robots I have merged with

        Args:
            robot (int): the robot I have now merged with
        """

        self.merged[robot] = True

    def is_merged(self, robot: int) -> bool:
        """Check if I am merged with the a robot

        Args:
            robot (int): the robot I want to check if I am merged with

        Returns:
            bool: True if I am merged with that robot
        """

        return robot in self.merged

    def alcs(self, robot: int) -> None:
        """Perform an active loop closure search (ALCS) step. 
        Here we find some poses that need loops, find possible loop 
        closures with them, and then select the best possible loop
        closure. Note that this is performed with a specified robot. 

        Args:
            robot (int): the robot we want to do an alcs step with
        """

        if robot in self.merged:  # we can only do alcs when the robots are merged
            start_time = time.time()
            self.find_poses_needing_loops(robot)
            self.search_for_possible_loops(robot)
            self.simulate_loop_closure(robot)
            self.alcs_run_time.append(time.time() - start_time)

    def step(self, path: str) -> None:
        """Increase the step of SLAM
        """
        # TODO: make poses update in real-time for the object detection
        if self.slam_step == 0:
            self.start_graph()
            self.update_graph()
            self.update_scene_graph()
        if self.slam_step + 1 >= self.total_steps:
            self.is_shutdown = True
        if not self.is_shutdown:
            self.slam_step += 1
            self.add_factors()

        self.update_graph()
        self.update_scene_graph()

    def update_scene_graph(self):
        if len(self.images) != 0:
            keyframe = Keyframe(self.slam_step,
                                self.poses[self.slam_step],
                                self.images[self.slam_step],
                                self.points[self.slam_step])
        else:
            keyframe = Keyframe(self.slam_step,
                                self.poses[self.slam_step],
                                None,
                                self.points[self.slam_step])
        self.object_detection.add_object(keyframe, self.state_estimate)
        self.loop_closure_manager.add_scan(self.points[self.slam_step])

    def start_graph(self) -> None:
        """Start the SLAM graph by inserting the prior.
        """

        pose = self.poses_g[0]
        factor = gtsam.PriorFactorPose2(X(0), pose, self.prior_model)
        self.graph.add(factor)
        self.values.insert(X(0), pose)
        self.values_added[0] = True

    def add_factors(self) -> None:
        """Add the most recent factors to the graph. The 
        factors we need to add are at self.slam_step
        """

        if self.slam_step not in self.factors: return
        factors_to_add = self.factors[self.slam_step]
        for factor in factors_to_add:
            i, j, transform, sigmas = factor
            sigmas = np.array(sigmas)

            if sigmas.shape == (3,):
                noise_model = self.create_noise_model(sigmas)
            else:
                noise_model = self.create_full_noise_model(sigmas)

            factor_gtsam = gtsam.BetweenFactorPose2(X(i), X(j), transform, noise_model)

            if j not in self.values_added:
                self.values.insert(X(j), self.poses_g[j])
                self.values_added[j] = True
            self.graph.add(factor_gtsam)

    def get_keys(self) -> np.array:
        """Return the array of ring keys from this SLAM step

        Returns:
            np.array: ring keys up to this self.slam_step
        """

        return np.array(self.keys[:self.slam_step])

    def get_key(self) -> Tuple[np.array, int]:
        """Get the current ring key and it's index

        Returns:
            Tuple[np.array, int]: the current or last ring key with it's index
        """

        if self.slam_step >= len(self.keys):
            return self.keys[-1], len(self.keys) - 1
        return self.keys[self.slam_step], self.slam_step

    def get_context(self) -> np.array:
        """Get the current scan context image

        Returns:
            np.array: the current or last scan context image
        """

        if self.slam_step >= len(self.context):
            return self.context[-1]
        return self.context[self.slam_step]

    def get_context_index(self, index: int) -> np.array:
        """Get the scan context image at index

        Args:
            index (int): the step we want scan context from 

        Returns:
            np.array: the scan context image
        """

        return self.context[index]

    def get_tree(self) -> KDTree:
        """Return the KDTree from this SLAM step

        Returns:
            KDTree: KDTree up to self.slam_step
        """

        return KDTree(self.keys[:self.slam_step])

    def get_robot_points(self, index: int, submap_size: int) -> np.array:
        """Get the submap at an index given the submap size. 
        Use utils.py get_points. 

        Args:
            index (int): the index we want the points at
            submap_size (int): the size of the submap in each direction

        Returns:
            np.array: the points ready for registration
        """

        return get_points(index, submap_size, self.points, self.poses)

    def get_pose_gtsam(self) -> gtsam.Pose2:
        """Return the pose at the current step

        Returns:
            gtsam.Pose2: the gtsam pose
        """
        # TODO UPDATE
        if self.slam_step >= len(self.poses_g): return self.poses_g[-1]
        return self.poses_g[self.slam_step]

    def get_pose_gtsam_at_index(self, index: int) -> gtsam.Pose2:
        """Returns the pose at the requested step, step index

        Args:
            index (int): the slam step we want the pose from 

        Returns:
            gtsam.Pose2: the gtsam pose
        """

        assert (index < len(self.state_estimate))
        assert (index >= 0)
        return numpy_to_gtsam(self.state_estimate[index])

    def add_loop_to_pcm_queue(self, loop: LoopClosure) -> None:
        """Add a loop closure the pcm queue. Prune any old loops before we add.

        Args:
            loop (LoopClosure): the loop closure we want to add
        """

        # which robot is this loop closure with?
        # pull the relavant PCM queue
        if loop.target_robot_id not in self.pcm_dict:
            pcm_queue = []
        else:
            pcm_queue = self.pcm_dict[loop.target_robot_id]

        # update the pcm queue
        while (pcm_queue and loop.source_key - pcm_queue[0].source_key > self.pcm_queue_size):
            pcm_queue.pop(0)
        pcm_queue.append(loop)  # add the loop
        self.pcm_dict[loop.target_robot_id] = pcm_queue  # store the pcm queue

    def do_pcm(self, robot_id: int) -> list:
        """Check the self.pcm_queue for any valid loop closures. Return them as a list.

        Args:
            robot_id (int): the ID of the robot the most recent loop is with. This 
            is the queue we need to check with PCM. 

        Returns:
            list: a list of LoopClosures
        """

        assert (len(self.pcm_dict[robot_id]) != 0)

        valid_loops = []
        start_time = time.time()
        valid_indexes = verify_pcm(self.pcm_dict[robot_id], self.min_pcm)
        self.pcm_run_time.append(time.time() - start_time)
        for i in valid_indexes:
            if self.pcm_dict[robot_id][i].inserted == False:
                self.pcm_dict[robot_id][i].inserted = True
                valid_loops.append(self.pcm_dict[robot_id][i])
        return valid_loops

    def check_for_data(self, robot_id: int, keyframe_id: int) -> Tuple[list, int]:
        """Checks if we need any data exchange to complete the ICP call.
        returns a list of [[robot_id_number, keyframe_id]]

        Args:
            robot (int): robot id number we are calling icp with
            index (int): the keyframe index for the above robot

        Returns:
            list: a list of the point clouds we need to complete this job
            int: the cost of requesting this data
        """

        data_requests = []  # list of the data we need
        comms_cost = 0
        # check the whole submap to see if we need any of the point clouds
        for i in range(keyframe_id - self.submap_size, keyframe_id + self.submap_size + 1):
            if (robot_id, i) not in self.point_clouds_received and i >= 0:  # if we don't have it, ask for it
                data_requests.append([robot_id, i])
                self.point_clouds_received[(robot_id, i)] = True
                comms_cost += 32 + 32  # we only need two 32 bit integers
        return data_requests, comms_cost

    def get_data(self, data_request: list) -> int:
        """Find the communication cost of getting the above data

        Args:
            data_request (list): the list of robot keyframes we want

        Returns:
            int: the cost of exchanging this data
        """

        comms_cost = 0
        for row in data_request:
            _, i = row
            if i < len(self.points):  # check for out of range
                # get the comms cost
                # Each point is a pair of 32 bit floats
                comms_cost += len(self.points[i]) * 2 * 32

        return comms_cost

    def update_graph(self) -> None:
        """Update the state estimate based on what we have in
        the graph.
        """

        # push the newest factors into the ISAM2 instance
        self.isam.update(self.graph, self.values)
        self.graph.resize(0)  # clear the graph and values once we push it to ISAM2
        self.values.clear()
        isam = gtsam.ISAM2(self.isam)  # make a copy

        # add and update the partner robot trajectories
        for robot in self.partner_robot_state_estimates.keys():
            isam = self.merge_trajectory(isam, robot)
        self.isam_combined = isam

        # update the state estimate
        values = isam.calculateEstimate()

        # Update MY whole trajectory
        temp = []
        temp_2 = []
        for x in range(self.slam_step + 1):
            pose = values.atPose2(X(x))
            cov = isam.marginalCovariance(X(x))
            temp.append([pose.x(), pose.y(), pose.theta()])
            temp_2.append(cov)
        self.state_estimate = np.array(temp)
        self.covariance = np.array(temp_2)

        # update my the estimate of my partner robots in my frame
        for robot in self.partner_robot_state_estimates.keys():
            if len(self.multi_robot_frames[robot]) == 0: continue
            for i in range(len(self.partner_robot_trajectories[robot])):
                self.partner_robot_state_estimates[robot][i] = values.atPose2(robot_to_symbol(robot, i))
                self.partner_robot_covariance[robot][i] = isam.marginalCovariance(robot_to_symbol(robot, i))

    def merge_trajectory(self, isam: gtsam.ISAM2, robot_id: int) -> gtsam.ISAM2:
        """Add the partner robot trajectory to the slam graph. Note that we use and return a copy
        of the isam instance. 

        Args:
            isam (gtsam.ISAM2): copy of the isam instance
            robot_id (int): the robot id we want to merge with

        Returns:
            gtsam.ISAM2: the update isam instance
        """

        # check if we can even perform a merge
        if len(self.multi_robot_frames[robot_id]) == 0: return isam  # do we have any loop closures?

        # objects to update isam
        values = gtsam.Values()
        graph = gtsam.NonlinearFactorGraph()

        # add the whole trajectory
        for i in range(len(self.partner_robot_trajectories[robot_id]) - 1):
            pose_i = numpy_to_gtsam(self.partner_robot_trajectories[robot_id][i])  # get the poses in gtsam
            pose_i_plus_1 = numpy_to_gtsam(self.partner_robot_trajectories[robot_id][i + 1])
            pose_i = self.partner_reference_frames[robot_id].compose(pose_i)  # place in the correct ref frame
            pose_i_plus_1 = self.partner_reference_frames[robot_id].compose(pose_i_plus_1)
            pose_between = pose_i.between(pose_i_plus_1)  # get the pose between them and package as a factor
            factor = gtsam.BetweenFactorPose2(robot_to_symbol(robot_id, i),
                                              robot_to_symbol(robot_id, i + 1),
                                              pose_between,
                                              self.prior_model)  # TODO update noise model
            graph.add(factor)

            # if we have a loop closure at this step, then there is no need to add another intitial guess
            if i not in self.multi_robot_frames[robot_id]:
                # if we have solved for this pose before use that as an intitial guess
                if robot_id in self.partner_robot_state_estimates and i in self.partner_robot_state_estimates[robot_id]:
                    values.insert(robot_to_symbol(robot_id, i), self.partner_robot_state_estimates[robot_id][i])
                else:
                    values.insert(robot_to_symbol(robot_id, i), pose_i)

        # initial guess for last frame
        if i + 1 not in self.multi_robot_frames[robot_id]:
            # if we have solved for this pose before use that as an intitial guess
            if robot_id in self.partner_robot_state_estimates and i + 1 in self.partner_robot_state_estimates[robot_id]:
                values.insert(robot_to_symbol(robot_id, i + 1), self.partner_robot_state_estimates[robot_id][i + 1])
            else:
                values.insert(robot_to_symbol(robot_id, i + 1), pose_i_plus_1)

        isam.update(graph, values)

        return isam

    def merge_slam(self, loop_closures: list, robust=False) -> None:
        """Add any multi-robot loop closures. 

        Args:
            loop_closures (list): A list of multi-robot loop closures
        """

        for loop in loop_closures:

            if loop.target_robot_id in self.multi_robot_frames:
                if loop.target_key in self.multi_robot_frames[loop.target_robot_id]:
                    continue

            self.icp_success_count += 1

            # parse some info
            source_symbol = X(loop.source_key)
            target_symbol = robot_to_symbol(loop.target_robot_id, loop.target_key)
            noise_model = self.create_noise_model(self.prior_sigmas)  #TODO update noise model
            if loop.method == "alcs":
                print("using robust noise model")
                noise_model = self.create_robust_noise_model(self.prior_sigmas)

            '''if loop.method == "alcs":
                print(self.slam_step)
                print(loop.source_robot_id,loop.target_robot_id,loop.source_key,loop.target_key)
                print(loop.estimated_transform)
                temp_source = numpy_to_gtsam(self.state_estimate[loop.source_key])
                temp_target = self.partner_robot_state_estimates[loop.target_robot_id][loop.target_key]
                test_transform = temp_source.between(temp_target)
                print(test_transform)
                print("---------")'''

            # build a factor and add it
            factor = gtsam.BetweenFactorPose2(source_symbol,
                                              target_symbol,
                                              loop.estimated_transform,
                                              noise_model)

            self.graph.add(factor)

            # Check if we have ever added a frame from this robot
            if loop.target_robot_id not in self.multi_robot_frames:
                self.multi_robot_frames[loop.target_robot_id] = {}
                self.partner_robot_state_estimates[loop.target_robot_id] = {}
                self.partner_robot_covariance[loop.target_robot_id] = {}

            # Check if we have added this particular frame from this robot 
            if loop.target_key not in self.multi_robot_frames[loop.target_robot_id]:
                self.multi_robot_frames[loop.target_robot_id][loop.target_key] = True
                self.values.insert(target_symbol, loop.target_pose)  # add the initial guess

        # see if we need to set the reference frame for this partner robot
        if loop.target_robot_id not in self.partner_reference_frames:
            self.partner_reference_frames[loop.target_robot_id] = loop.target_pose.compose(
                loop.target_pose_their_frame.inverse())

        self.inter_robot_loop_closures += loop_closures
        self.update_graph()  # upate the graph with the new info\
        # write_g2o(self.isam.getFactorsUnsafe(), self.isam.calculateEstimate(), "multi_robot.g2o")

    def update_partner_trajectory(self, robot_id: int, trajectory: np.array) -> int:
        """Update the trajectory. This is from the other robot performing SLAM.
        This update is performed by one robot sending it's trajctory to another. 

        Args:
            robot_id (int): the id of the robot
            trajectory (np.array): the trajectory from the robot

        Returns:
            int: comms cost of sending this data, return 0 if it is not sent
        """

        # check if we actually need to communicate this information
        flag = False
        for i, row in enumerate(trajectory):
            pose = numpy_to_gtsam(row)
            if robot_id in self.partner_robot_trajectories and i < len(self.partner_robot_trajectories[robot_id]):
                pose_two = numpy_to_gtsam(self.partner_robot_trajectories[robot_id][i])
                pose_between = pose.between(pose_two)
                if np.sqrt(pose_between.x() ** 2 + pose_between.y() ** 2) > 0.1 or np.degrees(
                        pose_between.theta()) > 3.0:
                    flag = True

        self.partner_robot_trajectories[robot_id] = trajectory  # pass through the info
        if flag:
            return len(trajectory) * 2 * 32  # return the cost if we actually need to send it
        else:
            return 0

    def find_poses_needing_loops(self, robot: int) -> None:
        """Find poses that need loop closures

        Args:
            robot (int): the robot we want to search with
        """

        self.poses_needing_loops[robot] = []  # clear before every step
        for i in self.partner_robot_covariance[robot]:  # loop over all the covariance matricies for this robot
            det = np.linalg.det(self.partner_robot_covariance[robot][i])
            if det > self.min_uncertainty:  # .0005: # check the determinant
                self.poses_needing_loops[robot].append(i)

    def search_for_possible_loops(self, robot: int) -> None:
        """Check for possible loop closures between robots. Only check poses that
        actually need a loop closure.

        Args:
            robot (int): the robot we want to check for possible loop closures with. 
        """

        self.possible_loops[robot] = []  # clear before every step
        for i, (_, pose) in enumerate(zip(self.points, self.state_estimate)):  # loop over my poses
            pose = numpy_to_gtsam(pose)
            for j in self.poses_needing_loops[robot]:  # loop over the poses that need loops
                # check if we have this pair as a loop closure already            
                if robot in self.multi_robot_frames and j in self.multi_robot_frames[robot]: continue
                pose_between = pose.between(self.partner_robot_state_estimates[robot][j])  # get the distance between
                if abs(pose_between.x()) <= 6 and abs(pose_between.y()) <= 6 and abs(
                        np.degrees(pose_between.theta())) < 50:
                    self.possible_loops[robot].append([i, j])  # log if small enough

    def simulate_loop_closure(self, robot: int) -> None:
        """Simulate the impact of the possible loop closures in self.possible loops. 
        Keep only the best outcome. 

        Args:
            robot (int): the robot we want to search with
        """

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        ratio_list = []
        loop_list = []
        for (source_key, target_key) in self.possible_loops[robot]:

            # check if we have ever tested this with ICP before
            if (source_key, robot, target_key) in self.tested_loops: continue

            # copy of isam combined
            isam = gtsam.ISAM2(self.isam_combined)

            # get the covariance at this pose before we insert the hypothetical loop closure
            cov_before = isam.marginalCovariance(robot_to_symbol(robot, target_key))

            # get the transform and package it
            one = numpy_to_gtsam(self.state_estimate[source_key])
            two = self.partner_robot_state_estimates[robot][target_key]
            pose_between = one.between(two)
            factor = gtsam.BetweenFactorPose2(X(source_key),
                                              robot_to_symbol(robot, target_key),
                                              pose_between,
                                              self.prior_model)  # TODO update noise model
            # insert and update isam 
            graph.add(factor)
            isam.update(graph, values)
            graph.resize(0)  # clear the graph and values once we push it to ISAM2
            values.clear()

            # get the covariance at this pose after we add the loop closure
            cov_after = isam.marginalCovariance(robot_to_symbol(robot, target_key))

            # get the ratio of the determinants to grade the impact of this loop closure
            ratio = np.linalg.det(cov_after) / np.linalg.det(cov_before)
            ratio_list.append(ratio)
            loop_list.append((source_key, target_key))

        self.best_possible_loops[robot] = None
        if len(ratio_list) > 0:
            self.best_possible_loops[robot] = loop_list[np.argmin(ratio_list)]
            i, j = loop_list[np.argmin(ratio_list)]
            self.tested_loops[(i, robot, j)] = True  # log

    def pass_data_cost(self) -> list:
        """Pass the cost of my most recent point cloud to 
        the rest of my team. 

        Returns:
            list: tthe costs of my point clouds up to now
        """

        if self.my_exchange_costs is None:
            self.my_exchange_costs = []
            for row in self.points:
                self.my_exchange_costs.append(len(row) * 2 * 32)
        return self.my_exchange_costs[:self.slam_step + 1]

    def update_partner_costs(self, robot_id: int, comms_cost: list):
        """Update the cost of exchangeing a point cloud

        Args:
            robot_id (int): the robot id 
            comms_cost (int): the cost to exchange this keyframe
        """

        self.partner_robot_exchange_costs[robot_id] = comms_cost

        # get the max comms cost we have ever seen. Look at ours, and what we have been sent
        max_list = []
        for robot in self.partner_robot_exchange_costs.keys():
            max_list.append(np.max(self.partner_robot_exchange_costs[robot]))
        if self.my_exchange_costs is not None:
            max_list.append(np.max(self.my_exchange_costs))

        # normalize the comms costs using the max, assuming the min would be zero
        max_val = np.max(max_list)
        for robot in self.partner_robot_exchange_costs.keys():
            self.partner_robot_exchange_costs[robot] = self.partner_robot_exchange_costs[robot] / max_val

    def run_metrics(self, path="") -> None:
        """Get and save some metrics about the mission 

        Args:
            mission(str): the mission we are running 
            mode (int): the mode the mission was in 
            study_step (int): the step in the study, the ith mission
        """

        euclidan_error = []
        rotational_error = []

        # euclidian error and rotation
        if self.truth is not None:
            truth_ref_frame = self.truth[0]
            for pose_est, pose_true in zip(self.state_estimate, self.truth):
                pose_est = numpy_to_gtsam(pose_est)
                pose_true = truth_ref_frame.between(pose_true)
                diff = pose_est.between(pose_true)
                euclidan_error.append(np.sqrt(diff.x() ** 2 + diff.y() ** 2))
                rotational_error.append(np.degrees(diff.theta()))
            euclidan_error = np.array(euclidan_error)
            rotational_error = np.array(rotational_error)
            self.mse = np.mean(euclidan_error ** 2)
            self.rmse = np.sqrt(np.mean(euclidan_error ** 2))

            # How good are we are estimating our partners location
            for robot in self.partner_robot_state_estimates.keys():

                euclidan_error = []
                rotational_error = []

                truth_ref_frame = self.truth[0]
                for step in self.partner_robot_state_estimates[robot].keys():
                    pose_est = self.partner_robot_state_estimates[robot][step]
                    pose_true = self.partner_truth[robot][step]
                    pose_true = truth_ref_frame.between(pose_true)
                    diff = pose_est.between(pose_true)
                    euclidan_error.append(np.sqrt(diff.x() ** 2 + diff.y() ** 2))
                    rotational_error.append(np.degrees(diff.theta()))
                euclidan_error = np.array(euclidan_error)
                rotational_error = np.array(rotational_error)
                # print(np.mean(euclidan_error), np.sqrt(np.mean(euclidan_error**2)))

        # uncertainty
        team_uncertainty = {}
        for robot in self.partner_robot_state_estimates.keys():
            temp = []
            counter = []
            for i in range(len(self.partner_robot_covariance[robot])):
                temp.append(np.linalg.det(self.partner_robot_covariance[robot][i]))
                counter.append(i)
            team_uncertainty[robot] = (counter, temp)
        self.team_uncertainty = team_uncertainty

        # print(self.mse,self.rmse)

        data_log = {}
        data_log["mse"] = self.mse
        data_log["rmse"] = self.rmse
        data_log["covariance"] = self.partner_robot_covariance
        data_log["my_covariance"] = self.covariance
        data_log["icp_count"] = self.icp_count
        data_log["icp_success_count"] = self.icp_success_count
        data_log["pcm_run_time"] = self.pcm_run_time
        data_log["alcs_reg_time"] = self.alcs_reg_time
        data_log["alcs_run_time"] = self.alcs_run_time
        data_log["draco_reg_time"] = self.draco_reg_time
        with open(f"{path}test_data/{self.mission}_{self.robot_id}.pickle",
                  'wb') as handle:
            pickle.dump(data_log, handle)

    def animate_step(self, path) -> None:

        plt.clf()
        fig, ax = plt.subplots()

        # title
        # plt.title("ROBOT: " + str(self.robot_id))

        # my own trajectory
        plt.plot(self.state_estimate[:, 1], self.state_estimate[:, 0], c="black")

        # w = Wedge((self.state_estimate[-1][1],self.state_estimate[-1][0]),5,theta-65,theta+65)
        # ax.add_artist(w)
        # w.set_facecolor("black")

        # plot the partner robot trajectory
        colors = {1: "blue", 2: "purple", 3: "orange"}
        for robot in self.partner_robot_state_estimates.keys():
            temp = []
            for frame in sorted(self.partner_robot_state_estimates[robot].keys()):
                pose = self.partner_robot_state_estimates[robot][frame]
                temp.append([pose.y(), pose.x()])
            temp = np.array(temp)
            plt.plot(temp[:, 0], temp[:, 1], c=colors[robot])

        # plot inter robot loop closures
        for loop in self.inter_robot_loop_closures:
            one = numpy_to_gtsam(self.state_estimate[loop.source_key])
            if loop.target_key >= len(self.partner_robot_state_estimates[loop.target_robot_id]): continue
            two = self.partner_robot_state_estimates[loop.target_robot_id][loop.target_key]
            plt.plot([one.y(), two.y()], [one.x(), two.x()], c="cyan", zorder=5)

        # ground truth as dotted line
        if self.truth is not None:
            truth_zero = self.truth[0]
            est_zero = numpy_to_gtsam(self.state_estimate[0])
            truth_in_my_frame = []
            for row in self.truth[:len(self.state_estimate)]:
                between = truth_zero.between(row)
                temp = est_zero.compose(between)
                truth_in_my_frame.append([temp.y(), temp.x()])
            truth_in_my_frame = np.array(truth_in_my_frame)
            plt.plot(truth_in_my_frame[:, 0], truth_in_my_frame[:, 1], c="black", linestyle='dashed')

            # ground truth of partner robots
            for robot in self.partner_truth:
                temp = est_zero.compose(truth_zero.inverse())
                temp = temp.compose(self.partner_truth[robot][0])
                plot_list = []
                for row in self.partner_truth[robot]:
                    between = self.partner_truth[robot][0].between(row)
                    plot_pose = temp.compose(between)
                    plot_list.append([plot_pose.x(), plot_pose.y()])
                plot_list = np.array(plot_list)
                plt.plot(plot_list[:, 1], plot_list[:, 0], c=colors[robot], linestyle='dashed')

        # draw the point clouds
        for cloud, pose in zip(self.points, self.state_estimate):
            cloud = transform_points_gtsam(cloud, numpy_to_gtsam(pose))
            plt.scatter(cloud[:, 1], cloud[:, 0], c="black", s=5)

        for robot in self.possible_loops.keys():
            if len(self.possible_loops[robot]) == 0: continue
            for (i, j) in self.possible_loops[robot]:
                if i >= len(self.state_estimate): continue
                one = numpy_to_gtsam(self.state_estimate[i])
                if j >= len(self.partner_robot_state_estimates[robot]): continue
                two = self.partner_robot_state_estimates[robot][j]
                plt.plot([one.y(), two.y()], [one.x(), two.x()], c="green")

        '''if len(self.best_possible_loops) != 0:
            i, r, j = self.best_possible_loops
            one = numpy_to_gtsam(self.state_estimate[i])
            two = self.partner_robot_state_estimates[r][j]
            plt.plot([one.y(),two.y()],[one.x(),two.x()],c="purple")'''

        # draw the covariance matrix
        for robot in self.partner_robot_covariance:
            for i in self.partner_robot_covariance[robot]:
                pose = self.partner_robot_state_estimates[robot][i]
                cov = self.partner_robot_covariance[robot][i]
                det = np.linalg.det(cov)
                sigma_x = np.sqrt(cov[0][0])
                sigma_y = np.sqrt(cov[1][1])
                '''e = Ellipse(xy=(pose.y(),pose.x()),width=sigma_y, height=sigma_x, angle=pose.theta())
                ax.add_artist(e)

                if det > 0.005:
                    e.set_facecolor("red")
                else:
                    e.set_facecolor("black")'''

        if self.mode == 1:
            mode_name = "ALCS"
        else:
            mode_name = "DRACO"

        plt.axis("square")
        plt.xticks([])
        plt.yticks([])
        folder = path + str(self.mission) + "/" + str(mode_name) + "/" + str(self.robot_id) + "/"
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{folder}{self.slam_step}.png")
        plt.clf()
        plt.close()

    def plot(self) -> None:
        """Visulize the mission
        """

        self.run_metrics()

        '''for robot in self.team_uncertainty:
            plt.plot(self.team_uncertainty[robot][0],self.team_uncertainty[robot][1])
        plt.show()'''

        '''for loop in self.possible_loops:
            i, r, j = loop
            if i >= len(self.state_estimate): continue
            one = numpy_to_gtsam(self.state_estimate[i])
            if j >= len(self.partner_robot_state_estimates[r]): continue
            two = self.partner_robot_state_estimates[r][j]
            source_points = transform_points(self.points[i],one)
            target_points = transform_points(self.dummy_cloud,two)

            for cloud,pose in zip(self.points,self.state_estimate):
                cloud = transform_points(cloud,numpy_to_gtsam(pose))
                plt.scatter(cloud[:,1],cloud[:,0],c="black",s=5)

            for robot in self.partner_robot_state_estimates.keys():
                temp = []
                for frame in sorted(self.partner_robot_state_estimates[robot].keys()):
                    pose = self.partner_robot_state_estimates[robot][frame]
                    temp.append([pose.y(),pose.x()])
                temp = np.array(temp)
                plt.plot(temp[:,0],temp[:,1])
            
            plt.plot(self.state_estimate[:,1],self.state_estimate[:,0],c="black")
            plt.scatter(source_points[:,1],source_points[:,0],c="blue",zorder=2)
            plt.scatter(target_points[:,1],target_points[:,0],c="red",zorder=0)
            plt.plot([one.y(),two.y()],[one.x(),two.x()],c="purple")
            plt.axis("square")
            plt.show()'''

        # my own trajectory
        '''plt.plot(self.state_estimate[:,1],self.state_estimate[:,0],c="black")
        plt.title("ROBOT: " + str(self.robot_id))

        # plot the partner robot trajectory
        for robot in self.partner_robot_state_estimates.keys():
            temp = []
            for frame in sorted(self.partner_robot_state_estimates[robot].keys()):
                pose = self.partner_robot_state_estimates[robot][frame]
                temp.append([pose.y(),pose.x()])
            temp = np.array(temp)
            plt.plot(temp[:,0],temp[:,1])

        truth_zero = self.truth[0]
        est_zero = numpy_to_gtsam(self.state_estimate[0])
        truth_in_my_frame = []
        for row in self.truth[:len(self.state_estimate)]:
           between = truth_zero.between(row)
           temp = est_zero.compose(between)
           truth_in_my_frame.append([temp.y(),temp.x()])
        truth_in_my_frame = np.array(truth_in_my_frame)
        plt.plot(truth_in_my_frame[:,0],truth_in_my_frame[:,1],c="black",linestyle='dashed')

        for cloud,pose in zip(self.points,self.state_estimate):
            cloud = transform_points(cloud,numpy_to_gtsam(pose))
            plt.scatter(cloud[:,1],cloud[:,0],c="black",s=5)

        for loop in self.inter_robot_loop_closures:
            one = numpy_to_gtsam(self.state_estimate[loop.source_key])
            if loop.target_key >= len(self.partner_robot_state_estimates[loop.target_robot_id]): continue
            two = self.partner_robot_state_estimates[loop.target_robot_id][loop.target_key]
            plt.plot([one.y(),two.y()],[one.x(),two.x()],c="red")

        for loop in self.loops_tested:
            i, r, j = loop
            if i >= len(self.state_estimate): continue
            one = numpy_to_gtsam(self.state_estimate[i])
            if j >= len(self.partner_robot_state_estimates[r]): continue
            two = self.partner_robot_state_estimates[r][j]
            # plt.plot([one.y(),two.y()],[one.x(),two.x()],c="purple")

        for loop in self.possible_loops:
            i, r, j = loop
            if i >= len(self.state_estimate): continue
            one = numpy_to_gtsam(self.state_estimate[i])
            if j >= len(self.partner_robot_state_estimates[r]): continue
            two = self.partner_robot_state_estimates[r][j]
            plt.plot([one.y(),two.y()],[one.x(),two.x()],c="purple")


        plt.axis("square")
        plt.show()'''

        '''for robot in self.partner_robot_covariance.keys():
            temp = []
            temp_2 = []
            i = 0
            for row in self.partner_robot_covariance[robot].keys():
                row = self.partner_robot_covariance[robot][row]
                temp.append(np.linalg.det(row))
                temp_2.append(i)
                i += 1
            plt.plot(temp_2,temp)
        plt.show()'''

    def get_graph(self):
        return self.object_detection.get_graph()

    def get_keyframes(self, keyframe_id_set):
        msgs = {}
        for keyframe_id in keyframe_id_set:
            pose_array = self.state_estimate[keyframe_id, :]
            covariance = self.covariance[keyframe_id, :, :]
            if self.truth is not None:
                pose_array_truth = self.truth[keyframe_id]
            else:
                pose_array_truth = None
            msg = RobotMessage(robot_id=self.robot_id,
                               keyframe_id=keyframe_id,
                               pose=pose_array,
                               covariance=covariance,
                               pose_truth=pose_array_truth)
            msgs[keyframe_id] = msg
        return msgs

    def receive_graph_from_neighbor(self, neighbor_id, graph):
        self.object_detection.graphs_neighbor[neighbor_id] = graph

    def receive_keyframes_from_neighbor(self, neighbor_id, ids, keyframes):
        self.loop_closure_manager.add_keyframes_neighbor(neighbor_id, ids, keyframes)
        try:
            self.loop_closure_manager.transformations_neighbor[neighbor_id] = (
                self.object_detection.transformations_neighbor)[neighbor_id]
        except:
            print("No transformation found for neighbor in object detection")

        return self.loop_closure_manager.update_keyframe_poses_transformed(neighbor_id, self.state_estimate)

    def perform_graph_match(self):
        keyframe_id_self, keyframe_id_to_request = self.object_detection.compare_all_neighbor_graph()
        for robot_id in keyframe_id_to_request.keys():
            if self.update_poses:
                if robot_id in self.loop_closure_manager.poses_id_neighbor:
                    keyframe_id_to_request[robot_id] = (
                        keyframe_id_to_request[robot_id].union(self.loop_closure_manager.poses_id_neighbor[robot_id]))
            else:
                keyframe_id_to_request[robot_id] = (
                    self.loop_closure_manager.get_ids_pose_not_received(robot_id, keyframe_id_to_request[robot_id]))
        return keyframe_id_self, keyframe_id_to_request

    def get_scans(self, robot_id_to, keyframe_id_set):
        return self.loop_closure_manager.get_keyframes(robot_id_to, keyframe_id_set)

    def receive_scans_from_neighbor(self, neighbor_id, ids, keyframes):
        if len(keyframes) == 0:
            return
        self.loop_closure_manager.add_scans_neighbor(neighbor_id, ids, keyframes)
        self.loop_closure_manager.perform_icp(neighbor_id, self.state_estimate)

    def perform_gcm(self):
        time0 = time.time()
        loops = self.loop_closure_manager.perform_pcm(self.state_estimate)
        time1 = time.time()
        self.pcm_run_time.append([time1 - time0])
        self.icp_success_count = len(self.loop_closure_manager.loops_added)
        return loops

    def add_inter_robot_loop_closure(self, loop_closures):
        neighbor_robot_state_to_update = set()
        for loop in loop_closures:
            source_ch = chr(96 + loop.robot_id)
            source_id = loop.source_keyframe_id
            target_id = loop.target_keyframe_id
            source_key = gtsam.symbol(source_ch, source_id)
            target_key = X(target_id)

            # part 1: add between factors
            between_pose = gtsam.Pose2(loop.between_pose[0],
                                       loop.between_pose[1],
                                       loop.between_pose[2])
            if self.loop_closure_manager.noise_model_type == 0:
                noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array(self.prior_sigmas))
            elif self.loop_closure_manager.noise_model_type == 1:
                noise_model = gtsam.noiseModel.Gaussian.Covariance(loop.cov)
            else:
                noise_model = self.create_robust_noise_model(self.prior_sigmas)

            factor_btwn = gtsam.BetweenFactorPose2(source_key, target_key, between_pose, noise_model)
            self.graph.add(factor_btwn)

            source_pose_msg = self.loop_closure_manager.poses_neighbor[loop.robot_id][source_id]
            if source_key not in self.loop_closure_manager.robot_key_added:
                # add initial value for inter-robot loop closure
                self.loop_closure_manager.robot_key_added.add(source_key)
                # target_pose = gtsam.Pose2(self.state_estimate[target_id][0],
                #                           self.state_estimate[target_id][1],
                #                           self.state_estimate[target_id][2])
                # self.values.insert(source_key,
                #                    target_pose.compose(between_pose.inverse()))
                source_pose = gtsam.Pose2(source_pose_msg.pose_transformed[0],
                                          source_pose_msg.pose_transformed[1],
                                          source_pose_msg.pose_transformed[2])
                self.values.insert(source_key, source_pose)

                if self.loop_closure_manager.inter_factor_type == 0:
                    self.add_inter_robot_priori_factor(loop)
                elif self.loop_closure_manager.inter_factor_type == 1:
                    self.add_inter_robot_between_facter(loop)
                else:
                    neighbor_robot_state_to_update.add(loop.robot_id)
        if neighbor_robot_state_to_update:
            for robot_id in neighbor_robot_state_to_update:
                self.add_inter_robot_whole_graph(robot_id)
        self.update_graph()

    def add_inter_robot_priori_factor(self, loop):
        source_ch = chr(96 + loop.robot_id)
        source_id = loop.source_keyframe_id
        source_key = gtsam.symbol(source_ch, source_id)
        source_keyframe = self.loop_closure_manager.poses_neighbor[loop.robot_id][source_id]
        source_pose = source_keyframe.pose
        prior_pose = gtsam.Pose2(source_pose[0], source_pose[1], source_pose[2])
        priori_factor = gtsam.PriorFactorPose2(source_key, prior_pose, source_keyframe.covariance_transformed)
        self.graph.add(priori_factor)

    def add_inter_robot_between_facter(self, loop):
        if loop.robot_id not in self.source_key_before:
            # no between factor for the first keyframe
            self.source_key_before[loop.robot_id] = loop.source_keyframe_id
            return
        # between factor key 0
        source_ch = chr(96 + loop.robot_id)
        source_0_id = self.source_key_before[loop.robot_id]
        source_0_key = gtsam.symbol(source_ch, source_0_id)
        source_0_pose_msg = self.loop_closure_manager.poses_neighbor[loop.robot_id][source_0_id]
        source_0_pose = source_0_pose_msg.pose
        # between factor key 1
        source_1_id = loop.source_keyframe_id
        source_1_key = gtsam.symbol(source_ch, source_1_id)
        source_1_pose_msg = self.loop_closure_manager.poses_neighbor[loop.robot_id][source_1_id]
        source_1_pose = source_1_pose_msg.pose
        # construct between pose
        source_btwn = gtsam.Pose2(source_0_pose[0], source_0_pose[1], source_0_pose[2]).between(
            gtsam.Pose2(source_1_pose[0], source_1_pose[1], source_1_pose[2]))
        # propagate the noise model
        cov0 = source_0_pose_msg.covariance
        cov1 = source_1_pose_msg.covariance
        cov_btwn = compute_covariance(state_to_matrix(source_0_pose), state_to_matrix(source_1_pose), cov0, cov1)
        source_noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_btwn)
        factor_btwn = gtsam.BetweenFactorPose2(source_0_key, source_1_key,
                                               source_btwn, source_noise_model)
        # update the previous key
        self.source_key_before[loop.robot_id] = loop.source_keyframe_id
        self.graph.add(factor_btwn)

    def get_factors_current(self):
        return self.factors[self.slam_step]

    def receive_factors_from_neighbor(self, robot_id, factors):
        if robot_id not in self.loop_closure_manager.factors_neighbor.keys():
            self.loop_closure_manager.factors_neighbor[robot_id] = []
        self.loop_closure_manager.factors_neighbor[robot_id].append(factors)

        # If do the whole graph update, update after initial inter robot loop closures are added
        if self.loop_closure_manager.inter_factor_type == 2 and robot_id in self.loop_closure_manager.neighbor_added:
            self.add_inter_robot_whole_graph(robot_id)

    def receive_latest_states_from_neighbor(self, robot_id, states):
        self.partner_robot_trajectories[robot_id] = states

    def list2factor(self, factor, robot_id):
        i, j, transform, sigmas = factor
        sigmas = np.array(sigmas)

        if sigmas.shape == (3,):
            noise_model = self.create_noise_model(sigmas)
        else:
            noise_model = self.create_full_noise_model(sigmas)
        Xi = gtsam.symbol(chr(96 + robot_id), i)
        Xj = gtsam.symbol(chr(96 + robot_id), j)

        return gtsam.BetweenFactorPose2(Xi, Xj, transform, noise_model)

    def add_to_values(self, robot_id, keyframe_id):
        key = gtsam.symbol(chr(96 + robot_id), keyframe_id)
        if key in self.loop_closure_manager.robot_key_added:
            return
        pose = state_to_matrix(self.partner_robot_trajectories[robot_id][keyframe_id, :])
        try:
            R, t = self.loop_closure_manager.get_transformation(robot_id)
            transformation_matrix = np.eye(3)
            transformation_matrix[:2, :2] = R
            transformation_matrix[:2, 2] = t
        except:
            return
        pose = matrix_to_state(transformation_matrix @ pose)
        self.values.insert(key, numpy_to_gtsam(pose))
        self.loop_closure_manager.robot_key_added.add(key)

    def add_inter_robot_whole_graph(self, robot_id):
        # step 1: add factors from neighbor
        while self.loop_closure_manager.factors_neighbor[robot_id]:
            factors = self.loop_closure_manager.factors_neighbor[robot_id].pop(0)
            for factor in factors:
                self.graph.add(self.list2factor(factor, robot_id))
                self.add_to_values(robot_id, factor[0])
                self.add_to_values(robot_id, factor[1])

    def plot_figure(self, path):
        plt.figure(figsize=(8, 8), dpi=150)
        # self.object_detection.plot_figure()

        plt.plot(self.state_estimate[:, 0], self.state_estimate[:, 1], 'c-')
        plt.scatter(self.object_detection.points[:, 0], self.object_detection.points[:, 1], s=1, c='k')
        values = self.isam.calculateEstimate()

        for loop in self.loop_closure_manager.loops_added:
            robot_ns = loop.robot_id
            if robot_ns == 1:
                color = 'blue'
            elif robot_ns == 2:
                color = 'red'
            else:
                color = 'orange'
            id = loop.source_keyframe_id
            points = self.loop_closure_manager.points_neighbor[robot_ns][id].points
            pose = values.atPose2(gtsam.symbol(chr(96 + robot_ns), id))
            points = transform_points(points, np.array([pose.x(), pose.y(), pose.theta()]))
            plt.scatter(pose.x(), pose.y(), c=color, s=20)
            plt.scatter(points[:, 0], points[:, 1], s=1, c=color)

        plt.xlim([-80, 80])
        plt.ylim([-80, 80])
        plt.axis('equal')

        plt.savefig(path + 'test_data/' + str(self.robot_id) + '/' + str(self.slam_step) + '.png')
        plt.close()

    def prepare_loops_to_send(self, loops):
        return self.loop_closure_manager.prepare_loops_to_send(loops)

    def check_factor_status(self, robot_id, key):
        if key >= self.partner_robot_trajectories[robot_id].shape[0]:
            return False
        if self.loop_closure_manager.get_transformation(robot_id) is None:
            return False
        return True

    def receive_loops_from_neighbor(self, loop_msgs):
        if not self.loop_closure_manager.exchange_inter_robot_factor:
            return
        robot_neighbors = set()
        loop_msgs.extend(self.loop_closure_manager.inter_loops_to_add)
        for loop_msg in loop_msgs:
            ch0 = chr(96 + loop_msg['r0'])
            ch1 = chr(96 + loop_msg['r1'])

            id0 = loop_msg['key0']
            id1 = loop_msg['key1']

            pose = loop_msg['pose']
            cov = loop_msg['cov']
            if loop_msg['r0'] == self.robot_id:
                key0 = X(id0)
            elif not self.check_factor_status(loop_msg['r0'], id0):
                self.loop_closure_manager.inter_loops_to_add.append(loop_msg)
                continue
            else:
                key0 = gtsam.symbol(ch0, id0)
                robot_neighbors.add(loop_msg['r0'])

            if loop_msg['r1'] == self.robot_id:
                key1 = X(id1)
            elif not self.check_factor_status(loop_msg['r1'], id1):
                self.loop_closure_manager.inter_loops_to_add.append(loop_msg)
                continue
            else:
                key1 = gtsam.symbol(ch1, id1)
                robot_neighbors.add(loop_msg['r1'])

            if self.loop_closure_manager.noise_model_type == 0:
                noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array(self.prior_sigmas))
            elif self.loop_closure_manager.noise_model_type == 1:
                noise_model = gtsam.noiseModel.Gaussian.Covariance(cov)
            else:
                noise_model = self.create_robust_noise_model(self.prior_sigmas)

            factor = gtsam.BetweenFactorPose2(key0, key1, gtsam.Pose2(pose[0], pose[1], pose[2]), noise_model)
            self.graph.add(factor)

        for robot_id in robot_neighbors:
            self.add_inter_robot_whole_graph(robot_id)
        self.update_graph()

    def write_all_trajectories(self, path):
        # TODO: update partner_robot_state_estimate here
        if self.loop_closure_manager.inter_factor_type == 2:
            self.isam.update()
            self.isam.update()
            final_values = self.isam.calculateEstimate()
            for robot_id, robot_state in self.partner_robot_trajectories.items():
                self.partner_robot_state_estimates[robot_id] = np.zeros([robot_state.shape[0]-1, robot_state.shape[1]])
                for keyframe_id in range(self.partner_robot_trajectories[robot_id].shape[0]-1):
                    ch = chr(96 + robot_id)
                    key = gtsam.symbol(ch, keyframe_id)
                    if key in final_values.keys():
                        pose = final_values.atPose2(key)
                        self.partner_robot_state_estimates[robot_id][keyframe_id] = (
                            np.array([pose.x(), pose.y(), pose.theta()]))

        # record self trajectory
        robot_self_file_path = f"{path}test_data/{self.mission}_{self.robot_id}_self.txt"
        state_estimate = self.state_estimate
        for robot_id in sorted(self.partner_robot_state_estimates.keys()):
            state_estimate = np.concatenate((state_estimate, self.partner_robot_state_estimates[robot_id]))
        write_trajectory(robot_self_file_path, state_estimate)
        if self.truth is not None:
            truth = self.truth[:self.slam_step+1]
            for robot_id in sorted(self.partner_truth.keys()):
                len_neighbor = self.partner_robot_trajectories[robot_id].shape[0] - 1
                truth.extend(self.partner_truth[robot_id][:len_neighbor])
            write_trajectory(f"{path}test_data/{self.mission}_{self.robot_id}_self_gt.txt", truth)

def write_g2o(graph, values, filename):
    factors = [graph.at(i) for i in range(graph.size())]
    lines = []
    for factor in factors:
        if isinstance(factor, gtsam.BetweenFactorPose2):
            keys = factor.keys()
            key0 = gtsam.Symbol(keys[0])
            ch0 = chr(key0.chr())
            id0 = key0.index()
            measure = factor.measured()
            key1 = gtsam.Symbol(keys[1])
            ch1 = chr(key1.chr())
            id1 = key1.index()
            cov = factor.noiseModel().covariance()
            lines.append(f'Between: {ch0}{id0} & {ch1}{id1}, ['
                         f'{measure.x():.2f} {measure.y():.2f} {measure.theta():.2f}], \n'
                         f'{cov}\n')
        if isinstance(factor, gtsam.PriorFactorPose2):
            keys = factor.keys()
            key = gtsam.Symbol(keys[0])
            ch = chr(key.chr())
            id = key.index()
            prior = factor.prior()
            cov = factor.noiseModel().covariance()
            lines.append(f'Prior: {ch}{id}, [{prior.x():.2f} {prior.y():.2f} {prior.theta():.2f}], \n'
                         f'{cov}\n')

    with open(filename, "w") as file:
        for line in lines:
            file.write(line)
