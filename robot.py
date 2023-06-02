from scipy.spatial import KDTree
from typing import Tuple
import numpy as np
import gtsam
import matplotlib.pyplot as plt
import time

from utils import get_all_context,get_points,verify_pcm, X, robot_to_symbol, numpy_to_gtsam,transform_points, check_frame_for_overlap

from loop_closure import LoopClosure

class Robot():
    def __init__(self,robot_id:int,data:dict,SUBMAP_SIZE:int,BEARING_BINS:int,RANGE_BINS:int,MAX_RANGE:int,MAX_BEARING:int):
        
        self.robot_id = robot_id # unique ID number

        self.slam_step = 0 # we want to track how far along the mission is, what SLAM step are we on
        self.total_steps = len(data["poses"])

        self.poses = data["poses"] # poses as numpy arrays
        self.poses_g = data["poses_g"] # poses from above but as gtsam.Pose2
        self.points = data["points"]  # raw points at each pose
        self.points_t = data["points_t"] # transformed points at each pose
        self.truth = data["truth"] # ground truth for each pose
        self.factors = data["factors"] # the factors in the graph

        # get the scan context images and ring keys
        self.keys, self.context = get_all_context(self.poses,
                                                            self.points,
                                                            SUBMAP_SIZE,
                                                            BEARING_BINS,
                                                            RANGE_BINS,
                                                            MAX_RANGE,
                                                            MAX_BEARING)
        
        self.submap_size = SUBMAP_SIZE
        self.pcm_dict = {}
        self.pcm_queue_size = 5
        self.min_pcm = 2

        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.values_added = {}
        self.isam_params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(self.isam_params)
        self.isam_combined = None
        self.state_estimate = [] # my own state estimate
        self.covariance = []
        self.partner_robot_state_estimates = {} # my estimates about the other robots
        self.partner_robot_trajectories = {} # the trajectory each robot sends to me, they figure this out
        self.partner_reference_frames = {} # my estimate of where I think the this other robot started
        self.multi_robot_frames = {} # the frames I have added as loop closures from other robots
        self.partner_robot_covariance = {}
        self.partner_robot_exchange_costs = {}
        self.my_exchange_costs = None

        self.prior_sigmas = [0.1, 0.1, 0.01]
        self.prior_model = self.create_noise_model(self.prior_sigmas)

        self.inter_robot_loop_closures = []

        self.point_clouds_received = {} # log when we have gotten a point cloud. Format: robot_id_number, keyframe_id

        self.possible_loops = {}
        self.best_possible_loops = []

        self.mse = None
        self.rmse = None
        self.team_uncertainty = []
        
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
    
    def step(self) -> None:
        """Increase the step of SLAM
        """

        if self.slam_step == 0:
            self.start_graph()
            self.update_graph()
        self.slam_step += 1
        self.add_factors()
        self.search_for_possible_loops()
        self.update_graph()
        self.simulate_loop_closure()

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
            i,j,transform,sigmas = factor
            sigmas = np.array(sigmas)

            if sigmas.shape == (3,): 
                noise_model = self.create_noise_model(sigmas)
            else: 
                noise_model = self.create_full_noise_model(sigmas)

            factor_gtsam = gtsam.BetweenFactorPose2(X(i),X(j),transform,noise_model)

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
            return self.keys[-1], len(self.keys)-1
        return self.keys[self.slam_step], self.slam_step
    
    def get_context(self) -> np.array:
        """Get the current scan context image

        Returns:
            np.array: the current or last scan context image
        """

        if self.slam_step >= len(self.context):
            return self.context[-1]
        return self.context[self.slam_step]
    
    def get_context_index(self,index:int) -> np.array:
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
    
    def get_robot_points(self,index:int,submap_size:int) -> np.array:
        """Get the submap at an index given the submap size. 
        Use utils.py get_points. 

        Args:
            index (int): the index we want the points at
            submap_size (int): the size of the submap in each direction

        Returns:
            np.array: the points ready for registration
        """

        return get_points(index,submap_size,self.points,self.poses)
    
    def get_pose_gtsam(self) -> gtsam.Pose2:
        """Return the pose at the current step

        Returns:
            gtsam.Pose2: _description_
        """
    
        if self.slam_step >= len(self.poses_g):return self.poses_g[-1]
        return self.poses_g[self.slam_step]
    
    def add_loop_to_pcm_queue(self, loop:LoopClosure) -> None:
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
        pcm_queue.append(loop) # add the loop
        self.pcm_dict[loop.target_robot_id] = pcm_queue # store the pcm queue 
        
    def do_pcm(self,robot_id:int) -> list:
        """Check the self.pcm_queue for any valid loop closures. Return them as a list.

        Args:
            robot_id (int): the ID of the robot the most recent loop is with. This 
            is the queue we need to check with PCM. 

        Returns:
            list: a list of LoopClosures
        """

        assert(len(self.pcm_dict[robot_id]) != 0)

        valid_loops = []
        valid_indexes = verify_pcm(self.pcm_dict[robot_id],self.min_pcm)
        for i in valid_indexes: 
            if self.pcm_dict[robot_id][i].inserted == False:
                self.pcm_dict[robot_id][i].inserted = True
                valid_loops.append(self.pcm_dict[robot_id][i])
        return valid_loops
    
    def check_for_data(self, robot_id:int, keyframe_id:int) -> Tuple[list,int]:
        """Checks if we need any data exchange to complete the ICP call.
        returns a list of [[robot_id_number, keyframe_id]]

        Args:
            robot (int): robot id number we are calling icp with
            index (int): the keyframe index for the above robot

        Returns:
            list: a list of the point clouds we need to complete this job
            int: the cost of requesting this data
        """

        data_requests = [] # list of the data we need
        comms_cost = 0
        # check the whole submap to see if we need any of the point clouds
        for i in range(keyframe_id-self.submap_size,keyframe_id+self.submap_size+1):
            if (robot_id,i) not in self.point_clouds_received and i >= 0: # if we don't have it, ask for it
                data_requests.append([robot_id,i])
                self.point_clouds_received[(robot_id,i)] = True
                comms_cost += 32 + 32 # we only need two 32 bit integers
        return data_requests,comms_cost
    
    def get_data(self,data_request:list) -> int:
        """Find the communication cost of getting the above data

        Args:
            data_request (list): the list of robot keyframes we want

        Returns:
            int: the cost of exchanging this data
        """

        comms_cost = 0
        for row in data_request:
            _, i = row
            if i < len(self.points): # check for out of range
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
        isam = gtsam.ISAM2(self.isam) # make a copy   

        # add and update the partner robot trajectories
        for robot in self.partner_robot_state_estimates.keys():
            isam = self.merge_trajectory(isam,robot)
        self.isam_combined = isam

        # update the state estimate
        values = isam.calculateEstimate()

        # Update MY whole trajectory
        temp = []
        temp_2 = []
        for x in range(self.slam_step+1):
            pose = values.atPose2(X(x))
            cov = isam.marginalCovariance(X(x))
            temp.append([pose.x(),pose.y(),pose.theta()])
            temp_2.append(cov)
        self.state_estimate = np.array(temp)
        self.covariance = np.array(temp_2)

        # update my the estimate of my partner robots in my frame
        for robot in self.partner_robot_state_estimates.keys():
            if len(self.multi_robot_frames[robot]) == 0: continue
            for i in range(len(self.partner_robot_trajectories[robot])):
                self.partner_robot_state_estimates[robot][i] = values.atPose2(robot_to_symbol(robot,i))
                self.partner_robot_covariance[robot][i] = isam.marginalCovariance(robot_to_symbol(robot,i))

    def merge_trajectory(self,isam:gtsam.ISAM2,robot_id:int) -> gtsam.ISAM2:
        """Add the partner robot trajectory to the slam graph. Note that we use and return a copy
        of the isam instance. 

        Args:
            isam (gtsam.ISAM2): copy of the isam instance
            robot_id (int): the robot id we want to merge with

        Returns:
            gtsam.ISAM2: the update isam instance
        """

        # check if we can even perform a merge
        if len(self.multi_robot_frames[robot_id]) == 0: return isam # do we have any loop closures?
    
        # objects to update isam
        values = gtsam.Values()
        graph = gtsam.NonlinearFactorGraph()

        # add the whole trajectory
        for i in range(len(self.partner_robot_trajectories[robot_id]) - 1):
            pose_i = numpy_to_gtsam(self.partner_robot_trajectories[robot_id][i]) # get the poses in gtsam
            pose_i_plus_1 = numpy_to_gtsam(self.partner_robot_trajectories[robot_id][i+1])
            pose_i = self.partner_reference_frames[robot_id].compose(pose_i) # place in the correct ref frame
            pose_i_plus_1 = self.partner_reference_frames[robot_id].compose(pose_i_plus_1)
            pose_between = pose_i.between(pose_i_plus_1) # get the pose between them and package as a factor
            factor = gtsam.BetweenFactorPose2(robot_to_symbol(robot_id,i),
                                                robot_to_symbol(robot_id,i+1),
                                                pose_between,
                                                self.prior_model) # TODO update noise model
            graph.add(factor)

            # if we have a loop closure at this step, then there is no need to add another intitial guess
            if i not in self.multi_robot_frames[robot_id]:
                # if we have solved for this pose before use that as an intitial guess
                if robot_id in self.partner_robot_state_estimates and i in self.partner_robot_state_estimates[robot_id]:
                    values.insert(robot_to_symbol(robot_id,i), self.partner_robot_state_estimates[robot_id][i])
                else:
                    values.insert(robot_to_symbol(robot_id,i), pose_i)

        # initial guess for last frame
        if i+1 not in self.multi_robot_frames[robot_id]: 
            # if we have solved for this pose before use that as an intitial guess
            if robot_id in self.partner_robot_state_estimates and i+1 in self.partner_robot_state_estimates[robot_id]:
                values.insert(robot_to_symbol(robot_id,i+1), self.partner_robot_state_estimates[robot_id][i+1])
            else:
                values.insert(robot_to_symbol(robot_id,i+1), pose_i_plus_1)

        isam.update(graph, values)

        return isam
        
    def merge_slam(self,loop_closures:list) -> None:
        """Add any multi-robot loop closures. 

        Args:
            loop_closures (list): A list of multi-robot loop closures
        """

        for loop in loop_closures:
            
            # parse some info
            source_symbol = X(loop.source_key)
            target_symbol = robot_to_symbol(loop.target_robot_id,loop.target_key)
            noise_model = self.create_noise_model(self.prior_sigmas) #TODO update noise model

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
                self.values.insert(target_symbol, loop.target_pose) # add the initial guess

        # see if we need to set the reference frame for this partner robot
        if loop.target_robot_id not in self.partner_reference_frames:
            self.partner_reference_frames[loop.target_robot_id] = loop.target_pose.compose(
                                                                        loop.target_pose_their_frame.inverse())
        
        self.inter_robot_loop_closures += loop_closures
        self.update_graph() # upate the graph with the new info

    def update_partner_trajectory(self,robot_id:int,trajectory:np.array) -> int:
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
                if np.sqrt(pose_between.x()**2 + pose_between.y()**2) > 0.1 or np.degrees(pose_between.theta()) > 3.0:
                    flag = True

        self.partner_robot_trajectories[robot_id] = trajectory # pass through the info
        if flag: return len(trajectory) * 2 * 32 # return the cost if we actually need to send it
        else: return 0
        
    def search_for_possible_loops(self):
        """Check for possible loop closures between robots.
        """
                
        for j, (cloud, pose) in enumerate(zip(self.points,self.state_estimate)):
            pose = numpy_to_gtsam(pose)
            cloud = transform_points(cloud,pose) # place the cloud based on the most recent estimate
            for robot in self.partner_robot_state_estimates: 
                for i in self.partner_robot_state_estimates[robot]:
                    if (j,robot,i) in self.possible_loops: continue # check if we have a possible here already
                    # check if there is a possible loop between
                    # my pose i and partner robots jth pose
                    status = check_frame_for_overlap(cloud,
                                                     pose,
                                                     self.partner_robot_state_estimates[robot][i],
                                                     30)
                    if status: #if so log it
                        self.possible_loops[(j,robot,i)] = True

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
        return self.my_exchange_costs[:self.slam_step+1]
    
    def update_partner_costs(self,robot_id:int, comms_cost:list):
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


    def simulate_loop_closure(self):
        """Simulate the impact of the possible loop closures in self.possible loops. 
        Log the best K loop closures. 
        """
        
        # copy of isam combined
        isam = gtsam.ISAM2(self.isam_combined)
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        i = 0 # counter
        start_time = time.time() # timer
        ratio_list = []
        loop_list = []
        for (source_key, target_robot_id, target_key) in self.possible_loops:
            i += 1

            # get the covariance at this pose before we insert the hypothetical loop closure
            cov_before = isam.marginalCovariance(robot_to_symbol(target_robot_id,target_key))

            # get the transform and package it
            one = numpy_to_gtsam(self.state_estimate[source_key])
            two = self.partner_robot_state_estimates[target_robot_id][target_key]
            pose_between = one.between(two)
            factor = gtsam.BetweenFactorPose2(X(source_key),
                                                robot_to_symbol(target_robot_id,target_key),
                                                pose_between,
                                                self.prior_model) # TODO update noise model
            # insert and update isam 
            graph.add(factor)
            isam.update(graph, values)
            graph.resize(0)  # clear the graph and values once we push it to ISAM2
            values.clear()

            # get the covariance at this pose after we add the loop closure
            cov_after = isam.marginalCovariance(robot_to_symbol(target_robot_id,target_key))

            '''# get the cost of exchanging the data needed for this loop 
            # first check if we have the data on hand
            if (target_robot_id,target_key) in self.point_clouds_received:
                exchange_cost = 0
            else:
                exchange_cost = self.partner_robot_exchange_costs[target_robot_id][target_key]
                exchange_cost = 0'''

            # get the ratio of the determinants to grade the impact of this loop closure
            ratio = np.linalg.det(cov_after) / np.linalg.det(cov_before)
            ratio_list.append(ratio)
            loop_list.append([source_key,target_robot_id,target_key])

        ind = np.argpartition(ratio_list, 4)[:4]
        self.best_possible_loops = []
        if len(ind) > 0 : 
            for index in ind:
                self.best_possible_loops.append(loop_list[index])
        # print("Tested N Loops: ", i, " Runtime: ", time.time() - start_time)
             
    def run_metrics(self) -> None:
        """Generate some metrics
        """

        euclidan_error = []
        rotational_error = []

        # euclidian error and rotation
        truth_ref_frame = self.truth[0] 
        for pose_est, pose_true in zip(self.state_estimate,self.truth):
            pose_est = numpy_to_gtsam(pose_est)
            pose_true = truth_ref_frame.between(pose_true)
            diff = pose_est.between(pose_true)
            euclidan_error.append(np.sqrt(diff.x()**2 + diff.y()**2))
            rotational_error.append(np.degrees(diff.theta()))
        euclidan_error = np.array(euclidan_error)
        rotational_error = np.array(rotational_error)
        self.mse = np.mean(euclidan_error)
        self.rmse = np.sqrt(np.mean(euclidan_error**2))

        # uncertainty
        team_uncertainty = {}
        for robot in self.partner_robot_state_estimates.keys():
            temp = []
            counter = []
            for i in range(len(self.partner_robot_covariance[robot])):
                temp.append(np.linalg.det(self.partner_robot_covariance[robot][i]))
                counter.append(i)
            team_uncertainty[robot] = (counter,temp)

            print("list: ",len(temp))
        self.team_uncertainty = team_uncertainty

    def plot(self) -> None:
        """Visulize the mission
        """

        self.run_metrics()

        for robot in self.team_uncertainty:
            plt.plot(self.team_uncertainty[robot][0],self.team_uncertainty[robot][1])
        plt.show()
        
        # my own trajectory
        plt.plot(self.state_estimate[:,1],self.state_estimate[:,0],c="black")
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

        for loop in self.possible_loops:
            i, r, j = loop
            if i >= len(self.state_estimate): continue
            one = numpy_to_gtsam(self.state_estimate[i])
            if j >= len(self.partner_robot_state_estimates[r]): continue
            two = self.partner_robot_state_estimates[r][j]
            # plt.plot([one.y(),two.y()],[one.x(),two.x()],c="purple")

        for loop in self.best_possible_loops:
            i, r, j = loop
            if i >= len(self.state_estimate): continue
            one = numpy_to_gtsam(self.state_estimate[i])
            if j >= len(self.partner_robot_state_estimates[r]): continue
            two = self.partner_robot_state_estimates[r][j]
            plt.plot([one.y(),two.y()],[one.x(),two.x()],c="purple")


        plt.axis("square")
        plt.show()

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







    
    




        
    

    