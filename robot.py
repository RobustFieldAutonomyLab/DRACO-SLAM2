from scipy.spatial import KDTree
from typing import Tuple
import numpy as np
import gtsam

from utils import get_all_context,get_points,verify_pcm, X, robot_to_symbol

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
        self.state_estimate = []

        self.prior_sigmas = [0.1, 0.1, 0.01]
        self.prior_model = self.create_noise_model(self.prior_sigmas)

        self.multi_robot_frames = {}

        self.inter_robot_loop_closures = []

        self.point_clouds_received = {} # log when we have gotten a point cloud. Format: robot_id_number, keyframe_id
        
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
        self.update_graph()

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
    
    def update_graph(self) -> None:
        """Update the state estimate based on what we have in
        the graph.
        """

        # push the newest factors into the ISAM2 instance
        self.isam.update(self.graph, self.values)
        self.graph.resize(0)  # clear the graph and values once we push it to ISAM2
        self.values.clear()

        # Update the whole trajectory
        values = self.isam.calculateEstimate()
        temp = []
        for x in range(self.slam_step+1):
            pose = values.atPose2(X(x))
            temp.append([pose.x(),pose.y(),pose.theta()])
        self.state_estimate = np.array(temp)

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
        self.inter_robot_loop_closures += valid_loops
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
    
    def merge_slam(self,loop_closures:list) -> None:
        """Use the multi-robot loop closures to merge our SLAM graphs. 
        Here we will need to add any loop closures we have found and
        the partner robot's trajectory.
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
            
            # track which frames we have added to the graph
            if loop.target_robot_id not in self.multi_robot_frames: 
                self.multi_robot_frames[loop.target_robot_id] = {}
            if loop.target_key not in self.multi_robot_frames[loop.target_robot_id]:
                self.multi_robot_frames[loop.target_robot_id][loop.target_key] = True
                self.values.insert(target_symbol, loop.target_pose) # add the initial guess

        self.update_graph() # upate the graph with the new info






    
    




        
    

    