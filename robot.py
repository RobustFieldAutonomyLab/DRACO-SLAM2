from scipy.spatial import KDTree
from typing import Tuple
import numpy as np
import gtsam

from utils import get_all_context,get_points,verify_pcm

from loop_closure import LoopClosure

class Robot():
    def __init__(self,data:dict,SUBMAP_SIZE:int,BEARING_BINS:int,RANGE_BINS:int,MAX_RANGE:int,MAX_BEARING:int):
        
        self.slam_step = 0 # we want to track how far along the mission is, what SLAM step are we on
        self.total_steps = len(data["poses"])

        self.poses = data["poses"] # poses as numpy arrays
        self.poses_g = data["poses_g"] # poses from above but as gtsam.Pose2
        self.points = data["points"]  # raw points at each pose
        self.points_t = data["points_t"] # transformed points at each pose
        self.truth = data["truth"] # ground truth for each pose

        # get the scan context images and ring keys
        self.keys, self.context = get_all_context(self.poses,
                                                            self.points,
                                                            SUBMAP_SIZE,
                                                            BEARING_BINS,
                                                            RANGE_BINS,
                                                            MAX_RANGE,
                                                            MAX_BEARING)
        
        self.submap_size = SUBMAP_SIZE
        self.pcm_queue = []
        self.pcm_queue_size = 5
        self.min_pcm = 2

        self.inter_robot_loop_closures = []

        self.point_clouds_received = {} # log when we have gotten a point cloud. Format: robot_id_number, keyframe_id
        
    def step(self) -> None:
        """Increase the step of SLAM
        """

        self.slam_step += 1

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

        # update the pcm queue
        while (self.pcm_queue and loop.source_key - self.pcm_queue[0].source_key > self.pcm_queue_size):
            self.pcm_queue.pop(0)
        self.pcm_queue.append(loop) # add the loop

    def do_pcm(self) -> list:
        """Check the self.pcm_queue for any valid loop closures. Return them as a list.

        Returns:
            list: a list of LoopClosures
        """

        valid_loops = []
        valid_indexes = verify_pcm(self.pcm_queue,self.min_pcm)
        for i in valid_indexes: 
            if self.pcm_queue[i].inserted == False:
                self.pcm_queue[i].inserted = True
                valid_loops.append(self.pcm_queue[i])
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






    
    




        
    

    