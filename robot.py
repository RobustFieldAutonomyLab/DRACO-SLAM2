from scipy.spatial import KDTree
from typing import Tuple
import numpy as np

from utils import get_all_context,get_points


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
    
    
    

    