import gtsam
import numpy as np

class LoopClosure():
    """A class to store all the info needed for a loop closure
    """
    
    def __init__(self,source_key,target_key,source_points,target_points,source_context,target_context,true_source=None,true_target=None) -> None:
        self.source_key = source_key
        self.target_key = target_key
        self.source_points = source_points
        self.target_points = target_points
        self.status = True
        self.grade = None
        self.message = "initialized"
        self.gi_transform = None
        self.icp_transform = None
        self.estimated_transform = None
        self.source_context = source_context
        self.target_context = target_context

        self.source_points_init = None
        self.reg_points = None 
        self.fit_score = None

        self.count = None
        self.ratio = None
        self.context_diff = None
        self.overlap = None

        self.true_source = true_source
        self.true_target = true_target

        self.source_pose = None
        self.target_pose = None

        self.source_robot_id = None
        self.target_robot_id = None

        self.pose_error = None

        self.inserted = False

    def place_loop(self,source_pose:gtsam.Pose2) -> None:
        """Place the target pose in my frame using the icp transform. 

        Args:
            source_pose (gtsam.Pose2): the pose of the source frame, from my
            own robot mission
        """

        # get the transform between the frames
        temp = self.icp_transform.inverse() 
        self.estimated_transform = temp.compose(self.gi_transform.inverse()) 

        # find the target pose in my frame
        self.source_pose = source_pose
        self.target_pose = source_pose.compose(self.estimated_transform)

        # set a covariance matrix
        self.cov = np.eye(3) * self.fit_score * 20.0