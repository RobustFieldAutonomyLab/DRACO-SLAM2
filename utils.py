import gtsam
import pickle
import rosbag
import numpy as np

from scipy.spatial.transform import Rotation

from loop_closure import LoopClosure
from bruce_slam import pcl

def gtsam_to_numpy(pose : gtsam.Pose2) -> np.array:
    """Convert a gtsam pose2 to a numpy array

    Args:
        pose (gtsam.Pose2): the pose we want to convert

    Returns:
        np.array: numpy array [x,y,theta]
    """

    return np.array([pose.x(),pose.y(),pose.theta()])

def numpy_to_gtsam(pose : np.array) -> gtsam.Pose2:
    """Conver a numpy array into a gtsam.Pose2

    Args:
        pose (np.array): the numpy array [x,y,theta]

    Returns:
        gtsam.Pose2: the pose as a gtsam.Pose2
    """

    return gtsam.Pose2(pose[0],pose[1],pose[2])

def rot_to_euler(matrix : np.array) -> float:
    """Get the angle from a 2x2 rotation matrix

    Args:
        matrix (np.array): the rotation matrix

    Returns:
        float: the single angle from the matrix
    """

    # get the theta rotation angle for ICP
    matrix = np.row_stack((np.column_stack((matrix, [0, 0])), [0, 0, 1]))
    theta = Rotation.from_matrix(matrix).as_euler("xyz")[2]
    return theta

def get_ground_truth(path:str,id:int,stamps:list) -> list:
    """Get a ground truth trajectory using the timestamps from a SLAM run

    Args:
        path (str): _description_
        id (int): the id of the robot mission (1,2,3,etc.)
        stamps (list): the list of timestamps from the slam keyframes

    Returns:
        list: list of GTSAM.Pose2, these are the true poses for SLAM
    """

    with open(r"/home/jake/Desktop/holoocean_bags/scrape/" + str(id) +".pickle", "rb") as input_file:
        data_one = pickle.load(input_file)
    bag = rosbag.Bag('/home/jake/Desktop/holoocean_bags/' + str(id) + '.bag')
    table = {}
    for stamp in stamps:
        stamp = stamp.to_sec()
        table[stamp] = stamp

    out = []
    for topic, msg, t in bag.read_messages(topics=['/pose_true']):
        temp = (msg.header.stamp).to_sec()
        if temp in table:
            table[temp]
            x = msg.pose.position.x
            y = msg.pose.position.y
            z = msg.pose.position.z
            qx = msg.pose.orientation.x
            qy = msg.pose.orientation.y
            qz = msg.pose.orientation.z
            qw = msg.pose.orientation.w
            pose = gtsam.Pose3(gtsam.Rot3.Quaternion(qw, qx, qy, qz),gtsam.Point3(x,-y,z))
            out.append(gtsam.Pose2(pose.x(),pose.y(),pose.rotation().yaw()))
    bag.close()
    return out

def grade_loop(loop_out:LoopClosure, one:gtsam.Pose2, two:gtsam.Pose2, max_distance:float, max_rotation:float) -> bool:
    """Test if a loop closure is correct or incorrect. Returns boolean of that question. 

    Args:
        loop_out (LoopClosure): the loop closure, after run through the registration system
        one (gtsam.Pose2): the true pose of source
        two (gtsam.Pose2): the true pose of target
        max_distance (float): the max distance between poses to be correct
        max_rotation (float): the max theta angle beteween poses to be correct

    Returns:
        bool: True/False if this loop closure is correct/incorrect
    """

    true_between = one.between(two) # get the true transform between poses
    temp = loop_out.icp_transform.inverse() 
    icp_result = temp.compose(loop_out.gi_transform.inverse()) # get the ICP derived transform
    differnce = true_between.between(icp_result) # compare them 
    distance = np.sqrt(differnce.x()**2 + differnce.y()**2)
    rotation = abs(differnce.theta())
    return max_distance > distance and max_rotation > rotation

def transform_points(points: np.array, pose: gtsam.Pose2) -> np.array:
        """transform a set of 2D points given a pose

        Args:
            points (np.array): point cloud to be transformed
            pose (gtsam.Pose2): transformation to be applied

        Returns:
            np.array: transformed point cloud
        """

        # check if there are actually any points
        if len(points) == 0:
            return np.empty_like(points, np.float32)

        # convert the pose to matrix format
        T = pose.matrix().astype(np.float32)

        # rotate and translate to the global frame
        return points.dot(T[:2, :2].T) + T[:2, 2]
    
def get_points(pose_index : int, submap_size : int, points : list, poses :list) -> np.array:
    """ Get an aggragated set of points, the reference frame for which at pose_index.

    Args:
        pose_index (int): the index we want the reference frame at
        submap_size (int): the size of the submap we want, how many frames before/after this frame
        points (list): a list of the np.array points from the robot mission
        poses (list): a list of the np.array poses from the robot mission

    Returns:
        np.array: aggragated point cloud in the reference frame of pose_index
    """
    
    if(pose_index >= len(points)):
        raise KeyError
    cloud = np.array(points[pose_index])
    ref_pose = gtsam.Pose2(poses[pose_index][0],poses[pose_index][1],poses[pose_index][2])
    for i in range(pose_index-1, pose_index-submap_size-1, -1):
        if i>0:
            temp = np.array(points[i])
            temp_pose = gtsam.Pose2(poses[i][0],poses[i][1],poses[i][2])
            temp_pose = ref_pose.between(temp_pose)
            temp = transform_points(temp, temp_pose)
            cloud = np.row_stack((cloud,temp))
    for i in range(pose_index+1, pose_index+submap_size+1, 1):
        if i<len(points):
            temp = np.array(points[i])
            temp_pose = gtsam.Pose2(poses[i][0],poses[i][1],poses[i][2])
            temp_pose = ref_pose.between(temp_pose)
            temp = transform_points(temp, temp_pose)
            cloud = np.row_stack((cloud,temp))
    cloud = pcl.downsample(cloud,0.3)
    return cloud


def get_scan_context_aggragated(points : np.array, bearing_bins : int, range_bins : int, 
                                                    max_range : int, max_bearing : int) -> np.array:
    """Perform scan context for an aggragated point cloud

    Args:
        points (np.array): the point cloud we want converted to a ring key and context image
        bearing_bins (int): the number of bins in the bearing axis
        range_bins (int): the number of bins in the range axis
        max_range (int): the max range in the context image
        max_bearing (int):the max angle in the context image

    Returns:
        np.array: a scan context image of the provided point cloud
    """

    # instanciate the image
    polar_image = np.zeros((bearing_bins, range_bins))

    # convert to discrete polar coords
    r_cont = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))  # first contiuous polar coords
    b_cont = abs(np.degrees(np.arctan2(points[:, 0], points[:, 1])))  
    r_dis = np.array((r_cont / max_range) * range_bins).astype(np.uint16)  # discret coords
    b_dis = np.array((((b_cont / max_bearing) + 1) / 2) * bearing_bins).astype(np.uint16)

    # clip the vales
    r_dis = np.clip(r_dis, 0, range_bins - 1)
    b_dis = np.clip(b_dis, 0, bearing_bins - 1)

    # populate the image
    for i, j in zip(b_dis, r_dis):
        polar_image[i][j] = 1

    # build the ring key
    ring_key = np.sum(polar_image, axis=0)

    return polar_image, ring_key


def get_all_context(poses : list, 
                                points : np.array, 
                                SUBMAP_SIZE : int, 
                                BEARING_BINS : int, 
                                RANGE_BINS : int, 
                                MAX_RANGE : int,
                                MAX_BEARING : int) -> list:
    """Get all the context images from a whole mission

    Args:
        poses (list): _description_
        points (np.array): _description_
        SUBMAP_SIZE (int): number of clouds in the submaps
        BEARING_BINS (int): number of bearing bins in the context image
        RANGE_BINS (int): number of range bins in the context image
        MAX_RANGE (int): max range in the context image
        MAX_BEARING (int): max bearing in the context image

    Returns:
        list: _description_
    """

    keys = []
    context = []
    for i in range(len(poses)):
        submap = get_points(i,SUBMAP_SIZE,points,poses)
        context_img, ring_key = get_scan_context_aggragated(submap,
                                                            BEARING_BINS,
                                                            RANGE_BINS,
                                                            MAX_RANGE,
                                                            MAX_BEARING)
        keys.append(ring_key)
        context.append(context_img)
    return keys, context
