import gtsam
import numpy as np

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


def get_all_context(poses : list, points : np.array, 
                                        SUBMAP_SIZE : int, 
                                        BEARING_BINS : int, 
                                        RANGE_BINS : int, 
                                        MAX_RANGE : int,
                                        MAX_BEARING : int) -> list:
    """_summary_

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
