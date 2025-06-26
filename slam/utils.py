import gtsam
import pickle
import rosbag
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.spatial.transform import Rotation
from collections import defaultdict
from itertools import combinations
import time

from slam.loop_closure import LoopClosure
from bruce_slam import pcl
import os

def gtsam_to_numpy(pose: gtsam.Pose2) -> np.array:
    """Convert a gtsam pose2 to a numpy array

    Args:
        pose (gtsam.Pose2): the pose we want to convert

    Returns:
        np.array: numpy array [x,y,theta]
    """

    return np.array([pose.x(), pose.y(), pose.theta()])


def numpy_to_gtsam(pose: np.array) -> gtsam.Pose2:
    """Conver a numpy array into a gtsam.Pose2

    Args:
        pose (np.array): the numpy array [x,y,theta]

    Returns:
        gtsam.Pose2: the pose as a gtsam.Pose2
    """

    return gtsam.Pose2(pose[0], pose[1], pose[2])


def rot_to_euler(matrix: np.array) -> float:
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


def get_ground_truth(bag_path: str, stamps: list) -> list:
    """Get a ground truth trajectory using the timestamps from a SLAM run

    Args:
        pickle_path (str): path to pickle file
        bag_path (str): path to bag file
        stamps (list): the list of timestamps from the slam keyframes

    Returns:
        list: list of GTSAM.Pose2, these are the true poses for SLAM
    """
    bag = rosbag.Bag(bag_path)
    table = {}
    for stamp in stamps:
        stamp = stamp.to_sec()
        table[stamp] = stamp

    out = []
    for topic, msg, t in bag.read_messages(topics=['/pose_true']):
        temp = (msg.header.stamp).to_sec()
        if temp in table:
            x = msg.pose.position.x
            y = msg.pose.position.y
            z = msg.pose.position.z
            qx = msg.pose.orientation.x
            qy = msg.pose.orientation.y
            qz = msg.pose.orientation.z
            qw = msg.pose.orientation.w
            pose = gtsam.Pose3(gtsam.Rot3.Quaternion(qw, qx, qy, qz), gtsam.Point3(x, -y, z))
            out.append(gtsam.Pose2(pose.x(), pose.y(), pose.rotation().yaw()))
    bag.close()
    return out


def load_data(file_path: str, real_data=False, bag_path=None) -> dict:
    """Load in the data from a single robot. Data is loaded
    as a python dictionary.

    Args:
        file_path (str): path to the pickle file that is a log of the 
        single robot SLAM mission
        real_data (bool): is this real world data?

    Returns:
        dict: the compiled data for a single robot
    """

    # Load up the data
    with open(file_path, "rb") as input_file:
        data_one = pickle.load(input_file)

    # parse out the datafile
    poses_one = data_one["poses"]
    poses_one_g = []
    for row in poses_one:
        poses_one_g.append(numpy_to_gtsam(row))
    points_one = data_one["points"]
    points_t_one = data_one["points_t"]
    time_one = data_one["time_stamps"]
    if not real_data:
        truth_one = get_ground_truth(bag_path, time_one)  # get the ground truth
    else:
        truth_one = None

    # populate a dictionary with all the data
    data = {}
    data["poses"] = poses_one  # poses as numpy arrays
    data["poses_g"] = poses_one_g  # poses from above but as gtsam.Pose2
    data["points"] = points_one  # raw points at each pose
    data["points_t"] = points_t_one  # transformed points at each pose
    data["truth"] = truth_one  # ground truth for each pose
    data["factors"] = data_one["factors"]  # the factors in the graph
    data["images"] = data_one["images"]  # the images at each pose

    return data


def grade_loop(loop_out: LoopClosure, one: gtsam.Pose2, two: gtsam.Pose2, max_distance: float, max_rotation: float) -> \
Tuple[bool, gtsam.Pose2]:
    """Test if a loop closure is correct or incorrect. Returns boolean of that question. 

    Args:
        loop_out (LoopClosure): the loop closure, after run through the registration system
        one (gtsam.Pose2): the true pose of source
        two (gtsam.Pose2): the true pose of target
        max_distance (float): the max distance between poses to be correct
        max_rotation (float): the max theta angle beteween poses to be correct

    Returns:
        Tuple[bool,gtsam.Pose2]: True/False if this loop closure is correct/incorrect and the error in gtsam format
    """

    true_between = one.between(two)  # get the true transform between poses
    temp = loop_out.icp_transform.inverse()
    icp_result = temp.compose(loop_out.gi_transform.inverse())  # get the ICP derived transform
    differnce = true_between.between(icp_result)  # compare them
    distance = np.sqrt(differnce.x() ** 2 + differnce.y() ** 2)
    rotation = abs(differnce.theta())
    return max_distance > distance and max_rotation > rotation, differnce


def grade_loop_list(loops: list, max_correct_distance: float, max_correct_rotation: float) -> list:
    for i in range(len(loops)):
        loops[i].grade, loops[i].pose_error = grade_loop(loops[i],
                                                         loops[i].true_source,
                                                         loops[i].true_target,
                                                         max_correct_distance,
                                                         max_correct_rotation)

    return loops


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


def get_points(pose_index: int, submap_size: int, points: list, poses: list) -> np.array:
    """ Get an aggragated set of points, the reference frame for which at pose_index.

    Args:
        pose_index (int): the index we want the reference frame at
        submap_size (int): the size of the submap we want, how many frames before/after this frame
        points (list): a list of the np.array points from the robot mission
        poses (list): a list of the np.array poses from the robot mission

    Returns:
        np.array: aggragated point cloud in the reference frame of pose_index
    """

    if (pose_index >= len(points)):
        raise KeyError
    cloud = np.array(points[pose_index])
    ref_pose = gtsam.Pose2(poses[pose_index][0], poses[pose_index][1], poses[pose_index][2])
    for i in range(pose_index - 1, pose_index - submap_size - 1, -1):
        if i > 0:
            temp = np.array(points[i])
            temp_pose = gtsam.Pose2(poses[i][0], poses[i][1], poses[i][2])
            temp_pose = ref_pose.between(temp_pose)
            temp = transform_points(temp, temp_pose)
            cloud = np.row_stack((cloud, temp))
    for i in range(pose_index + 1, pose_index + submap_size + 1, 1):
        if i < len(points):
            temp = np.array(points[i])
            temp_pose = gtsam.Pose2(poses[i][0], poses[i][1], poses[i][2])
            temp_pose = ref_pose.between(temp_pose)
            temp = transform_points(temp, temp_pose)
            cloud = np.row_stack((cloud, temp))
    cloud = pcl.downsample(cloud, 0.3)
    return cloud


def get_scan_context_aggragated(points: np.array, bearing_bins: int, range_bins: int,
                                max_range: int, max_bearing: int) -> np.array:
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


def get_all_context(poses: list, points: np.array, sc_config) -> list:
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
    SUBMAP_SIZE = sc_config['submap_size']
    BEARING_BINS = sc_config['bearing_bins']
    RANGE_BINS = sc_config['range_bins']
    MAX_RANGE = sc_config['max_range']
    MAX_BEARING = sc_config['max_bearing']
    keys = []
    context = []
    for i in range(len(poses)):
        submap = get_points(i, SUBMAP_SIZE, points, poses)
        context_img, ring_key = get_scan_context_aggragated(submap,
                                                            BEARING_BINS,
                                                            RANGE_BINS,
                                                            MAX_RANGE,
                                                            MAX_BEARING)
        keys.append(ring_key)
        context.append(context_img)
    return keys, context


def search_for_loops_with_prior(reg, robots: dict, comm_link, robot_id_source: int) -> list:
    loops = []
    for (source_frame, robot_id_target, target_frame) in robots[robot_id_source].best_possible_loops:

        # only evaluate the possible loop once
        if robots[robot_id_source].best_possible_loops[(source_frame, robot_id_target, target_frame)] == True:
            continue
        else:
            robots[robot_id_source].best_possible_loops[(source_frame, robot_id_target, target_frame)] = True

        # check for the data. If required get the data, log all comms costs
        required_data, comms_cost_request = robots[robot_id_source].check_for_data(robot_id_target, target_frame)
        comm_link.log_message(comms_cost_request)
        comms_cost_point_clouds = robots[robot_id_target].get_data(required_data)
        comm_link.log_message(comms_cost_point_clouds)

        # pull the points and state estimate
        # TODO SUBMAP SIZE
        source_points = robots[robot_id_source].get_robot_points(source_frame, 1)
        target_points = robots[robot_id_target].get_robot_points(target_frame, 1)
        target_pose_their_frame = robots[robot_id_target].state_estimate[target_frame]

        # pull the initial guess
        one = numpy_to_gtsam(robots[robot_id_source].state_estimate[source_frame])
        two = robots[robot_id_source].partner_robot_state_estimates[robot_id_target][target_frame]
        source_points = transform_points(source_points, one)
        target_points = transform_points(target_points, two)

        true_base = robots[robot_id_source].truth[0]
        true_one = robots[robot_id_source].truth[source_frame]
        true_two = robots[robot_id_target].truth[target_frame]
        # true_one = true_base.between(true_one)
        # true_two = true_base.between(true_two)

        # compute the loop closure
        loop = LoopClosure(source_frame,
                           target_frame,
                           source_points,
                           target_points,
                           None,
                           None,
                           target_pose_their_frame,
                           true_source=true_one,
                           true_target=true_two)
        loop.source_pose = one
        loop.target_pose = two
        loop.source_robot_id = robot_id_source
        loop.target_robot_id = robot_id_target

        loop = reg.evaluate_with_guess(loop, one)

        # do some more basic outlier rejection
        if loop.overlap >= 0.5:
            loops.append(loop)

    return loops


def do_loops(reg, robots: dict, comm_link, robot_id_source: int,
             MIN_POINTS: int, RATIO_POINTS: int, CONTEXT_DIFFERENCE: int, MIN_OVERLAP: float) -> list:
    loops = []
    for (ring_key_index, robot_id_target, j) in robots[robot_id_source].best_possible_loops:

        # only test each combo once
        if (ring_key_index, robot_id_target, j) in robots[robot_id_source].loops_tested: continue
        robots[robot_id_source].loops_tested[(ring_key_index, robot_id_target, j)] = True

        source_points = robots[robot_id_source].get_robot_points(ring_key_index, 5)  # pull the cloud at source
        source_context = robots[robot_id_source].get_context()  # pull the context image at source

        target_points = robots[robot_id_target].get_robot_points(j, 5)  # pull the cloud at target
        target_context = robots[robot_id_target].get_context_index(j)  # pull the context image at target
        target_pose_their_frame = robots[robot_id_target].state_estimate[
            j]  # the target robots jth pose in it's own ref frame

        loop = LoopClosure(ring_key_index,
                           j,
                           source_points,
                           target_points,
                           source_context,
                           target_context,
                           target_pose_their_frame,
                           true_source=robots[robot_id_source].truth[ring_key_index],
                           true_target=robots[robot_id_target].truth[j])
        loop_out = reg.evaluate(loop, 10, -1, -1, 0.55)
        robots[robot_id_source].icp_count += 1
        if loop.status:
            loop_out.place_loop(robots[robot_id_source].get_pose_gtsam())
            # print(loop.estimated_transform,loop.true_source.between(loop.true_target))
            loop_out.source_robot_id = robot_id_source
            loop_out.target_robot_id = robot_id_target
            loops.append(loop_out)

    return loops


def search_for_loops(reg, robots: dict, comm_link, robot_id_source: int, robot_id_target: int, config: dict) -> list:
    """Search for loops between the most recent frame in robot_id_source and all the frames in robot_id_target. 
    Apply ICP between any possible loop closures. We package these loop closures and return them as a list.

    Args:
        reg (Registration): the registration tool, this is used to perform ICP
        comm_link (CommLink): the communications link tracker
        robots (dict): a dictionary of Robot objects, this contains all robot data
        robot_id_source (int): the id number of the source robot, like a vin number
        robot_id_target (int): the id number of the target root, like a vin number
        MIN_POINTS (int): the min number of points in both clouds to try registration
        RATIO_POINTS (int): the ratio between the point clouds to try registration
        MIN_OVERLAP (float): the minimum overlap between the points after global init ICP
        MAX_TREE_DIST (int): the max distance allowed in feature space when doing kdsearch
        KNN (int): the number of neighbors we want when doing kd search

    Returns:
        list: a list of LoopClosures
    """

    MIN_POINTS = config['min_points']
    RATIO_POINTS = config['ratio_points']
    CONTEXT_DIFFERENCE = config['context_difference']
    MIN_OVERLAP = config['min_overlap']
    MAX_TREE_DIST = config['max_tree_dist']
    KNN = config['knn']
    alcs_overlap = config['alcs_overlap'] # for DRACO 1

    loop_list = []

    if robots[robot_id_source].is_merged(robot_id_target) == False:
        time0 = time.time()
        ring_key, ring_key_index = robots[robot_id_source].get_key()  # get the descipter we want to search with
        tree = robots[robot_id_target].get_tree()  # get the tree we want to search against
        distances, indexes = tree.query(ring_key, k=KNN, distance_upper_bound=MAX_TREE_DIST)  # search
        time1 = time.time()
        comm_link.log_time(time1 - time0, time_type='ringkey')

        # TODO check if we need submap size here
        source_points = robots[robot_id_source].get_robot_points(ring_key_index, 1)  # pull the cloud at source
        source_context = robots[robot_id_source].get_context()  # pull the context image at source

        indexes = indexes[distances <= MAX_TREE_DIST]  # filter the infinites
        submap_size = 1

    else:
        if robots[robot_id_source].best_possible_loops[robot_id_target] is None: return loop_list
        source_key, target_key = robots[robot_id_source].best_possible_loops[robot_id_target]
        ring_key_index = source_key
        indexes = [target_key]

        submap_size = 1
        source_points = robots[robot_id_source].get_robot_points(ring_key_index,
                                                                 submap_size)  # pull the cloud at source
        source_context = robots[robot_id_source].get_context()  # pull the context image at source

    count = 0
    for j in indexes:  # loop over the matches from the tree search
        if j < robots[robot_id_target].total_steps:  # protect for out of range
            if robots[robot_id_source].is_merged(robot_id_target) == False and robot_id_target in robots[
                robot_id_source].partner_robot_state_estimates:
                if j in robots[robot_id_source].partner_robot_state_estimates[robot_id_target]:
                    test_pose_source = numpy_to_gtsam(robots[robot_id_source].state_estimate[ring_key_index])
                    test_pose_target = robots[robot_id_source].partner_robot_state_estimates[robot_id_target][j]
                    test_pose = test_pose_source.between(test_pose_target)
                    dist = np.sqrt(test_pose.x() ** 2 + test_pose.y() ** 2)
                    rot = np.degrees(abs(test_pose.theta()))
                    if dist > 10 or rot > 60:
                        continue

            # log the comms cost
            required_data, comms_cost_request = robots[robot_id_source].check_for_data(robot_id_target, j)
            comm_link.log_message(comms_cost_request, usage_type='scan_id')
            comms_cost_point_clouds = robots[robot_id_target].get_data(required_data)
            comm_link.log_message(comms_cost_point_clouds, usage_type='scan')

            target_points = robots[robot_id_target].get_robot_points(j, submap_size)  # pull the cloud at target
            target_context = robots[robot_id_target].get_context_index(j)  # pull the context image at target
            target_pose_their_frame = robots[robot_id_target].state_estimate[
                j]  # the target robots jth pose in it's own ref frame

            loop = LoopClosure(ring_key_index,
                               j,
                               source_points,
                               target_points,
                               source_context,
                               target_context,
                               target_pose_their_frame)

            if robots[robot_id_source].truth is not None: loop.true_source = robots[robot_id_source].truth[
                ring_key_index]
            if robots[robot_id_target].truth is not None: loop.true_target = robots[robot_id_target].truth[j]

            loop.source_robot_id = robot_id_source
            loop.target_robot_id = robot_id_target
            if robots[robot_id_source].is_merged(robot_id_target) == False:
                start_time = time.time()
                loop_out = reg.evaluate(loop, MIN_POINTS, RATIO_POINTS, CONTEXT_DIFFERENCE, MIN_OVERLAP)
                comm_link.log_time(time.time() - start_time, 'icp')
                robots[robot_id_source].draco_reg_time.append(time.time() - start_time)
            else:
                start_time = time.time()
                # 60, 85
                loop_out = reg.evaluate(loop, 10, 100000, 100000000, alcs_overlap, alt=False)
                robots[robot_id_source].alcs_reg_time.append(time.time() - start_time)
                loop_out.method = "alcs"
            robots[robot_id_source].icp_count += 1
            if loop_out.ratio is not None:
                loop_list.append(loop_out)
            count += 1

    return loop_list


def reject_loops(loops: list, min_points: int, ratio_points: float, context_difference: int,
                 min_overlap: float) -> list:
    """Cull loop closures from the input list based on several thresholds

    Args:
        loops (list): the list of loop closures
        min_points (int): the minimum required points
        ratio_points (float): the max ratio between point clouds
        context_difference (int): the max difference between context images
        min_overlap (float): the minimum overlap between pointclouds

    Returns:
        list: the same list of loop closures, but with each status varible set
    """

    for i in range(len(loops)):

        # check for min number of points in BOTH clouds
        if min(loops[i].count) < min_points:
            loops[i].status = False

        # check the ratio between the point clouds
        if max(loops[i].ratio) > ratio_points:
            loops[i].status = False

        # check the difference between context images
        if loops[i].context_diff > context_difference:
            loops[i].status = False

        # check the overlap between two point clouds after registration
        if loops[i].overlap < min_overlap:
            loops[i].status = False

    return loops


def keep_best_loop(loops: list) -> LoopClosure:
    """Keep only the best loop closure from a batch

    Args:
        loops (list): a list of loop closures

    Returns:
        LoopClosure: the loopclosure with the highest overlap
    """

    scores = []
    loops_copy = []
    for loop in loops:
        if loop.status:
            scores.append(loop.overlap)
            loops_copy.append(loop)

    if len(scores) > 0:
        return loops_copy[np.argmax(scores)], True
    else:
        return None, False


def plot_loop(loop: LoopClosure, loop_path: str) -> None:
    """Plot a loop closure using matplotlib

    Args:
        loop (LoopClosure): the loop closure we want to vis
    """

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
    '''ax1.scatter(loop.reg_points[:,0],loop.reg_points[:,1],c="blue")
    ax1.scatter(loop.target_points[:,0],loop.target_points[:,1],c="red")'''
    title = str(loop.overlap) + "_" + str(loop.context_diff)
    plt.title(title)

    source_temp = transform_points(loop.source_points, loop.source_pose)
    target_temp = transform_points(loop.target_points, loop.target_pose)

    ax1.scatter(source_temp[:, 0], -source_temp[:, 1], c="blue")
    ax1.scatter(target_temp[:, 0], -target_temp[:, 1], c="red")
    ax1.axis("square")

    if loop.true_source is not None:

        source_temp = transform_points(loop.source_points, loop.true_source)
        target_temp = transform_points(loop.target_points, loop.true_target)

        ax2.scatter(source_temp[:, 0], -source_temp[:, 1], c="blue")
        ax2.scatter(target_temp[:, 0], -target_temp[:, 1], c="red")
        ax2.axis("square")

        if loop.status and loop.grade:
            fig.suptitle("True Positive", fontsize=20)
        elif loop.status and loop.grade == False:
            fig.suptitle("False Positive", fontsize=20)
        elif loop.status == False and loop.grade:
            fig.suptitle("False Negative", fontsize=20)
        elif loop.status == False and loop.grade == False:
            fig.suptitle("True Negative", fontsize=20)

    plt.axis("square")
    if not os.path.exists(loop_path):
        os.makedirs(loop_path)
    plt.savefig(f"{loop_path}{loop.source_robot_id}_"
                f"{loop.target_robot_id}_"
                f"{loop.source_key}_{loop.target_key}.png")
    plt.clf()
    plt.close()


def find_cliques(G: defaultdict):
    """Returns all maximal cliques in an undirected graph.
    Args:
        G (defaultdict): consicentcy graph
    """

    if len(G) == 0:
        return

    adj = {u: {v for v in G[u] if v != u} for u in G}
    Q = [None]

    subg = set(G)
    cand = set(G)
    u = max(subg, key=lambda u: len(cand & adj[u]))
    ext_u = cand - adj[u]
    stack = []

    try:
        while True:
            if ext_u:
                q = ext_u.pop()
                cand.remove(q)
                Q[-1] = q
                adj_q = adj[q]
                subg_q = subg & adj_q
                if not subg_q:
                    yield Q[:]
                else:
                    cand_q = cand & adj_q
                    if cand_q:
                        stack.append((subg, cand, ext_u))
                        Q.append(None)
                        subg = subg_q
                        cand = cand_q
                        u = max(subg, key=lambda u: len(cand & adj[u]))
                        ext_u = cand - adj[u]
            else:
                Q.pop()
                subg, cand, ext_u = stack.pop()
    except IndexError:
        pass


def verify_pcm(queue: list, min_pcm_value: int) -> list:
    """Get the pairwise consistent measurements.
        Args:
            queue (list): the list of loop closures being checked.
            min_pcm_value (int): the min pcm value we want
        Returns:
            list: returns any pairwise consistent loops. We return a list of indexes in the provided queue.
        """

    # check if we have enough loops to bother
    if len(queue) < min_pcm_value:
        return []

    # convert the loops to a consistentcy graph
    G = defaultdict(list)
    for (a, ret_il), (b, ret_jk) in combinations(zip(range(len(queue)), queue), 2):
        pi = ret_il.target_pose
        pj = ret_jk.target_pose
        pil = ret_il.estimated_transform
        plk = ret_il.source_pose.between(ret_jk.source_pose)
        pjk1 = ret_jk.estimated_transform
        pjk2 = pj.between(pi.compose(pil).compose(plk))

        error = gtsam.Pose2.Logmap(pjk1.between(pjk2))
        md = error.dot(np.linalg.inv(ret_jk.cov)).dot(error)
        # chi2.ppf(0.99, 3) = 11.34
        if md < 11.34:  # this is not a magic number
            G[a].append(b)
            G[b].append(a)

    # find the sets of consistent loops
    maximal_cliques = list(find_cliques(G))

    # if we got nothing, return nothing
    if not maximal_cliques:
        return []

    # sort and return only the largest set, also checking that the set is large enough
    maximum_clique = sorted(maximal_cliques, key=len, reverse=True)[0]
    if len(maximum_clique) < min_pcm_value:
        return []

    return maximum_clique


def check_frame_for_overlap(source_points: np.array, source_pose: gtsam.Pose2, target_pose: gtsam.Pose2,
                            min_points: int) -> bool:
    """This function tests if there is enough overlap between two poses
    for there to possibly be a loop closure. Note that we only have one set of points not two. 
    We project the source_points into the target frame and use the sensor model to check if 
    there is significant overlap between these two frames without data exchange. 

    Args:
        source_points (np.array): source point cloud
        source_pose (gtsam.Pose2): the location of the source point cloud
        target_pose (gtsam.Pose2): the location of the target frame
        min_points (int): the minimum number of points for this to be a potential loop closure

    Returns:
        bool: True/False, is there enough overlap to be a potential loop closure
    """

    # transform the points and parse the frame
    x, y, theta = target_pose.x(), target_pose.y(), target_pose.theta()

    # get the range and bearing relative to the target pose
    r = np.sqrt((x - source_points[:, 0]) ** 2 + (y - source_points[:, 1]) ** 2)
    b = np.degrees(np.arctan2(y - source_points[:, 1],
                              x - source_points[:, 0]))

    # filter the points based on sensor range
    source_points = source_points[r <= 30]
    b = b[r <= 30]

    # filter based on sensor bearing, max left and right angle
    yaw = 180 + np.degrees(theta)
    yaw_min = yaw - 65.
    yaw_max = yaw + 65.
    source_points = source_points[(b <= yaw_max) & (b >= yaw_min)]

    return len(source_points) >= min_points


def X(index: int) -> gtsam.symbol:
    """Convert an index to a gtsam symbol. 

    Args:
        index (int): the value we want symbolized

    Returns:
        gtsam.symbol: the output gtsam symbol
    """

    return gtsam.symbol("x", index)


def robot_to_symbol(robot_id: int, keyframe_id: int) -> gtsam.symbol:
    """Get the gtsam symbol of a given robot at a given keyframe

    Args:
        robot_id (int): the robot id number we want the symbol for
        keyframe_id (int): the keyframe id number 

    Returns:
        gtsam.symbol: symbol object with the input info encoded
    """

    letters = {1: "a", 2: "b", 3: "c", 4: "d"}  # lookup table for robot symbols
    return gtsam.symbol(letters[robot_id], keyframe_id)


def flip_loops(loops: list) -> list:
    """Invert a loop closure list so it can be transmited to another robot. 

    Args:
        loops (list): the list of loop closures I found

    Returns:
        list: the same loop closures inverted so you can add them to your graph
    """

    loops_out = []
    for loop in loops:
        target_key = loop.source_key
        source_key = loop.target_key
        estimated_transform = loop.estimated_transform.inverse()
        target_pose_their_frame = gtsam_to_numpy(loop.source_pose)
        loop_fliped = LoopClosure(source_key, target_key, None, None, None, None, target_pose_their_frame)
        loop_fliped.target_robot_id = loop.source_robot_id
        loop_fliped.source_robot_id = loop.target_robot_id
        loop_fliped.estimated_transform = estimated_transform
        loop_fliped.source_pose = loop.target_pose_their_frame
        loop_fliped.target_pose = loop_fliped.source_pose.compose(estimated_transform)
        loop_fliped.cov = loop.cov
        loops_out.append(loop_fliped)

    return loops_out


def create_full_cloud() -> np.array:
    max_range = 30
    max_bearing = 65 + 90
    min_bearing = -65 + 90
    range_res = max_range / 512.
    bearing_res = 130. / 512

    arr = []
    b = min_bearing
    while b < max_bearing:
        r = 0.
        while r < max_range:
            b_rad = np.deg2rad(b)
            x = r * np.sin(b_rad)
            y = r * np.cos(b_rad)
            arr.append([x, y])
            r += range_res
        b += bearing_res

    return np.array(arr)
