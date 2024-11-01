import argparse
import yaml
from slam.comm_link import CommLink
from slam.registration import Registration
from slam.robot import Robot
import slam.utils as utils
import copy
import time
import sys


def run(input_bag: str, input_pickle: str, input_yaml: str, output_folder: str):
    with open(input_yaml, 'r') as file:
        config = yaml.safe_load(file)
    run_info = config['dataset']
    sim = run_info['sim']
    mission = run_info['mission']

    # define a registration system
    reg = Registration(config['global_icp'])

    # load all robot datum
    data_robots = {}
    for i in range(run_info['num_robots']):
        pickle_path = f"{input_pickle}{mission}_{i + 1}.pickle"
        if sim:
            bag_path = f"{input_bag}{mission}_{i + 1}.bag"
        else:
            bag_path = None
        data_this_robot = utils.load_data(pickle_path, not sim, bag_path)
        data_robots[i + 1] = data_this_robot

    robots = {}
    for key in data_robots.keys():
        robots[key] = Robot(key, data_robots[key], run_info, config['scan_context'])

    # read neighbor ground truth
    for robot in robots.keys():
        for partner in robots.keys():
            if robot == partner:
                continue
            robots[robot].partner_truth[partner] = robots[partner].truth

    loop_list = []
    comm_link = CommLink()

    for _ in range(run_info['total_slam_steps']):
        # step the robots forward
        for robot_id in robots.keys():
            robots[robot_id].step(output_folder)
            comm_link.log_message(128)  # log the descriptor sharing

        # exchange the graph
        graph = True
        if graph:
            # at each stamp, everybody update the graph with each other if there is anything new in the graph
            for robot_id_target  in robots.keys():
                for robot_id_source in robots.keys():
                    if robot_id_target == robot_id_source:
                        continue
                    # exchange the graph with neighbor
                    robots[robot_id_target].receive_graph_from_neighbor(robot_id_source,
                                                                        robots[robot_id_source].get_graph())

                keyframe_id_self, keyframe_id_to_request = robots[robot_id_target].perform_graph_match()
                # if potential candidates are find, request the keyframes
                if keyframe_id_to_request:
                    for robot_id_source in keyframe_id_to_request.keys():
                        if robot_id_target == robot_id_source:
                            sys.stderr.write("Should not request from robot self!")
                        # asking latest keyframe poses from neighbor
                        pose_msgs = robots[robot_id_source].get_keyframes(keyframe_id_to_request[robot_id_source])
                        # receive & process keyframe poses from neighbor
                        scan_request_msg = (robots[robot_id_target].
                                            receive_keyframes_from_neighbor(robot_id_source,
                                                                            keyframe_id_to_request[robot_id_source],
                                                                            pose_msgs))
                        # asking scan from neighbor
                        scan_msgs = robots[robot_id_source].get_scans(scan_request_msg)
                        # receive & process scan from neighbor
                        robots[robot_id_target].receive_scans_from_neighbor(robot_id_source,
                                                                            scan_request_msg,
                                                                            scan_msgs)
                # time0 = time.time()
                # robots[robot_id_target].object_detection.plot_figure()
                # time1 = time.time()
                # print(f"plot_figure time: {time1 - time0:.3f}")



        # search for loop closures
        search = False
        if search:
            for robot_id_source in robots.keys():
                for robot_id_target in robots.keys():
                    if robot_id_target == robot_id_source:
                        continue  # do not search with self
                    robots[robot_id_source].points_partner[robot_id_target] = robots[robot_id_target].points

                    # perform an ALCS search
                    robots[robot_id_source].alcs(robot_id_target)

                    # Update the partner trajectory and account for comms
                    state_cost = robots[robot_id_source].update_partner_trajectory(robot_id_target,
                                                                                   robots[
                                                                                       robot_id_target].state_estimate)
                    comm_link.log_message(state_cost)

                    # perform some loop closure search
                    loops = utils.search_for_loops(reg, robots, comm_link,
                                                   robot_id_source, robot_id_target, config['loop_closure'])

                    # only keep going if we found any loop closures
                    if len(loops) == 0:
                        continue
                    loop_, best_loop = utils.keep_best_loop(loops)  # retain only the best loop from the batch
                    if not best_loop:
                        # make sure the best loop is not outlier
                        continue

                    # do some bookkeeping
                    loop_.place_loop(robots[robot_id_source].get_pose_gtsam_at_index(loop_.source_key))

                    # check the status of the loop closure
                    if loop_.status:
                        if robots[robot_id_source].is_merged(robot_id_target):
                            valid_loops = [loop_]
                        else:  # update and solve PCM
                            robots[robot_id_source].add_loop_to_pcm_queue(loop_)
                            valid_loops = robots[robot_id_source].do_pcm(robot_id_target)

                        # if we have a valid solution from PCM, merge the graphs
                        if len(valid_loops) > 0:
                            robots[robot_id_source].merge_slam(valid_loops)  # merge my graph
                            flipped_valid_loops = utils.flip_loops(valid_loops)  # flip and send the loops
                            for i in range(len(flipped_valid_loops)): comm_link.log_message(96 + 16 + 16)
                            robots[robot_id_target].merge_slam(flipped_valid_loops)  # merge the partner robot graph
                            if run_info['mode']:
                                robots[robot_id_source].update_merge_log(robot_id_target)  # log that we have merged
                                robots[robot_id_target].update_merge_log(robot_id_source)

                        for valid in valid_loops:
                            loop_list.append(valid)
                            utils.plot_loop(valid, f"{output_folder}{mission}/loop/")


def main():
    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument("-b", "--input_bag", type=str,
                        default="/home/rfal/Documents/data/Simulated_Ros_Bags/", help="Path to bag files")
    parser.add_argument("-p", "--input_pickle", type=str,
                        default="/home/rfal/Documents/data/scrape/", help="Path to pickle files")
    parser.add_argument("-y", "--input_yaml", type=str,
                        default="config/param.yaml", help="Yaml file for all parameters")
    parser.add_argument("-o", "--output", type=str,
                        default="/home/rfal/Documents/data/animate/", help="Output image directory")

    args = parser.parse_args()

    with open(args.input_yaml, 'r') as file:
        config = yaml.safe_load(file)

    print(f"If simulation mode: {config['dataset']['sim']}.")
    for study_step in range(config['dataset']['study_samples']):
        run(args.input_bag, args.input_pickle, args.input_yaml, args.output)


if __name__ == "__main__":
    main()
