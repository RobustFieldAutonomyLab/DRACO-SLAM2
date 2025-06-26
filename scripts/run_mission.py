import argparse
import yaml
from slam.comm_link import CommLink
from slam.registration import Registration
from slam.robot import Robot
import slam.utils as utils
import copy
import time
import sys
import matplotlib.pyplot as plt
import os


def run(input_bag: str, input_pickle: str, input_yaml: str, output_folder: str):
    with open(input_yaml, 'r') as file:
        config = yaml.safe_load(file)
    run_info = config['dataset']
    sim = run_info['sim']
    mission = run_info['mission']

    if config['version'] == 2:
        graph = True
    elif config['version'] == 1:
        graph = False
        # define a registration system, this is for DRACO1, for comprasion purpose
        reg = Registration(config['global_icp'])
    else:
        return

    # load all robot datum
    # All data saved in pickle files are local SLAM result from Bruce SLAM
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
        # for DRACO1
        robots[key] = Robot(key, data_robots[key], run_info, config['scan_context'])

    # read neighbor ground truth
    for robot in robots.keys():
        for partner in robots.keys():
            if robot == partner:
                continue
            robots[robot].partner_truth[partner] = robots[partner].truth

    loop_list = []
    comm_link = CommLink()

    # prepare for the plots
    if config['visualization']:
        visualization = True
        os.makedirs(f'{output_folder}result_{mission}', exist_ok=True)
    else:
        visualization = False

    for slam_step in range(run_info['total_slam_steps']):
        print("SLAM step: ", slam_step)
        # step the robots forward
        for robot_id in robots.keys():
            robots[robot_id].step(output_folder)
            if not graph:
                comm_link.log_message(128, usage_type='ringkey')  # log the descriptor sharing
        ######### start of DRACO2 #########
        # exchange the graph
        if graph:
            # at each stamp, everybody update the graph with each other if there is anything new in the graph
            for id_target, robot_target in robots.items():
                # receive info
                for id_source, robot_source in robots.items():
                    if id_target == id_source:
                        continue
                    # exchange the graph with neighbor
                    graph_source = robot_source.get_graph()
                    comm_link.log_message(len(graph_source[0]) * 64 + 8, usage_type='graph')
                    # exchange with 2 16 bits float center (x,y), 8 bits int type
                    # 2 16 bits size (width, height), 8 bits int type
                    # 1 8 bit robot_id
                    # 2*16 + 2*16 = 64 bits per node 8 * (2+2) = 32
                    robot_target.receive_graph_from_neighbor(id_source, graph_source)

                    # for visualization purpose
                    robot_target.points_partner[id_source] = robot_source.points

                    if robot_target.loop_closure_manager.inter_factor_type == 3:
                        state_cost = robot_target.update_partner_trajectory(id_source, robot_source.state_estimate)
                        comm_link.log_message(state_cost, usage_type='pose')
                    else:
                        robot_target.receive_latest_states_from_neighbor(id_source, robot_source.state_estimate)
                        robot_target.receive_factors_from_neighbor(id_source, robot_source.get_factors_current())

                _, keyframe_id_to_request = robot_target.perform_graph_match()
                # if potential candidates are find, request the keyframes
                # exchange the keyframes
                for id_source, robot_source in robots.items():
                    if id_source == id_target:
                        continue
                    # no need to record these, already updated in update_partner_trajectory
                    # TODO: get these info locally
                    if id_source in keyframe_id_to_request.keys():
                        id_to_request_this = keyframe_id_to_request[id_source]
                        # 8 bit robot_ns, 8 bit per node id
                        # asking latest keyframe poses from neighbor
                        # no limits on poses passing
                        pose_msgs = robot_source.get_keyframes(id_to_request_this)
                        # receive & process keyframe poses from neighbor
                        scan_request_msg = (robot_target.receive_keyframes_from_neighbor(id_source,
                                                                                         id_to_request_this,
                                                                                         pose_msgs))
                    else:
                        scan_request_msg = set()
                    comm_link.log_message(len(scan_request_msg), usage_type='scan_id')
                    # asking scan from neighbor
                    # limits the number of scan passing each time
                    scan_request_msg, scan_msgs = robot_source.get_scans(id_target, scan_request_msg)
                    for scan in scan_msgs.values():
                        comm_link.log_message(scan.get_size(), usage_type='scan')
                    # receive & process scan from neighbor
                    robot_target.receive_scans_from_neighbor(id_source, scan_request_msg, scan_msgs)
                valid_loops = robot_target.perform_gcm()
                if len(valid_loops) == 0:
                    continue
                if robot_target.loop_closure_manager.inter_factor_type == -1:
                    pass
                elif robot_target.loop_closure_manager.inter_factor_type == 3:
                    valid_loops = robot_target.add_inter_robot_loop_closure_draco(valid_loops)
                    if robot_target.loop_closure_manager.exchange_inter_robot_factor:
                        for id_source, robot_source in robots.items():
                            if id_source == id_target:
                                continue
                            flipped_valid_loops = utils.flip_loops(valid_loops)
                            for i in range(len(flipped_valid_loops)):
                                comm_link.log_message(96 + 16 + 16, usage_type='loop')
                            robot_source.merge_slam(flipped_valid_loops)
                else:
                    robot_target.add_inter_robot_loop_closure(valid_loops)
                    inter_loop_msgs = robot_target.prepare_loops_to_send(valid_loops)
                    for id_source, robot_source in robots.items():
                        robot_source.receive_loops_from_neighbor(inter_loop_msgs)

            if visualization:
                fig = plt.figure(figsize=(16, 9), dpi=200)
                spec = fig.add_gridspec(22, 3)
                import seaborn as sns
                sns.set_style('darkgrid')

                for i, robot_target in enumerate(robots.values()):
                    axis_image = fig.add_subplot(spec[:9, i])
                    axis_grid = fig.add_subplot(spec[9:, i])
                    robot_target.plot_figure_with_sonar_image(axis_image, axis_grid)
                    # time1 = time.time()
                    # print(f"plot_figure time: {time1 - time0:.3f}")
                plt.tight_layout()
                fig.savefig(f'{output_folder}result_{mission}/{slam_step}.png')
                plt.close()
        ######### end of DRACO2 #########
        # search is the signal of running DRACO1
        if config['version'] == 2:
            search = False
        elif config['version'] == 1:
            search = True
        ######### start of DRACO1 #########
        if search:
            for robot_id in robots.keys():
                robots[robot_id].animate_step(output_folder)
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
                    comm_link.log_message(state_cost, usage_type='pose')

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
                            for i in range(len(flipped_valid_loops)):
                                comm_link.log_message(96 + 16 + 16, usage_type='loop')
                            robots[robot_id_target].merge_slam(flipped_valid_loops)  # merge the partner robot graph
                            if run_info['mode']:
                                robots[robot_id_source].update_merge_log(robot_id_target)  # log that we have merged
                                robots[robot_id_target].update_merge_log(robot_id_source)

                        for valid in valid_loops:
                            loop_list.append(valid)
                            utils.plot_loop(valid, f"{output_folder}{mission}/loop/")
        ######### end of DRACO1 #########
    for robot_id, robot in robots.items():
        robot.write_all_trajectories(f'{output_folder}result_{mission}')
    comm_link.report(f'{output_folder}result_{mission}', mission,
                     f"{mission}", 0)
    comm_link.plot()


def main():
    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument("-b", "--input_bag", type=str,
                        default="/home/rfal/Documents/data/Simulated_Ros_Bags/", help="Path to bag files")
    parser.add_argument("-p", "--input_pickle", type=str,
                        default="/home/rfal/Documents/data/scrape/", help="Path to pickle files")
    parser.add_argument("-y", "--input_yaml", type=str,
                        default="config/param.yaml", help="Yaml file for all parameters")
    parser.add_argument("-o", "--output", type=str,
                        default="/home/rfal/animate/", help="Output image directory")

    args = parser.parse_args()

    with open(args.input_yaml, 'r') as file:
        config = yaml.safe_load(file)

    print(f"If simulation mode: {config['dataset']['sim']}.")
    for study_step in range(config['dataset']['study_samples']):
        run(args.input_bag, args.input_pickle, args.input_yaml, args.output)


if __name__ == "__main__":
    main()
