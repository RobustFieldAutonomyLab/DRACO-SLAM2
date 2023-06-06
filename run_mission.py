import pickle
import gtsam
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data,search_for_loops,reject_loops,grade_loop_list,plot_loop,keep_best_loop, search_for_loops_with_prior,verify_pcm,flip_loops
from robot import Robot
from registration import Registration
from loop_closure import LoopClosure
from comm_link import CommLink

from config.usmma import *

def run(sampling_points,iterations,tolerance,max_translation,max_rotation,
        SUBMAP_SIZE,BEARING_BINS,RANGE_BINS,MAX_RANGE,MAX_BEARING,
        MIN_POINTS,RATIO_POINTS,CONTEXT_DIFFERENCE,MIN_OVERLAP,MAX_TREE_DIST,KNN):

    # define a registration system
    reg = Registration(sampling_points,iterations,tolerance,max_translation,max_rotation)

    # Load up the data
    data_one = load_data("/home/jake/Desktop/holoocean_bags/scrape/5.pickle",5)
    data_two = load_data("/home/jake/Desktop/holoocean_bags/scrape/6.pickle",6)
    data = {1:data_one,2:data_two}

    robots = {}
    robot_list = [1,2]
    for key in data.keys():
        robots[key] = Robot(key,data[key],SUBMAP_SIZE,BEARING_BINS,RANGE_BINS,MAX_RANGE,MAX_BEARING)

    queue = []
    loop_list = []
    mode = 0

    comm_link = CommLink()

    for slam_step in range(63):
        print(slam_step)

        # step the robots forward
        for robot_id in robots.keys():
            robots[robot_id].step()
            comm_link.log_message(128) # log the descriptor sharing

        # search for loop closures
        for robot_id_source in robots.keys():
            for robot_id_target in robots.keys():
                if robot_id_target == robot_id_source: continue # do not search with self

                # Update the partner trajectory and account for comms
                state_cost = robots[robot_id_source].update_partner_trajectory(robot_id_target,robots[robot_id_target].state_estimate)
                comm_link.log_message(state_cost)

                if slam_step >= 1000:
                    loops = search_for_loops_with_prior(reg,robots,comm_link,robot_id_source)
                    if len(loops) > 0:
                        loop_, loop_search_status = keep_best_loop(loops) # retain only the best loop from the batch
                        robots[robot_id_source].merge_slam(loops,True) # merge my graph

                else:
                    # perform some loop closure search
                    loops = search_for_loops(reg,
                                            robots,
                                            comm_link,
                                            robot_id_source,
                                            robot_id_target,
                                            MIN_POINTS,
                                            RATIO_POINTS,
                                            CONTEXT_DIFFERENCE,
                                            MIN_OVERLAP,
                                            MAX_TREE_DIST,
                                            KNN)
                
                    # only keep going if we found any loop closures
                    if len(loops) == 0: continue
                    loop_, loop_search_status = keep_best_loop(loops) # retain only the best loop from the batch
                    if loop_search_status == False: continue # make sure the best loop not outlier

                    # do some book keeping
                    loop_.place_loop(robots[robot_id_source].get_pose_gtsam())
                    loop_.source_robot_id = robot_id_source
                    loop_.target_robot_id = robot_id_target

                    # check the status of the loop closure
                    if loop_.status:
                        
                        # update and solve PCM
                        robots[robot_id_source].add_loop_to_pcm_queue(loop_)
                        valid_loops = robots[robot_id_source].do_pcm(robot_id_target)

                        # if we have a valid solution from PCM, merge the graphs
                        if len(valid_loops) > 0:
                            robots[robot_id_source].merge_slam(valid_loops) # merge my graph
                            flipped_valid_loops = flip_loops(valid_loops) # flip and send the loops
                            for i in range(len(flipped_valid_loops)): comm_link.log_message(96 + 16 + 16)
                            robots[robot_id_target].merge_slam(flipped_valid_loops) # merge the partner robot graph
                            
                        for valid in valid_loops: 
                            loop_list.append(valid)
                            # robots[robot_id_source].plot()
                            # robots[robot_id_target].plot()
                        
    # plot each of the robots
    for robot in robots.keys():
        robots[robot].plot()

    comm_link.plot()

run(sampling_points,iterations,tolerance,max_translation,max_rotation,
        SUBMAP_SIZE,BEARING_BINS,RANGE_BINS,MAX_RANGE,MAX_BEARING,
        MIN_POINTS,RATIO_POINTS,CONTEXT_DIFFERENCE,MIN_OVERLAP,MAX_TREE_DIST,KNN)




            



            

