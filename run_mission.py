import sys
import pickle
import gtsam
import numpy as np
import matplotlib.pyplot as plt

from utils import do_loops,load_data, numpy_to_gtsam,search_for_loops,reject_loops,grade_loop_list,plot_loop,keep_best_loop, search_for_loops_with_prior,verify_pcm,flip_loops
from robot import Robot
from registration import Registration
from loop_closure import LoopClosure
from comm_link import CommLink

from config.usmma import *

def run(sim,mission,study_samples,total_slam_steps,
        sampling_points,iterations,tolerance,max_translation,max_rotation,
        SUBMAP_SIZE,BEARING_BINS,RANGE_BINS,MAX_RANGE,MAX_BEARING,
        MIN_POINTS,RATIO_POINTS,CONTEXT_DIFFERENCE,MIN_OVERLAP,MAX_TREE_DIST,KNN,
        mode,alcs_overlap,min_uncertainty):
    

    '''sim: if you are using simulated data or not, if True, the data loader grabs ground truth information
    mission, a string of the misson name, options are "plane" "usmma" "usmma_real"
    study_samples, how many times you want to run this single slam misson
    total_slam_steps, the number of steps you want to step in this slam mission
    sampling_points, parameter for global ICP, see SHGO docs
    iterations,parameter for global ICP, see SHGO docs
    tolerance, parameter for global ICP, see SHGO docs
    max_translation, parameter for global ICP, see SHGO docs, the largest translation it will check
    max_rotation, parameter for global ICP, see SHGO docs, the largest rotation it will check
    SUBMAP_SIZE, the number of frames before and after the current keyframe
    BEARING_BINS, scan context number of bearing bins
    RANGE_BINS, scan context number of range bins
    MAX_RANGE, the max range for scan context
    MAX_BEARING, the max bearing for scan context
    MIN_POINTS, the minimum points to try registration
    RATIO_POINTS, the ratio of points to try registration 
    CONTEXT_DIFFERENCE, the max context difference
    MIN_OVERLAP, the min overlap to accept a loop closure
    MAX_TREE_DIST,KNN, the max tree distance to accept a loop closure
    mode, DRACO (0) or ALCS (1)
    alcs_overlap, the overlap needed for ALCS
    min_uncertainty, the min unceratinaty for ALCS'''
    
    mode = int(mode) 
    alcs_overlap = float(alcs_overlap)
    min_uncertainty = float(min_uncertainty)
    sim = sim == "True"

    print(sim)
    
    for study_step in range(study_samples):

        # define a registration system
        reg = Registration(sampling_points,iterations,tolerance,max_translation,max_rotation)

        # Load up the data
        data_one = load_data("/home/jake/Desktop/holoocean_bags/scrape/"+mission+"_1.pickle",mission+"_1",sim)
        data_two = load_data("/home/jake/Desktop/holoocean_bags/scrape/"+mission+"_2.pickle",mission+"_2",sim)
        data_three = load_data("/home/jake/Desktop/holoocean_bags/scrape/"+mission+"_3.pickle",mission+"_3",sim)
        data = {1:data_one,2:data_two,3:data_three}

        robots = {}
        for key in data.keys():
            robots[key] = Robot(key,data[key],SUBMAP_SIZE,BEARING_BINS,RANGE_BINS,MAX_RANGE,MAX_BEARING)
            robots[key].mission = mission
            robots[key].mode = int(mode)
            robots[key].min_uncertainty = min_uncertainty

        # pass along the partner ground truth
        for robot in robots.keys():
            for partner in robots.keys():
                if robot == partner: continue
                robots[robot].partner_truth[partner] = robots[partner].truth

        loop_list = []
        comm_link = CommLink()
        # plane 80
        # usmma_sim 60
        # summa real 103
        for _ in range(total_slam_steps):

            # step the robots forward
            for robot_id in robots.keys():
                robots[robot_id].step()
                comm_link.log_message(128) # log the descriptor sharing

            # search for loop closures
            for robot_id_source in robots.keys():
                for robot_id_target in robots.keys():
                    if robot_id_target == robot_id_source: continue # do not search with self
                    robots[robot_id_source].points_partner[robot_id_target] = robots[robot_id_target].points

                    # perform an ALCS search
                    robots[robot_id_source].alcs(robot_id_target)

                    # Update the partner trajectory and account for comms
                    state_cost = robots[robot_id_source].update_partner_trajectory(robot_id_target,robots[robot_id_target].state_estimate)
                    comm_link.log_message(state_cost)
                        
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
                                            KNN,
                                            alcs_overlap)
                
                    # only keep going if we found any loop closures
                    if len(loops) == 0: continue
                    loop_, loop_search_status = keep_best_loop(loops) # retain only the best loop from the batch
                    if loop_search_status == False: continue # make sure the best loop not outlier

                    # do some book keeping
                    loop_.place_loop(robots[robot_id_source].get_pose_gtsam_at_index(loop_.source_key))

                    # check the status of the loop closure
                    if loop_.status:
                        if robots[robot_id_source].is_merged(robot_id_target):
                            valid_loops = [loop_]
                            # plot_loop(loop_)
                        else: # update and solve PCM
                            robots[robot_id_source].add_loop_to_pcm_queue(loop_)
                            valid_loops = robots[robot_id_source].do_pcm(robot_id_target)

                        # if we have a valid solution from PCM, merge the graphs
                        if len(valid_loops) > 0:
                            robots[robot_id_source].merge_slam(valid_loops) # merge my graph
                            flipped_valid_loops = flip_loops(valid_loops) # flip and send the loops
                            for i in range(len(flipped_valid_loops)): comm_link.log_message(96 + 16 + 16)
                            robots[robot_id_target].merge_slam(flipped_valid_loops) # merge the partner robot graph
                            if mode == 1:
                                robots[robot_id_source].update_merge_log(robot_id_target) # log that we have merged
                                robots[robot_id_target].update_merge_log(robot_id_source) 
                        
                        for valid in valid_loops: 
                            loop_list.append(valid)
                            plot_loop(valid,mission)
                            # if valid.method == "alcs":
                            '''print(slam_step)
                            print(robot_id_source,robot_id_target)
                            print(valid.source_robot_id,valid.target_robot_id,valid.source_key,valid.target_key)
                            print("------------")
                            plot_loop(valid)'''
                            
        # plot each of the robots
        '''for robot in robots.keys():
            robots[robot].run_metrics(mission,mode,study_step)
        comm_link.report(mission,mode,study_step)
        '''
        
def main():
    _, sim, mission,study_samples,total_slam_steps, mode, alcs_overlap, min_uncertainty = sys.argv

    print("running mission")
    print("Mission: ", mission)
    print("Study Samples: ",study_samples)
    print("Slam Steps: ", total_slam_steps)
    print("Mode: ", mode)
    print("alcs_overlap: ",alcs_overlap)
    print("Min Uncertainty: ",min_uncertainty)
    run(sim,mission,int(study_samples),int(total_slam_steps),
        sampling_points,iterations,tolerance,max_translation,max_rotation,
        SUBMAP_SIZE,BEARING_BINS,RANGE_BINS,MAX_RANGE,MAX_BEARING,
        MIN_POINTS,RATIO_POINTS,CONTEXT_DIFFERENCE,MIN_OVERLAP,MAX_TREE_DIST,KNN,
        mode,alcs_overlap,min_uncertainty)

if __name__ == "__main__":
    main()


            



            

