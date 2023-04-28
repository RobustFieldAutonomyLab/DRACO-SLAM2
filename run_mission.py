import pickle
import gtsam
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data,search_for_loops,reject_loops,grade_loop_list,plot_loop,keep_best_loop,verify_pcm
from robot import Robot
from registration import Registration
from loop_closure import LoopClosure

SUBMAP_SIZE = 1
BEARING_BINS = 100
RANGE_BINS = 16
MAX_RANGE = 40
MAX_BEARING = 180
KNN = 5
MAX_TREE_DIST = 20

sampling_points = 500
iterations = 5
tolerance = .01
max_translation = 10
max_rotation = np.radians(100.)

min_points = 75
ratio_points = 2.0         
context_difference = 100
min_overlap = 0.55

max_correct_distance = 2.5
max_correct_rotation = np.radians(2.5)


# define a registration system
reg = Registration(sampling_points,iterations,tolerance,max_translation,max_rotation)

# Load up the data
data_one = load_data("/home/jake/Desktop/holoocean_bags/scrape/1.pickle",1)
data_two = load_data("/home/jake/Desktop/holoocean_bags/scrape/2.pickle",2)
data = {1:data_one,2:data_two}

robots = {}
robot_list = [1,2]
for key in data.keys():
    robots[key] = Robot(data[key],SUBMAP_SIZE,BEARING_BINS,RANGE_BINS,MAX_RANGE,MAX_BEARING)

queue = []

for slam_step in range(67):
    print(slam_step)

    # step the robots forward
    for robot_id in robots.keys():
        robots[robot_id].step()

    if slam_step < 22: continue

    # search for loop closures
    #for robot_id_source in robots.keys():
    robot_id_source = 1
    
    for robot_id_target in robots.keys():
        if robot_id_target == robot_id_source: continue # do not search with self
        loops = search_for_loops(reg,robots,robot_id_source,robot_id_target,MAX_TREE_DIST,KNN)
        loops = reject_loops(loops,min_points,ratio_points,context_difference,min_overlap)
        loops = grade_loop_list(loops,max_correct_distance,max_correct_rotation)
        loop_ = keep_best_loop(loops)
        if loop_.status:
            loop_.place_loop(robots[robot_id_source].get_pose_gtsam())
            queue.append(loop_)
            pcm_result = verify_pcm(queue,2)
            
            plot_loop(loop_)


            

