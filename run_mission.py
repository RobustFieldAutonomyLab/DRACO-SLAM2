import pickle
import gtsam
import numpy as np
import matplotlib.pyplot as plt

from utils import load_data,search_for_loops,reject_loops,grade_loop_list,plot_loop,keep_best_loop,verify_pcm
from robot import Robot
from registration import Registration
from loop_closure import LoopClosure
from comm_link import CommLink

from config.usmma import *

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
loop_list = []

comm_link = CommLink()

for slam_step in range(63):
    print(slam_step)

    # step the robots forward
    for robot_id in robots.keys():
        robots[robot_id].step()

    # search for loop closures
    for robot_id_source in robots.keys():
        for robot_id_target in robots.keys():
            if robot_id_target == robot_id_source: continue # do not search with self
            loops = search_for_loops(reg,robots,comm_link,robot_id_source,robot_id_target,MAX_TREE_DIST,KNN)
            loops = reject_loops(loops,min_points,ratio_points,context_difference,min_overlap)
            loops = grade_loop_list(loops,max_correct_distance,max_correct_rotation)

            if len(loops) == 0: continue
            loop_ = keep_best_loop(loops)
            loop_.place_loop(robots[robot_id_source].get_pose_gtsam())

            if loop_.status:
                robots[robot_id_source].add_loop_to_pcm_queue(loop_)
                valid_loops = robots[robot_id_source].do_pcm()
                for valid in valid_loops: 
                    valid.source_robot_id = robot_id_source
                    valid.target_robot_id = robot_id_target
                    loop_list.append(valid)


comm_link.plot()

temp = robots[1].truth
poses_one = []
for row in temp:
    poses_one.append([row.x(),row.y()])
poses_one = np.array(poses_one)

temp = robots[2].truth
poses_two = []
for row in temp:
    poses_two.append([row.x(),row.y()])
poses_two = np.array(poses_two)

plt.figure(figsize=(20,10))
plt.scatter(poses_one[:,0],-poses_one[:,1],c="blue")
plt.plot(poses_one[:,0],-poses_one[:,1],c="blue")
plt.scatter(poses_two[:,0],-poses_two[:,1],c="red")
plt.plot(poses_two[:,0],-poses_two[:,1],c="red")

for loop in loop_list:
    i,j = loop.source_key,loop.target_key
    source_robot, target_robot = loop.source_robot_id,loop.target_robot_id
    print(source_robot,target_robot)

    x1,y1 = robots[source_robot].truth[i].x(), robots[source_robot].truth[i].y()
    x2,y2 = robots[target_robot].truth[j].x(), robots[target_robot].truth[j].y()
    
    plt.plot([x1,x2],[-y1,-y2],color="black")
    plt.scatter([x1],[-y1],color="black",marker="*",s=150)
    plt.scatter([x2],[-y2],color="black",marker="*",s=150)


plt.axis("square")
plt.show()





            



            

