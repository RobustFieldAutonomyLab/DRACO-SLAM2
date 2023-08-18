import pickle
import numpy as np
import matplotlib.pyplot as plt

# pull the data logs from mode 0
slam_log_0 = {}
comm_log_0 = {}
for step in range(100):
    for robot in range(1,4):
        with open("results/"+str(robot)+"_0_"+str(step)+".pickle", "rb") as input_file:
            slam_log_0[(robot,step)] = pickle.load(input_file)
  
    with open("results/0_"+str(step)+"_comms.pickle", "rb") as input_file:
        comm_log_0[step] = pickle.load(input_file)

# pull the data logs from mode 1
slam_log_1 = {}
comm_log_1 = {}
for step in range(100):
    for robot in range(1,4):
        with open("results/"+str(robot)+"_1_"+str(step)+".pickle", "rb") as input_file:
            slam_log_1[(robot,step)] = pickle.load(input_file)
        
    with open("results/1_"+str(step)+"_comms.pickle", "rb") as input_file:
        comm_log_1[step] = pickle.load(input_file)

# get the MSE and RMSE
print("MSE/RMSE")
for robot in range(1,4):
    mse = []
    rmse = []
    for step in range(100):
        mse.append(slam_log_0[(robot,step)]["mse"])
        rmse.append(slam_log_0[(robot,step)]["rmse"])      
    print(np.mean(mse),np.mean(rmse))

print("----------------")

for robot in range(1,4):
    mse = []
    rmse = []
    for step in range(100):
        mse.append(slam_log_1[(robot,step)]["mse"])
        rmse.append(slam_log_1[(robot,step)]["rmse"])      
    print(np.mean(mse),np.mean(rmse))

print("ICP Counts")
total = 0
good = 0
for robot in range(1,4):
    total += slam_log_0[(robot),0]["icp_count"]
    good += slam_log_0[(robot,0)]["icp_success_count"]
print(total, good, (good / total) * 100)

total = 0
good = 0
for robot in range(1,4):
    total += slam_log_1[(robot,0)]["icp_count"]
    good += slam_log_1[(robot,0)]["icp_success_count"]
print(total, good, (good / total) * 100)
print("----------------")
    

print("Mean run time / STD run time")
print("PCM")
pcm_run_times = []
for robot in range(1,4):
    for step in range(100):
        pcm_run_times += slam_log_0[(robot,step)]["pcm_run_time"]
print(np.mean(pcm_run_times),np.std(pcm_run_times))

print("Draco Reg")
draco_run_times = []
for robot in range(1,4):
    for step in range(100):
        draco_run_times += slam_log_0[(robot,step)]["draco_reg_time"]
print(np.mean(draco_run_times),np.std(draco_run_times))

print("ALCS Reg")
alcs_reg_time = []
for robot in range(1,4):
    for step in range(100):
        alcs_reg_time += slam_log_1[(robot,step)]["alcs_reg_time"]
print(np.mean(alcs_reg_time),np.std(alcs_reg_time))

print("ALCS Pred")
alcs_pred_time = []
for robot in range(1,4):
    for step in range(100):
        alcs_pred_time += slam_log_1[(robot,step)]["alcs_run_time"]
print(np.mean(alcs_pred_time),np.std(alcs_pred_time))


print("Total run times")
draco_total_times = []
for step in range(100):
    draco_total = 0
    for robot in range(1,4):
        draco_total += np.sum(slam_log_0[(robot,step)]["draco_reg_time"])
        draco_total += np.sum(slam_log_0[(robot,step)]["pcm_run_time"])
    draco_total_times.append(draco_total)

print("Draco total")
print(np.mean(draco_total_times),np.std(draco_total_times))

alcs_total_times = []
for step in range(100):
    alcs_total = 0
    for robot in range(1,4):
        alcs_total += np.sum(slam_log_1[(robot,step)]["alcs_reg_time"])
        alcs_total += np.sum(slam_log_1[(robot,step)]["alcs_run_time"])
    alcs_total_times.append(alcs_total)
print("ALCS Total")
print(np.mean(alcs_total_times),np.std(alcs_total_times))


def get_team_uncertainty(log:dict) -> np.array:
    """Get the total uncertainty of the team members in a multi-robot system.

    Args:
        log (dict): the incoming data log

    Returns:
        np.array: an array of the covariance matrix determinants
    """
    
    steps = max(log[list(log.keys())[0]].keys())
    output = []
    for i in range(steps):
        val = 0
        for robot in log.keys():
            val += np.linalg.det(log[robot][i])
        output.append(val)

    return output

print("team uncertainty")
print("DRACO uncertainty")
draco_uncertainty = {1:[],2:[],3:[]}
for step in range(100):
    for robot in range(1,4):
        draco_uncertainty[robot].append(np.sum(get_team_uncertainty(slam_log_0[(robot,step)]["covariance"])))
for robot in range(1,4):
    print(np.mean(draco_uncertainty[robot]),np.std(draco_uncertainty[robot]))

print("ALCS uncertainty")
alcs_uncertainty = {1:[],2:[],3:[]}
for step in range(100):
    for robot in range(1,4):
        alcs_uncertainty[robot].append(np.sum(get_team_uncertainty(slam_log_1[(robot,step)]["covariance"])))
for robot in range(1,4):
    print(np.mean(alcs_uncertainty[robot]),np.std(alcs_uncertainty[robot]))

print("my uncertainty")
print("Draco")
draco_self_uncertainty = {1:[],2:[],3:[]}
for step in range(100):
    for robot in range(1,4):
        uncertainty = 0
        for cov in slam_log_0[(robot,step)]["my_covariance"]: uncertainty += np.linalg.det(cov)
        draco_self_uncertainty[robot].append(uncertainty)
for robot in range(1,4):
    print(np.mean(draco_self_uncertainty[robot]),np.std(draco_self_uncertainty[robot]))
    
print("ALCS")
alcs_self_uncertainty = {1:[],2:[],3:[]}
for step in range(100):
    for robot in range(1,4):
        uncertainty = 0
        for cov in slam_log_1[(robot,step)]["my_covariance"]: uncertainty += np.linalg.det(cov)
        alcs_self_uncertainty[robot].append(uncertainty)
for robot in range(1,4):
    print(np.mean(alcs_self_uncertainty[robot]),np.std(alcs_self_uncertainty[robot]))
