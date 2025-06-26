import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

class CommLink():
    def __init__(self):
        self.start_time = time.time()
        self.time_stamps = []
        self.usage = []
        self.icp_time = []
        self.usage_dict = {}
        self.time_dict = {}

    def log_time(self, time_this, time_type:str):
        if time_type not in self.time_dict:
            self.time_dict[time_type] = []
        self.time_dict[time_type].append(time_this)

    def log_message(self, bits:int, usage_type:str):
        """Log the usage of comms.

        Args:
            bits (int): the number of bits in the message
        """

        self.time_stamps.append(time.time()-self.start_time)
        self.usage.append(bits)
        if usage_type not in self.usage_dict:
            self.usage_dict[usage_type] = []
        self.usage_dict[usage_type].append(bits)

    def plot(self):
        """Plot the usage
        """

        plt.plot(self.time_stamps,self.usage)
        plt.show()

    def report(self,folder, mission: str, mode: str, study_step: int) -> None:
        """Log a report of the data usage

        Args:
            mission(str): the mission we are running 
            mode (int): the mode the mission was done in 
            study_step (int): the index of the mission for the study, the ith mission
        """

        data_log = [self.time_stamps,self.usage, self.usage_dict, self.time_dict]
        with open(f"{folder}/{mode}_{study_step}_comms.pickle", 'wb') as handle:
            pickle.dump(data_log, handle)