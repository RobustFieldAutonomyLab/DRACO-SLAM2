import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

class CommLink():
    def __init__(self):
        self.start_time = time.time()
        self.time_stamps = []
        self.usage = []

    def log_message(self, bits:int):
        """Log the usage of comms.

        Args:
            bits (int): the number of bits in the message
        """

        self.time_stamps.append(time.time()-self.start_time)
        self.usage.append(bits)

    def plot(self):
        """Plot the usage
        """

        plt.plot(self.time_stamps,self.usage)
        plt.show()

    def report(self,mission: str, mode: int, study_step: int) -> None:
        """Log a report of the data usage

        Args:
            mission(str): the mission we are running 
            mode (int): the mode the mission was done in 
            study_step (int): the index of the mission for the study, the ith mission
        """

        data_log = [self.time_stamps,self.usage]
        with open('results/'+mission+"/"+str(mode)+"_"+str(study_step)+'_comms.pickle', 'wb') as handle:
            pickle.dump(data_log, handle)