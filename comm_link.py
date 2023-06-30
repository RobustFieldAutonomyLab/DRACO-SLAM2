import time
import numpy as np
import matplotlib.pyplot as plt

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

    def report(self):
        """Output the details
        """

        print(np.sum(self.usage))