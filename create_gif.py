import os
import cv2
import glob
import imageio

out_name = "usmma_sim_1.gif"
log_path = "/home/jake/Desktop/holoocean_bags/DC-DRACO/animate/usmma/ALCS/1/"

image_path_list = sorted(glob.glob(log_path + "*png"),key=os.path.getmtime)
image_list = []

for image_path in image_path_list: 
    image = cv2.imread(image_path)
    image_list.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Convert to gif using the imageio.mimsave method
imageio.mimsave(log_path+out_name, image_list, fps=5)