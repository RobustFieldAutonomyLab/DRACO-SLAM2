import numpy as np

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

MIN_POINTS = 75
RATIO_POINTS = 2.0         
CONTEXT_DIFFERENCE = 100
MIN_OVERLAP = 0.65 # plane 0.85