import gtsam
import numpy as np

from icp import GI_ICP
from utils import numpy_to_gtsam, transform_points, rot_to_euler
from loop_closure import LoopClosure

class Registration():
    """A class to handle the registration between point clouds
    """

    def __init__(self,sampling_points,iterations,tolerance,max_translation,max_rotation) -> None:
        self.icp = GI_ICP(sampling_points,iterations,tolerance,max_translation,max_rotation)

    def evaluate(self,loop:LoopClosure,
                 min_points:int, ratio_points:float, 
                 context_difference:int, min_overlap:float) -> LoopClosure:
        """Evaluate a potential loop closure using ICP and several outlier rejection methods.

        Args:
            loop (LoopClosure): the loop closure object
            min_points (int): the min points required in each cloud
            ratio_points (float): the ratio of points required between the clouds
            context_difference (int): the max difference in scan context images
            min_overlap (float): the minimum required overlap after ICP

        Returns:
            LoopClosure: the loop closure object, post evaluation
        """

        # 1. check the count
        loop.count = (len(loop.source_points), len(loop.target_points))
        if min_points != -1:
            if len(loop.source_points) < min_points or len(loop.target_points) < min_points:
                loop.status = False
                loop.message = "Low points: " + str(len(loop.source_points)) + "," + str(len(loop.target_points))
                return loop
            
        # 2. check the ratio 
        ratio_1 = len(loop.source_points) / len(loop.target_points)
        ratio_2 = len(loop.target_points) / len(loop.source_points)
        loop.ratio = (ratio_1, ratio_2)
        if ratio_points != -1:
            if ((len(loop.source_points) / len(loop.target_points)) < (1 / ratio_points) or
                    (len(loop.source_points) / len(loop.target_points)) > ratio_points):
                loop.status = False
                loop.message = "Ratio wrong: " + str(ratio_1) + "," + str(ratio_2)
                return loop
            
        # 3. check the context images
        loop.context_diff = np.sum(abs(loop.source_context - loop.target_context))
        if context_difference != -1:
            if np.sum(abs(loop.source_context - loop.target_context)) > context_difference:
                loop.status = False
                loop.message = "High context differnce: " + str(np.sum(abs(loop.source_context - loop.target_context)))
                return loop
            
        # use the global optimizer to init the ICP call
        init_status, go_icp_result = self.icp.initialize(loop.source_points,loop.target_points)
        if init_status == False:
            loop.status = False
            loop.message = "GI Failure"
            return loop

        # update the points using the transform from the global init above
        loop.source_points_init = transform_points(loop.source_points,numpy_to_gtsam(go_icp_result.x))
        loop.gi_transform = numpy_to_gtsam(go_icp_result.x)

        # 4. check the overlap and get the fit score
        overlap, fit_score = self.icp.overlap(loop.source_points_init,loop.target_points)
        loop.overlap = overlap
        if min_overlap != -1:
            if overlap < min_overlap:
                loop.status = False
                loop.message = "Low overlap: " + str(overlap)
                return loop
            
        # populate the fit score
        loop.fit_score = fit_score

        # refine the transform estimate using standard ICP
        icp_status, icp_transform = self.icp.refine(loop.source_points_init, loop.target_points)

        if icp_status == False:
            loop.status = False
            loop.message = "ICP Failure"
            return loop

        # update the point cloud
        loop.reg_points = loop.source_points_init.dot(icp_transform[:2, :2].T) + np.array(
                                                [icp_transform[0][2], icp_transform[1][2]])
        loop.icp_transform = gtsam.Pose2(icp_transform[0][2], icp_transform[1][2], rot_to_euler(icp_transform[:2, :2]))
        loop.message =("Count: " + str(loop.count)
                        + " Ratio: " + str(loop.ratio) 
                        + " Context Diff: " + str(loop.context_diff) 
                        + " Overlap: " + str(overlap))
        return loop
