import cv2
import gtsam
import numpy as np
from scipy.optimize import shgo, differential_evolution

from slam.utils import gtsam_to_numpy,numpy_to_gtsam,transform_points

from bruce_slam import pcl

class GI_ICP():
    def __init__(self,sampling_points,iterations,tolerance,max_translation,max_rotation) -> None:
        self.sampling_points = sampling_points
        self.iterations = iterations
        self.tolerance = tolerance
        self.pose_bounds = np.array(
            [
                [-max_translation, max_translation],
                [-max_translation, max_translation],
                [-max_rotation, max_rotation],
            ]
        )

        self.icp = pcl.ICP()
        self.icp.loadFromYaml("config/icp.yaml")
        
    def global_pose_optimization_routine(
            self, source_points, source_pose, target_points, target_pose, point_noise=0.5
        ):
            """Build a routine for SHGO to optimize
            source_points: the source point cloud
            source_pose: the frame for the source points
            target_points: the target point cloud
            target_pose: target points pose initial guess
            returns: a functon to be called and optimized by SHGO
            """

            # a container for the poses tested by the optimizer
            pose_samples = []

            # Build a grid that will fit the target points
            xmin, ymin = np.min(target_points, axis=0) - 2 * point_noise
            xmax, ymax = np.max(target_points, axis=0) + 2 * point_noise
            resolution = point_noise / 10.0
            xs = np.arange(xmin, xmax, resolution)
            ys = np.arange(ymin, ymax, resolution)
            target_grids = np.zeros((len(ys), len(xs)), np.uint8)

            # conver the target points to a grid
            r = np.int32(np.round((target_points[:, 1] - ymin) / resolution))
            c = np.int32(np.round((target_points[:, 0] - xmin) / resolution))
            r = np.clip(r, 0, target_grids.shape[0] - 1)
            c = np.clip(c, 0, target_grids.shape[1] - 1)
            target_grids[r, c] = 255

            # dilate this grid
            dilate_hs = int(np.ceil(point_noise / resolution))
            dilate_size = 2 * dilate_hs + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_size, dilate_size), (dilate_hs, dilate_hs)
            )
            target_grids = cv2.dilate(target_grids, kernel)

            def subroutine(x):
                """The subroutine to be run at every step by the scipy optimizer
                x: the pose of the source points, [x, y, theta]
                """

                # conver this pose to a gtsam object
                delta = numpy_to_gtsam(x)
                sample_source_pose = source_pose.compose(delta)
                sample_transform = target_pose.between(sample_source_pose)

                # transform the source points using sample_transform
                points = transform_points(source_points, sample_transform)

                # convert the points into discrete grid coords
                r = np.int32(np.round((points[:, 1] - ymin) / resolution))
                c = np.int32(np.round((points[:, 0] - xmin) / resolution))

                # get the points that are in an occupied grid cell
                inside = (
                    (0 <= r)
                    & (r < target_grids.shape[0])
                    & (0 <= c)
                    & (c < target_grids.shape[1])
                )

                # tabulate cost, the number of source points inside an occupied cell
                cost = -np.sum(target_grids[r[inside], c[inside]] > 0)

                # log the pose sample
                pose_samples.append(np.r_[gtsam_to_numpy(sample_source_pose), cost])

                return cost

            return subroutine, pose_samples

    def global_pose_optimization_execute(self, subroutine, bounds=None, optimizer=1) -> np.array:
            """Using the provided subroutine, use scipy SHGO to find a pose
            Args:
                subroutine (function): the function to be minimized
                bounds (_type_, optional): _description_. Defaults to None.
            Returns:
                np.array: the minimizer result
            """

            if bounds is None:
                bounds = self.pose_bounds

            if optimizer == 1:
                return differential_evolution(
                    func=subroutine,
                    bounds=bounds,
                    seed=1
                )
            
            else:
                return shgo(
                    func=subroutine,
                    bounds=bounds,
                    n=self.sampling_points,
                    iters=self.iterations,
                    sampling_method="sobol",
                    minimizer_kwargs={"options": {"ftol": self.tolerance}},
                )
    
    def initialize(self, source_points : np.array, target_points : np.array) -> list:
        """Perform global initilization of ICP

        Args:
            source_points (np.array): source point cloud 
            target_points (np.array): target point cloud

        Returns:
            list[bool,np.array]: status flag and the optimizer result, the pose [x,y,theta]
        """
         
        # build the subroutine and run the global optimizer to perform global ICP
        subroutine, _ = self.global_pose_optimization_routine(
                                                            source_points,
                                                            gtsam.Pose2(0, 0, 0),
                                                            target_points,
                                                            gtsam.Pose2(0, 0, 0),
                                                        )
        global_result = self.global_pose_optimization_execute(subroutine)
        return global_result.success, global_result
    
    def refine(self, source_points : np.array, target_points : np.array) -> list:
        """Refine the transform between a pair of clouds using standard ICP. Only 
        to be applied after the global init.

        Args:
            source_points (np.array): source point cloud 
            target_points (np.array): target point cloud

        Returns:
            list[bool]: status and transform estimate
        """
        icp_status, icp_transform = self.icp.compute(source_points, target_points, np.eye(3))
        return icp_status, icp_transform
    
    def overlap(self, source_points : np.array, target_points : np.array) ->list:
        """Get the overlap between a pair of point clouds

        Args:
            source_points (np.array): source point cloud 
            target_points (np.array): target point cloud

        Returns:
            list[float,float]: the overlap and fit score between the clouds
        """
        
        idx, distances = pcl.match(source_points, target_points, 1, 0.5)
        overlap = len(idx[idx != -1]) / len(idx[0])
        fit_score = np.mean(distances[distances < float("inf")])
        return overlap, fit_score
