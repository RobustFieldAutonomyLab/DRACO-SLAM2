import gtsam
import numpy as np

from slam.icp import GI_ICP
from sklearn.covariance import MinCovDet
from scipy.spatial.transform import Rotation
from slam.utils import numpy_to_gtsam, transform_points, rot_to_euler, grade_loop
from slam.loop_closure import LoopClosure
import time
import matplotlib.pyplot as plt


class Registration():
    """A class to handle the registration between point clouds
    """

    def __init__(self, config) -> None:
        sampling_points = config['sampling_points']
        iterations = config['iterations']
        tolerance = config['tolerance']
        max_translation = config['max_translation']
        max_rotation = config['max_rotation']
        self.icp = GI_ICP(sampling_points, iterations, tolerance, max_translation, max_rotation)
        self.icp_two = GI_ICP(sampling_points, iterations, tolerance, max_translation, np.radians(max_rotation))

    def predict(self, points: np.array, samples: int) -> np.array:
        """Predict the covariance if we call ICP on this point cloud

        Args:
            points (np.array): the point cloud
            samples (int): the number of samples to get a cov matrix

        Returns:
            np.array: the 3x3 covariance matrix
        """

        arr = []

        # loop to generate some samples
        for j in range(samples):

            # define the clouds
            source_cloud = np.array(points)
            target_cloud = np.array(points)

            # check that the source cloud has any points
            if len(source_cloud) > 1:
                dx = np.random.normal(0, 1.0)  # generate a random x
                dy = np.random.normal(0, 1.0)  # generate a random y
                da = np.random.normal(0, np.radians(1.0))  # generate a random theta
                step = gtsam.Pose2(dx, dy, da)  # compile as a gtsam pose
                T = step.matrix().astype(np.float32)  # convert to a matrix
                source_cloud = source_cloud.dot(T[:2, :2].T) + T[:2, 2]  # apply the transform to the source cloud

                # get the ICP transform
                _, T = self.icp.refine(source_cloud, target_cloud)
                source_cloud = source_cloud.dot(T[:2, :2].T) + T[:2, 2]

                # log the ICP output for covariance estimation
                r1 = T[:2, :2].T
                r = np.eye(3)
                r[0][0], r[0][1], r[1][0], r[1][1] = r1[0][0], r1[0][1], r1[1][0], r1[1][1]
                arr.append([T[1][2], T[0][2], Rotation.from_matrix(r).as_euler("xyz")[2]])

        # check if the results can support covariance estimation
        if (len(arr) > 2):
            cov = MinCovDet(random_state=1).fit(arr).covariance_
            return cov
        else:
            return None

    def evaluate_with_guess(self, loop: LoopClosure, source_pose) -> LoopClosure:

        # use the global optimizer to init the ICP call
        init_status, go_icp_result = self.icp_two.initialize(loop.source_points, loop.target_points)
        if init_status == False:
            loop.status = False
            loop.message = "GI Failure"
            return loop

        loop.source_points_init = transform_points(loop.source_points, numpy_to_gtsam(go_icp_result.x))
        loop.gi_transform = numpy_to_gtsam(go_icp_result.x)

        icp_status, icp_transform = self.icp.refine(loop.source_points_init, loop.target_points)
        loop.reg_points = loop.source_points_init.dot(icp_transform[:2, :2].T) + np.array(
            [icp_transform[0][2], icp_transform[1][2]])
        loop.icp_transform = gtsam.Pose2(icp_transform[0][2], icp_transform[1][2], rot_to_euler(icp_transform[:2, :2]))

        initial_estimate = loop.source_pose.between(loop.target_pose)
        updated_estimate = initial_estimate.compose(loop.gi_transform)
        updated_estimate = updated_estimate.compose(loop.icp_transform)
        new_target = loop.source_pose.compose(updated_estimate)
        loop.estimated_transform = loop.source_pose.between(new_target)
        # loop.estimated_transform = updated_estimate
        loop.overlap, loop.fit_score = self.icp_two.overlap(loop.reg_points, loop.target_points)
        loop.cov = np.eye(3) * loop.fit_score * 20.0

        true_between = loop.true_source.between(loop.true_target)
        '''print(loop.overlap)
        print(true_between.x(), true_between.y(), np.degrees(true_between.theta()))
        print(loop.estimated_transform.x(), loop.estimated_transform.y(),np.degrees(loop.estimated_transform.theta()))
        print("-------------")
              
        plt.scatter(loop.source_points[:,0],loop.source_points[:,1],c="orange")
        plt.scatter(loop.reg_points[:,0],loop.reg_points[:,1],c="red")
        plt.scatter(loop.target_points[:,0],loop.target_points[:,1],c="blue")
        plt.title(str(loop.overlap))
        plt.axis("square")
        plt.show()'''

        return loop

        '''true_between = loop.true_source.between(loop.true_target)
        test_between = updated_estimate.between(true_between)
        dist = np.sqrt(test_between.x()**2 + test_between.y()**2)
        rot = np.degrees(abs(test_between.theta()))
        overlap = self.icp_two.overlap(loop.reg_points,loop.target_points)'''

    def evaluate(self, loop: LoopClosure,
                 min_points: int, ratio_points: float,
                 context_difference: int, min_overlap: float, alt=False) -> LoopClosure:
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

        start_time = time.time()

        # 1. check the count
        loop.count = (len(loop.source_points), len(loop.target_points))
        if min_points != -1:
            if len(loop.source_points) < min_points or len(loop.target_points) < min_points:
                loop.status = False
                loop.message = "Low points: " + str(len(loop.source_points)) + "," + str(len(loop.target_points))
                return loop

        # # 2. check the ratio
        ratio_1 = len(loop.source_points) / len(loop.target_points)
        ratio_2 = len(loop.target_points) / len(loop.source_points)
        loop.ratio = (ratio_1, ratio_2)
        # if ratio_points != -1:
        #     if ((len(loop.source_points) / len(loop.target_points)) < (1 / ratio_points) or
        #             (len(loop.source_points) / len(loop.target_points)) > ratio_points):
        #         loop.status = False
        #         loop.message = "Ratio wrong: " + str(ratio_1) + "," + str(ratio_2)
        #         return loop

        # 3. check the context images
        loop.context_diff = np.sum(abs(loop.source_context - loop.target_context))
        # if context_difference != -1:
        #     if np.sum(abs(loop.source_context - loop.target_context)) > context_difference:
        #         loop.status = False
        #         loop.message = "High context differnce: " + str(np.sum(abs(loop.source_context - loop.target_context)))
        #         return loop

        # use the global optimizer to init the ICP call
        if alt:
            init_status, go_icp_result = self.icp_two.initialize(loop.source_points, loop.target_points)
        else:
            init_status, go_icp_result = self.icp.initialize(loop.source_points, loop.target_points)
        if init_status == False:
            loop.status = False
            loop.message = "GI Failure"
            return loop

        # update the points using the transform from the global init above
        loop.source_points_init = transform_points(loop.source_points, numpy_to_gtsam(go_icp_result.x))
        loop.gi_transform = numpy_to_gtsam(go_icp_result.x)

        # 4. check the overlap and get the fit score
        if alt:
            overlap, fit_score = self.icp_two.overlap(loop.source_points_init, loop.target_points)
        else:
            overlap, fit_score = self.icp.overlap(loop.source_points_init, loop.target_points)
        loop.overlap = overlap
        if min_overlap != -1:
            if overlap < min_overlap:
                loop.status = False
                loop.message = "Low overlap: " + str(overlap)
                return loop

        # populate the fit score
        loop.fit_score = fit_score
        loop.cov = np.eye(3) * loop.fit_score * 20.0

        # refine the transform estimate using standard ICP
        if alt:
            icp_status, icp_transform = self.icp_two.refine(loop.source_points_init, loop.target_points)
        else:
            icp_status, icp_transform = self.icp.refine(loop.source_points_init, loop.target_points)

        if icp_status == False:
            loop.status = False
            loop.message = "ICP Failure"
            return loop

        # update the point cloud
        loop.reg_points = loop.source_points_init.dot(icp_transform[:2, :2].T) + np.array(
            [icp_transform[0][2], icp_transform[1][2]])
        loop.icp_transform = gtsam.Pose2(icp_transform[0][2], icp_transform[1][2], rot_to_euler(icp_transform[:2, :2]))
        loop.message = ("Count: " + str(loop.count)
                        + " Ratio: " + str(loop.ratio)
                        + " Context Diff: " + str(loop.context_diff)
                        + " Overlap: " + str(overlap))
        return loop
