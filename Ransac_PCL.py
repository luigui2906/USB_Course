# Luis Roldao - Universidad Simon Bolivar
# 30-Nov-2017
# In order to create the environment for running this code, remember to run the
# following command in your Anaconda command line:
# --> conda create --name 3dclass --channel ccordoba12 python=2.7 pcl python-pcl numpy matplotlib mayavi

import pcl
from mayavi import mlab
import numpy as np
import time


def read_pcd_file(input_filename):
    return pcl.load(input_filename).to_array()


def write_pointcloud_file(pointcloud, output_path):
    output_pointcloud = pcl.PointCloud()
    output_pointcloud.from_array(np.float32(pointcloud))
    output_pointcloud.to_file(output_path)
    return


def viewer_pointcloud(pointcloud):
    mlab.figure(bgcolor=(1, 1, 1))
    mlab.points3d(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], color=(0, 0, 0), mode='point')
    mlab.show()
    return


def viewer_original_vs_ransac_pointcloud_vs_plane(ransac_pcl, original_pcl, plane_model):
    sensor_range = 120.0
    mlab.figure(bgcolor=(1, 1, 1))
    x, y = np.ogrid[-sensor_range+50:sensor_range+50:1, -sensor_range:sensor_range:1]
    mlab.points3d(original_pcl[:, 0], original_pcl[:, 1], original_pcl[:, 2], color=(0, 0, 0), mode='point')
    mlab.points3d(ransac_pcl[:, 0], ransac_pcl[:, 1], ransac_pcl[:, 2], color=(1, 0, 0), mode='point')
    mlab.surf(x, y, (-plane_model[3] - (plane_model[0]*x) - (plane_model[1]*y)) / plane_model[2],
              color=(0.8, 0.8, 1), opacity=0.3)
    mlab.show()
    return


def random_sampling_consensus(pointcloud, numb_iterations, threshold):

    best_score = -1
    best_numb_inliers = -1

    for iterations in range(numb_iterations):

        while True:
            # Getting 3 random points from the pointcloud
            points = np.random.choice(pointcloud.shape[0], 3, replace=False)
            p1, p2, p3 = pointcloud[points[0]], pointcloud[points[1]], pointcloud[points[2]]

            # Calculating the equation of the plane Ax + By + Cz + D = 0 given by the 3 points
            a = ((p3[1] - p2[1]) * (p1[2] - p2[2])) - ((p3[2] - p2[2]) * (p1[1] - p2[1]))
            b = ((p3[2] - p2[2]) * (p1[0] - p2[0])) - ((p3[0] - p2[0]) * (p1[2] - p2[2]))
            c = ((p3[0] - p2[0]) * (p1[1] - p2[1])) - ((p3[1] - p2[1]) * (p1[0] - p2[0]))
            d = (-a * p1[0]) + (-b * p1[1]) + (-c * p1[2])

            # Verifying that the 3 points are not collinear
            if not (a == b == c == 0):
                break

        # -------------------------------------------------------------------------------------------------------------
        # Getting the score for the plane model formed by the 3 points randomly selected, the score for each individual
        # point is --> (score = max(threshold - distance_point2plane), 0). We add all the scores in order to have the
        # plane model score
        # t0 = time.time()
        pointwise_score = (np.maximum(threshold - (np.abs(a*pointcloud[:, 0] + b*pointcloud[:, 1] + c*pointcloud[:, 2]
                                                          + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)), 0))
        current_score = np.sum(pointwise_score)
        # tf1 = time.time() - t0

        # Another less efficient way to do it
        # t0 = time.time()
        # current_score_2 = 0
        # for point_index in range(pointcloud.shape[0]):
        #     distance_point2plane = abs(a*pointcloud[point_index][0] + b*pointcloud[point_index][1] +
        #                                c*pointcloud[point_index][2] + d) / np.sqrt(a**2 + b**2 + c**2)
        #     if threshold - distance_point2plane >= 0:
        #         current_score_2 += threshold - distance_point2plane
        # tf2 = time.time() - t0
        # -------------------------------------------------------------------------------------------------------------
        # print(tf1, tf2)

        if current_score > best_score:
            best_score = current_score
            best_pointwise_score = pointwise_score
            plane_model = np.array([a, b, c, d])

    # Extracting the points fitted from the original pointcloud and saving them to the ransac pointcloud
    ransac_pointcloud = pointcloud[np.where(best_pointwise_score > 0)[0]]

    return ransac_pointcloud, plane_model


def calculate_rotation_matrix(plane_model1, plane_model2):
    I = np.identity(3)
    normal_vector1 = normalize_vector(np.array([plane_model1[0], plane_model1[1], plane_model1[2]]))
    normal_vector2 = normalize_vector(np.array([plane_model2[0], plane_model2[1], plane_model2[2]]))
    cross_nv1nv2 = np.cross(normal_vector1, normal_vector2)
    v1, v2, v3 = cross_nv1nv2[0], cross_nv1nv2[1], cross_nv1nv2[2]
    Vperp = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])
    c = np.dot(normal_vector1, np.transpose(normal_vector2))
    Result = I + Vperp + (np.dot(Vperp, Vperp)*(1/1+c))
    Result = np.c_[Result, np.zeros(Result.shape[0])]
    Result = np.r_[Result, [np.array([0, 0, 0, 1])]]
    return Result


def transform_pointcloud(transf_matrix, pointcloud):
    return np.delete(np.transpose(np.dot(transf_matrix,
                                         np.transpose(np.c_[pointcloud, np.ones(pointcloud.shape[0])]))), 3, axis=1)


def normalize_vector(vector):
    return vector/np.linalg.norm(vector)


def main():
    # Exercise 1
    # input_path = 'C:/Users/lroldaoj/Documents/PycharmProjects/Ransac_PCL/pointcloud_example.pcd'
    # pointcloud = read_pcd_file(input_path)
    # viewer_pointcloud(pointcloud)
    # ransac_pointcloud, plane_model = random_sampling_consensus(pointcloud, 100, 0.2)
    # viewer_original_vs_ransac_pointcloud_vs_plane(ransac_pointcloud, pointcloud, plane_model)

    #Exercise 2
    pointcloud1 = read_pcd_file('C:/Users/lroldaoj/Documents/PycharmProjects/Ransac_PCL/PointCloud1.pcd')
    pointcloud2 = read_pcd_file('C:/Users/lroldaoj/Documents/PycharmProjects/Ransac_PCL/PointCloud2rot.pcd')
    viewer_original_vs_ransac_pointcloud_vs_plane(pointcloud1, pointcloud2, np.array([12.40711689, -11.24007034,
                                                                                      -568.33337402, -999.6496582]))
    ransac_pointcloud1, plane_model1 = random_sampling_consensus(pointcloud1, 100, 0.2)
    ransac_pointcloud2, plane_model2 = random_sampling_consensus(pointcloud2, 100, 0.2)
    Transf_matrix = calculate_rotation_matrix(plane_model1, plane_model2)
    print(Transf_matrix)
    pointcloud2_fixed = transform_pointcloud(Transf_matrix, pointcloud2)
    viewer_original_vs_ransac_pointcloud_vs_plane(pointcloud1, pointcloud2_fixed, plane_model1)


if __name__ == '__main__':
    main()