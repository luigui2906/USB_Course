# Luis Roldao - Universidad Simon Bolivar
# 30-Nov-2017
# In order to create the environment for running this code, remember to run the
# following command in your Anaconda command line:
# --> conda create --name 3dclass --channel ccordoba12 python=2.7 pcl python-pcl numpy matplotlib mayavi

import pcl
from mayavi import mlab
import numpy as np
import time


# Read a .pcd file, just give the path to the file. The function will return the pointcloud as a numpy array.
def read_pcd_file(input_filename):
    return pcl.load(input_filename).to_array()


# Save your pointcloud as a .pcd file in order to use it in other # programs (cloudcompare for example).
def write_pointcloud_file(pointcloud, output_path):
    output_pointcloud = pcl.PointCloud()
    output_pointcloud.from_array(pointcloud)
    output_pointcloud.to_file(output_path)
    return


# To visualize the passed pointcloud.
def viewer_pointcloud(pointcloud):
    mlab.figure(bgcolor=(1, 1, 1))
    mlab.points3d(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], color=(0, 0, 0), mode='point')
    mlab.show()
    return


# To visualize two pointclouds (The original one and the one obtained after the Ransac normally) and the
# plane obtained by the Ransac all together.
def viewer_original_vs_ransac_pointcloud_vs_plane(ransac_pcl, original_pcl, plane_model):
    sensor_range = 120.0
    mlab.figure(bgcolor=(1, 1, 1))
    x, y = np.ogrid[-sensor_range:sensor_range:1, -sensor_range:sensor_range:1]
    mlab.points3d(original_pcl[:, 0], original_pcl[:, 1], original_pcl[:, 2], color=(0, 0, 0), mode='point')
    mlab.points3d(ransac_pcl[:, 0], ransac_pcl[:, 1], ransac_pcl[:, 2], color=(1, 0, 0), mode='point')
    mlab.surf(x, y, plane_function(plane_model[0], plane_model[1], plane_model[2], plane_model[3], x, y),
              color=(0.8, 0.8, 1), opacity=0.3)
    mlab.show()
    return


def plane_function(ransac_a, ransac_b, ransac_c, ransac_d, x, y):
    return (-ransac_d - (ransac_a*x) - (ransac_b*y)) / ransac_c


# --------------------------------------------------------------------------------------------------------------------
# This is the function to complete, it should receive a pointcloud (numpy array [x, y, z],[x, y, z]...),
# the number of iterations of the Ransac and the threshold to be used. It should return a new pointcloud
# numpy array with the points extracted by the Ransac and a numpy array with the variables of the plane
# (A, B, C, D) - Remember that the equation of the plane Ax+By+Cz+D=0 defines the plane itself.
def random_sampling_consensus(pointcloud, numb_iterations, threshold):


    # Take 3 random points from the pointcloud and extract the plane equation associated to them, remember
    # to check that the 3 points are not collinear.








    # Obtain a score for every point taking into account the distance from all the points in the pointcloud
    # to the extracted plane. Checkout how to calculate the distance from a point to a plane. The final score
    # for the iteration will be the addition of the score for every particular point





    # If the score of current iteration is better than precious scores obtained, update score and plane equation
    # since we have a new better fitting model with more inliers (possibly)





    # Once you have looped through all the numb_iterations of your Ransac and that you have your better fitting
    # model, then extract the inliers to create the new pointcloud (Ransac pointcloud that extracts the main plane)


    # Return the requested variables
    return ransac_pointcloud, plane_model
# --------------------------------------------------------------------------------------------------------------------


def main():
    input_path = # Give your input path in order to read the .pcd file
    pointcloud = read_pcd_file(input_path)
    viewer_pointcloud(pointcloud)
    ransac_pointcloud, plane_model = random_sampling_consensus(pointcloud, 100, 0.2)
    viewer_original_vs_ransac_pointcloud_vs_plane(ransac_pointcloud, pointcloud, plane_model)


if __name__ == '__main__':
    main()