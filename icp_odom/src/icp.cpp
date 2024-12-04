/**
** Created by Zhijian QIAO.
** UAV Group, Hong Kong University of Science and Technology
** email: zqiaoac@connect.ust.hk
**/
#include "icp.h"

#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <Eigen/Dense>

#include "parameters.h"

float last_distance = 0.0;

/*
Eigen::Matrix4d icp_registration(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud,
                                 Eigen::Matrix4d init_guess) {
  // This is an example of using pcl::IterativeClosestPoint to align two point
  // clouds In your project, you should implement your own ICP algorithm!!!

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(src_cloud);
  icp.setInputTarget(tar_cloud);
  icp.setMaximumIterations(params::max_iterations);  // set maximum iteration
  icp.setTransformationEpsilon(1e-6);  // set transformation epsilon
  icp.setMaxCorrespondenceDistance(
      params::max_distance);  // set maximum correspondence distance
  pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
  icp.align(aligned_cloud, init_guess.cast<float>());

  Eigen::Matrix4d transformation = icp.getFinalTransformation().cast<double>();
  return transformation;
}
*/
// TODO: Implement your own ICP algorithm here
Eigen::Matrix4d icp_registration(pcl::PointCloud<pcl::PointXYZ>::Ptr
src_cloud,
                                  pcl::PointCloud<pcl::PointXYZ>::Ptr
                                  tar_cloud, Eigen::Matrix4d init_guess)
{
  // Read the init_guess
  Eigen::Matrix4f transformation = init_guess.cast<float>();
  // Start iteration
  for(int i = 0; i < params::max_iterations; i++)
  {
    // Apply current transformation to source cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*src_cloud, *transformed_cloud, transformation);

    // Find Correspond(Nearest neighbour)
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(tar_cloud);
    std::vector<C_pair> correspondences;
    // For evaluation
    float distance_sum = 0;
    int point_count = 0;
    for(size_t j = 0; j < src_cloud->points.size(); j++)
    {
      std::vector<int> index(1);
      std::vector<float> distance(1);

      if(!kdtree.nearestKSearch(transformed_cloud->points[j], 1, index, distance))
        continue;
      
      if(distance[0] < (params::max_distance))
      {
        C_pair correspound(distance[0], transformed_cloud->points[j], tar_cloud->points[index[0]]);
        correspondences.push_back(correspound);
      }
      distance_sum += distance[0];
    }

    // Form the Correspound cloud with filtered points
    pcl::PointCloud<pcl::PointXYZ>::Ptr X_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr Y_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // Sort by distance
    std::sort(correspondences.begin(), correspondences.end());
    point_count = correspondences.size();
    for(int j = 0; j < correspondences.size() * 0.85; j++)
    {
      X_cloud->push_back(correspondences[j].getX());
      Y_cloud->push_back(correspondences[j].getY());
    }

    // Calculate the centroid x0 and y0
    Eigen::Vector4f x0;
    pcl::compute3DCentroid(*X_cloud, x0);
    Eigen::Vector4f y0;
    pcl::compute3DCentroid(*Y_cloud, y0);

    // Calculate the rotation
    // Calculate the difference to controid matrix
    Eigen::MatrixXf X = X_cloud->getMatrixXfMap(3, 4, 0);
    X = X.colwise() - x0.head(3);
    Eigen::MatrixXf Y = Y_cloud->getMatrixXfMap(3, 4, 0);
    Y = Y.colwise() - x0.head(3);

    // Calculate SVD of XY^T
    Eigen::MatrixXf XYT = X * Y.transpose();
    Eigen::JacobiSVD<Eigen::MatrixXf> XYT_SVD(XYT, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Calculate rotation
    Eigen::Matrix3f rotation = XYT_SVD.matrixV() * XYT_SVD.matrixU().transpose();
    // We need only one axis
    rotation(0, 2) = 0;
    rotation(1, 2) = 0;
    rotation(2, 0) = 0;
    rotation(2, 1) = 0;
    rotation(2, 2) = 1;
    // Form the transformation
    Eigen::Matrix4f iteration_transformation = Eigen::Matrix4f::Identity();
    iteration_transformation.block<3,3>(0,0) = rotation;

    // Calculate translation
    x0 = iteration_transformation * x0;
    Eigen::Vector4f centroidDiff = y0 - x0;
    // We don't need Z axis translation
    centroidDiff(2) = 0;
    iteration_transformation.block<3,1>(0, 3) = centroidDiff.head(3);

    // Apply the transformation
    transformation = iteration_transformation * transformation;
    // For debugging
    // ROS_INFO_STREAM("Matrix:\n" << transformation);
    // For early End
    if(last_distance > distance_sum && ((last_distance - distance_sum) / last_distance) < 0.001)
    {
      //ROS_INFO_STREAM("Matrix:\n" << transformation);
      ROS_INFO("Iteration: %d Distance: %f #Valid point: %d / %ld", i, distance_sum, point_count, tar_cloud->size());
      break; // End loop if the distance change is less than 0.001
    }
    last_distance = distance_sum;
  }
  last_distance = 0.0;
  // Return the transformation
  return transformation.cast<double>();
}
