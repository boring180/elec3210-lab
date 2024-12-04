#include "ekf_slam.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cmath>

using namespace std;
using namespace Eigen;

// In case of w = 0, we need to avoid division by zero
static inline double safeDivide(double numerator, double denominator)
{
  const double epsilon = 1e-6;
  if (std::abs(denominator) < epsilon)
  {
    denominator = (denominator < 0) ? -epsilon : epsilon;
  }
  return numerator / denominator;
}

EKFSLAM::~EKFSLAM() {}

EKFSLAM::EKFSLAM(ros::NodeHandle &nh) : nh_(nh)
{
  //    initialize ros publisher
  lidar_sub =
      nh_.subscribe("/velodyne_points", 1, &EKFSLAM::cloudHandler, this);
  odom_sub = nh_.subscribe("/odom", 1, &EKFSLAM::odomHandler, this);
  map_cylinder_pub =
      nh_.advertise<visualization_msgs::MarkerArray>("/map_cylinder", 1);
  obs_cylinder_pub =
      nh_.advertise<visualization_msgs::MarkerArray>("/obs_cylinder", 1);
  odom_pub = nh_.advertise<nav_msgs::Odometry>("ekf_odom", 1000);
  path_pub = nh_.advertise<nav_msgs::Path>("ekf_path", 1000);
  scan_pub = nh_.advertise<sensor_msgs::PointCloud2>("current_scan", 1);
  map_pub = nh_.advertise<sensor_msgs::PointCloud2>("cloud_map", 1);
  laserCloudIn =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  mapCloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

  extractCylinder = std::make_shared<ExtractCylinder>(nh_);

  globalId = -1;
  /**
   * TODO: initialize the state vector and covariance matrix
   */
  mState = Eigen::VectorXd::Zero(3);  // x, y, yaw  3 + 2N
  mCov = Eigen::MatrixXd::Identity(3, 3); // 3 + 2N x 3 + 2N
  R = Eigen::MatrixXd::Identity(2, 2);   // process noise   2 x 2
  R.diagonal() << 1e4, 1e4;
  Q = Eigen::MatrixXd::Identity(2, 2);   // measurement noise 2 x 2
  Q.diagonal() << 1e1, 1e1;
  std::cout << "EKF SLAM initialized" << std::endl;
}

void EKFSLAM::run()
{
  ros::Rate rate(1000);
  while (ros::ok())
  {
    if (cloudQueue.empty() || odomQueue.empty())
    {
      rate.sleep();
      continue;
    }

    cloudQueueMutex.lock();
    cloudHeader = cloudQueue.front().first;
    laserCloudIn = parseCloud(cloudQueue.front().second);
    cloudQueue.pop();
    cloudQueueMutex.unlock();

    // find the cloest odometry message
    odomMutex.lock();
    auto odomIter = odomQueue.front();
    auto odomPrevIter = odomQueue.front();
    while (!odomQueue.empty() && odomIter != odomQueue.back() &&
           odomIter.first.stamp < cloudHeader.stamp)
    {
      odomPrevIter = odomIter;
      odomIter = odomQueue.front();
      odomQueue.pop();
    }
    odomMutex.unlock();

    if (firstFrame)
    {
      firstFrame = false;
      Twb = Eigen::Matrix4d::Identity();
      cloudHeaderLast = cloudHeader;
      continue;
    }

    auto odomMsg = odomIter == odomQueue.back() ? odomPrevIter : odomIter;
    Eigen::Vector2d ut = Eigen::Vector2d(odomMsg.second->twist.twist.linear.x,
                                         odomMsg.second->twist.twist.angular.z);
    double dt = (cloudHeader.stamp - cloudHeaderLast.stamp).toSec();

    timer.tic();
    // Extended Kalman Filter
    // 1. predict
    predictState(mState, mCov, ut, dt);
    // 2. update
    updateMeasurement();
    timer.toc();

    // publish odometry and map
    map_pub_timer.tic();
    accumulateMap();
    publishMsg();
    cloudHeaderLast = cloudHeader;

    rate.sleep();
  }
}

// Normalize the angle into range [-pi, pi]
double EKFSLAM::normalizeAngle(double angle)
{
  if (angle > M_PI)
  {
    angle -= 2 * M_PI;
  }
  else if (angle < -M_PI)
  {
    angle += 2 * M_PI;
  }
  return angle;
}

Eigen::MatrixXd EKFSLAM::jacobGt(const Eigen::VectorXd &state,
                                 Eigen::Vector2d ut, double dt)
{
  int num_state = state.rows();
  Eigen::MatrixXd Gt = Eigen::MatrixXd::Identity(num_state, num_state);
  /**
   * TODO: implement the Jacobian Gt
   */
  double v = ut(0);
  double w = ut(1);
  Gt(0, 2) = safeDivide(cos(state(2) + w * dt) - cos(state(2)), w) * v;
  Gt(1, 2) = safeDivide(sin(state(2) + w * dt) - sin(state(2)), w) * v;
  return Gt;
}

Eigen::MatrixXd EKFSLAM::jacobFt(const Eigen::VectorXd &state,
                                 Eigen::Vector2d ut, double dt)
{
  int num_state = state.rows();
  Eigen::MatrixXd Ft = Eigen::MatrixXd::Zero(num_state, 2);
  /**
   * TODO: implement the Jacobian Ft
   */
  double v = ut(0);
  double w = ut(1);
  Ft(0, 0) = -safeDivide(sin(state(2)) - sin(state(2) + w * dt), w);
  Ft(1, 0) = safeDivide(cos(state(2)) - cos(state(2) + w * dt), w);
  Ft(2, 1) = dt;
  return Ft;
}

Eigen::MatrixXd EKFSLAM::jacobB(const Eigen::VectorXd &state,
                                Eigen::Vector2d ut, double dt)
{
  int num_state = state.rows();
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(num_state, 2);
  double v = ut(0);
  double w = ut(1);
  B(0, 0) = -safeDivide(sin(state(2)) - sin(state(2) + w * dt), w);
  B(1, 0) = safeDivide(cos(state(2)) - cos(state(2) + w * dt), w);
  B(2, 1) = dt;
  return B;
}

void EKFSLAM::predictState(Eigen::VectorXd &state, Eigen::MatrixXd &cov,
                           Eigen::Vector2d ut, double dt)
{
  // Note: ut = [v, w]
  Eigen::MatrixXd Gt = jacobGt(state, ut, dt);
  Eigen::MatrixXd Ft = jacobFt(state, ut, dt);
  state = state + jacobB(state, ut, dt) * ut; // update state
  // std::cout<< "Gt" << Gt.rows() << " x " << Gt.cols() << endl; 
  // std::cout<< "cov" << cov.rows() << " x " << cov.cols() << endl; 
  // std::cout<< "Ft" << Ft.rows() << " x " << Ft.cols() << endl; 
  // std::cout<< "R" << R.rows() << " x " << R.cols() << endl; 
  cov = Gt * cov * Gt.transpose() + Ft * R * Ft.transpose(); // update
  // std::cout << "End update" << endl;
}

Eigen::Vector2d EKFSLAM::transform(const Eigen::Vector2d &p,
                                   const Eigen::Vector3d &x)
{
  Eigen::Vector2d p_t;
  p_t(0) = p(0) * cos(x(2)) - p(1) * sin(x(2)) + x(0);
  p_t(1) = p(0) * sin(x(2)) + p(1) * cos(x(2)) + x(1);
  return p_t;
}

void EKFSLAM::addNewLandmark(const Eigen::Vector2d &lm,
                             const Eigen::MatrixXd &InitCov)
{
  // add new landmark to mState and mCov
  /**
   * TODO: implement the function
   */
  int origSize = mState.size();

  mState.conservativeResize(origSize + 2);
  mState.tail(2) = lm;

  mCov.conservativeResize(origSize + 2, origSize + 2);
  mCov.block(0, origSize, origSize, 2).setZero();
  mCov.block(origSize, 0, 2, origSize).setZero();
  mCov.block<2, 2>(origSize, origSize) = InitCov;
  // std::cout << "Add new LM" << std::endl;
}

void EKFSLAM::accumulateMap()
{
  Eigen::Matrix4d Twb = Pose3DTo6D(mState.segment(0, 3));
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*laserCloudIn, *transformedCloud, Twb);
  *mapCloud += *transformedCloud;

  pcl::VoxelGrid<pcl::PointXYZ> voxelSampler;
  voxelSampler.setInputCloud(mapCloud);
  voxelSampler.setLeafSize(0.5, 0.5, 0.5);
  voxelSampler.filter(*mapCloud);
}

void EKFSLAM::updateMeasurement()
{
  cylinderPoints = extractCylinder->extract(
      laserCloudIn, cloudHeader);                 // 2D pole centers in the laser/body frame
  Eigen::Vector3d xwb = mState.block<3, 1>(0, 0); // pose in the world frame
  int num_landmarks =
      (mState.rows() - 3) / 2;         // number of landmarks in the state vector
  // cout << "Current LM:  " << num_landmarks << endl;
  int num_obs = cylinderPoints.rows(); // number of observations
  // cout << "Detect LM:  " << num_obs << endl;
  Eigen::VectorXi indices = Eigen::VectorXi::Ones(num_obs) *
                            -1; // indices of landmarks in the state vector
  for (int i = 0; i < num_obs; ++i)
  {
    Eigen::Vector2d pt_transformed = transform(cylinderPoints.row(i), xwb);
    // 2D pole center in the world frame
    // Implement the data association here, i.e., find the
    // corresponding landmark for each observation
    /**
     * TODO: data association
     *
     * **/
    // The distance of NN
    double distance = INFINITY;
    int lm_index = -1;
    for (int j = 0; j < num_landmarks; j++)
    {
      double challenger_distance = sqrt(pow((pt_transformed(0) - mState(3 + 2 * j)), 2) + pow((pt_transformed(1) - mState(4 + 2 * j)), 2));
      if(distance > challenger_distance)
      {
        distance = challenger_distance;
        lm_index = j;
      }
    }
    // Calculate the square euclidian distance and apply a threshhold
    // cout << "NN: " << distance << endl;
    // The greater the threshold is, it is less likely to construct a new lm
    if (distance < 3)
    {
      indices(i) = lm_index;
    }

    if (indices(i) == -1)
    {
      indices(i) = ++globalId;
      addNewLandmark(pt_transformed, Q);
    }
  }
  // cout << "Number of landmark: " << mState.size() - 3 << endl;
  // simulating bearing model
  Eigen::VectorXd z = Eigen::VectorXd::Zero(2 * num_obs);
  for (int i = 0; i < num_obs; ++i)
  {
    const Eigen::Vector2d &pt = cylinderPoints.row(i);
    z(2 * i, 0) = pt.norm();
    z(2 * i + 1, 0) = atan2(pt(1), pt(0));
  }
  // update the measurement vector
  num_landmarks = (mState.rows() - 3) / 2;
  // cout << "Updated LM:  " << num_landmarks << endl;
  // cout << mState << endl;
  // cout << mCov << endl;
  for (int i = 0; i < num_obs; ++i)
  {
    int idx = indices(i);
     double square_distance = pow((mState(0) - mState(3 + 2 * idx)), 2) + pow((mState(1) - mState(4 + 2 * idx)), 2);
    // std::cout << "Start update for lm " << idx << endl;
    if (idx == -1 || idx + 1 > num_landmarks)
      continue;
    const Eigen::Vector2d &landmark = mState.block<2, 1>(3 + idx * 2, 0);
    // Implement the measurement update here, i.e., update the state vector and
    // covariance matrix
    /**
     * TODO: measurement update
     */
    // std::cout<< "mState " << mState.size() << endl; 
    Eigen::MatrixXd lowHt(2, 5); // 2 x 5
    // std::cout<< "i " << i << endl; 
    double deltaX = mState(3 + 2 * idx) - mState(0);
    double deltaY = mState(4 + 2 * idx) - mState(1);
    double q = pow(deltaX, 2) + pow(deltaY, 2);
    double norm = sqrt(q);
    lowHt << -norm * deltaX , -norm * deltaY,            0, norm * deltaX , norm * deltaY,
                      deltaY,        -deltaX,           -q,        -deltaY,        deltaX;
    lowHt = lowHt / q;
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(5, 3 + 2 * num_landmarks); // 5 x 3 + 2N
    F(0, 0) = 1;
    F(1 ,1) = 1;
    F(2, 2) = 1;
    F(3, 2 * idx + 3) = 1;
    F(4, 2 * idx + 4) = 1;
    Eigen::MatrixXd H = lowHt * F; // 2 x 3 + 2N
    // std::cout<< "H" << H.rows() << " x " << H.cols() << endl; 
    // std::cout<< "mCov" << mCov.rows() << " x " << mCov.cols() << endl; 
    Eigen::MatrixXd Q_weighted = Q * square_distance;
    Eigen::MatrixXd K = mCov * H.transpose() * (H * mCov * H.transpose() + Q_weighted).inverse();
    Eigen::Vector2d z_i;
    z_i << z(2 * i, 0), z(1 + 2 * i, 0);
    double theta_hat = normalizeAngle(atan2(deltaY, deltaX) - mState(2));
    Eigen::Vector2d z_i_hat;
    z_i_hat << norm, theta_hat;
    mState = mState + K * (z_i  - z_i_hat);
    Eigen::MatrixXd In = Eigen::MatrixXd::Identity(mCov.rows(), mCov.rows());
    mCov = (In - K * H) * mCov;
    // cout << mCov << endl;
    // std::cout << "Updated for lm " << idx << endl;
    // cout << "mState: " << mState.transpose() << endl;
    // cout << "mCov: " << mCov.transpose() << endl;
    // cout << "K: " << K.transpose() << endl;
    // cout << "H: " << H.transpose() << endl;
    // cout << "Q: " << Q.transpose() << endl;
  }
  // ROS_INFO_STREAM("Covariance Matrix:\n" << mCov);
  // ROS_INFO_STREAM("mState Vector:\n" << mState);
  mState(2) = normalizeAngle(mState(2));
  ROS_INFO_STREAM("Number of lm:\n" << (mState.rows() - 3) / 2);
}

void EKFSLAM::publishMsg()
{
  // publish map cylinder
  visualization_msgs::MarkerArray markerArray;
  int num_landmarks = (mState.rows() - 3) / 2;
  for (int i = 0; i < num_landmarks; ++i)
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = cloudHeader.stamp;
    marker.ns = "map_cylinder";
    marker.id = i;
    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = mState(3 + i * 2, 0);
    marker.pose.position.y = mState(3 + i * 2 + 1, 0);
    marker.pose.position.z = 0.5;
    marker.pose.orientation.x = 0;
    marker.pose.orientation.y = 0;
    marker.pose.orientation.z = 0;
    marker.pose.orientation.w = 1;
    marker.scale.x = 0.5;
    marker.scale.y = 0.5;
    marker.scale.z = 1;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;

    markerArray.markers.push_back(marker);
  }
  map_cylinder_pub.publish(markerArray);

  int num_obs = cylinderPoints.rows();
  markerArray.markers.clear();
  for (int i = 0; i < num_obs; ++i)
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = cloudHeader.stamp;
    marker.ns = "obs_cylinder";
    marker.id = i;
    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.action = visualization_msgs::Marker::ADD;
    Eigen::Vector2d pt =
        transform(cylinderPoints.row(i).transpose(), mState.segment(0, 3));
    marker.pose.position.x = pt(0);
    marker.pose.position.y = pt(1);
    marker.pose.position.z = 0.5;
    marker.pose.orientation.x = 0;
    marker.pose.orientation.y = 0;
    marker.pose.orientation.z = 0;
    marker.pose.orientation.w = 1;
    marker.scale.x = 0.5;
    marker.scale.y = 0.5;
    marker.scale.z = 1;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;

    markerArray.markers.push_back(marker);
  }
  obs_cylinder_pub.publish(markerArray);

  //    publish odom
  nav_msgs::Odometry odom;
  odom.header.frame_id = "map";
  odom.child_frame_id = "base_link";
  odom.header.stamp = cloudHeader.stamp;
  odom.pose.pose.position.x = mState(0, 0);
  odom.pose.pose.position.y = mState(1, 0);
  odom.pose.pose.position.z = 0;
  Eigen::Quaterniond q(
      Eigen::AngleAxisd(mState(2, 0), Eigen::Vector3d::UnitZ()));
  q.normalize();
  odom.pose.pose.orientation.x = q.x();
  odom.pose.pose.orientation.y = q.y();
  odom.pose.pose.orientation.z = q.z();
  odom.pose.pose.orientation.w = q.w();
  odom_pub.publish(odom);

  //    publish path
  path.header.frame_id = "map";
  path.header.stamp = cloudHeader.stamp;
  geometry_msgs::PoseStamped pose;
  pose.header = odom.header;
  pose.pose = odom.pose.pose;
  path.poses.push_back(pose);
  path_pub.publish(path);

  ////    publish map
  sensor_msgs::PointCloud2 mapMsg;
  pcl::toROSMsg(*mapCloud, mapMsg);
  mapMsg.header.frame_id = "map";
  mapMsg.header.stamp = cloudHeader.stamp;
  map_pub.publish(mapMsg);

  ////    publish laser
  sensor_msgs::PointCloud2 laserMsg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr laserTransformed(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*laserCloudIn, *laserTransformed,
                           Pose3DTo6D(mState.segment(0, 3)).cast<float>());
  pcl::toROSMsg(*laserTransformed, laserMsg);
  laserMsg.header.frame_id = "map";
  laserMsg.header.stamp = cloudHeader.stamp;
  scan_pub.publish(laserMsg);

  map_pub_timer.toc();
  std::cout << "x: " << mState(0, 0) << " y: " << mState(1, 0)
            << " theta: " << mState(2, 0) * 180 / M_PI
            << ", time ekf: " << timer.duration_ms() << " ms"
            << ", map_pub: " << map_pub_timer.duration_ms() << " ms"
            << std::endl;
}

void EKFSLAM::cloudHandler(
    const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
  cloudQueueMutex.lock();
  std_msgs::Header cloudHeader = laserCloudMsg->header;
  cloudQueue.push(std::make_pair(cloudHeader, laserCloudMsg));
  cloudQueueMutex.unlock();
}

void EKFSLAM::odomHandler(const nav_msgs::OdometryConstPtr &odomMsg)
{
  odomMutex.lock();
  std_msgs::Header odomHeader = odomMsg->header;
  odomQueue.push(std::make_pair(odomHeader, odomMsg));
  odomMutex.unlock();
}

pcl::PointCloud<pcl::PointXYZ>::Ptr EKFSLAM::parseCloud(
    const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTmp(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*laserCloudMsg, *cloudTmp);
  // Remove Nan points
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloudTmp, *cloudTmp, indices);
  return cloudTmp;
}
