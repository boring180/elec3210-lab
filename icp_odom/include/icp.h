/**
** Created by Zhijian QIAO.
** UAV Group, Hong Kong University of Science and Technology
** email: zqiaoac@connect.ust.hk
**/

#ifndef SRC_ICP_H
#define SRC_ICP_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

extern float last_distance;

class C_pair
{
    public:
    C_pair(float distance, pcl::PointXYZ X, pcl::PointXYZ Y)
    :distance(distance), X(X), Y(Y)
    {

    }

    pcl::PointXYZ getX()const
    {
        return this->X;
    }

    pcl::PointXYZ getY()const
    {
        return this->Y;
    }

    bool operator<(const C_pair& other)const
    {
        return this->distance < other.distance;
    }
    
    private:
    float distance;
    pcl::PointXYZ X;
    pcl::PointXYZ Y;
};

Eigen::Matrix4d icp_registration(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud,
                                 Eigen::Matrix4d init_guess);

#endif  // SRC_ICP_H
