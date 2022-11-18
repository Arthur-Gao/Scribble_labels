#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <map>
#include<unordered_map>

//pcl
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/impl/search.hpp>

#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "semantic_kitti_api.h"

typedef pcl::PointXYZINormal Point_T;
typedef pcl::PointCloud<Point_T>::Ptr pcTPtr;
typedef pcl::PointCloud<Point_T> pcT;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr pcPtr;

std::string outputDirName;

struct pc_box {
    Eigen::Vector3f  translation;
    Eigen::Quaternionf rotation;
    double width;
    double height;
    double depth;
};

int id = 0;

bool write_pcd_file(const std::string &fileName, pcTPtr &pointCloud, bool as_binary = true)
{
    if (as_binary)
    {
        if (pcl::io::savePCDFileBinary(fileName, *pointCloud) == -1)
        {
            PCL_ERROR("Couldn't write file\n");
            return false;
        }
    }
    else
    {
        if (pcl::io::savePCDFile(fileName, *pointCloud) == -1)
        {
            PCL_ERROR("Couldn't write file\n");
            return false;
        }
    }
    std::cout << "Write to [" << fileName << "] done\n";
    return true;
}

void check_normal(pcTPtr &normals)
{
    for (int i = 0; i < normals->points.size(); i++)
    {
        if (!pcl::isFinite<Point_T>(normals->points[i]))
        {
            normals->points[i].normal_x = 0.577; // 1/ sqrt(3)
            normals->points[i].normal_y = 0.577;
            normals->points[i].normal_z = 0.577;
        }
    }
}

bool get_pc_semantic_normal(pcTPtr &cloud,
                            int K)
{
    return true;
}

void get_bbox(pcl::PointCloud<pcl::PointXYZ> object, int id) {

    pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(object.makeShared());
    feature_extractor.compute();
    
    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D (object, minPt, maxPt);

    double size_height = maxPt.z - minPt.z;

    std::vector <float> moment_of_inertia;
    std::vector <float> eccentricity;
    pcl::PointXYZ min_point_AABB;
    pcl::PointXYZ max_point_AABB;
    pcl::PointXYZ min_point_OBB;
    pcl::PointXYZ max_point_OBB;
    pcl::PointXYZ position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    float major_value, middle_value, minor_value;
    Eigen::Vector3f major_vector, middle_vector, minor_vector;
    Eigen::Vector3f mass_center;
 
    feature_extractor.getMomentOfInertia(moment_of_inertia);
    feature_extractor.getEccentricity(eccentricity);
    feature_extractor.getAABB(min_point_AABB, max_point_AABB);
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    feature_extractor.getEigenValues(major_value, middle_value, minor_value);
    feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
    feature_extractor.getMassCenter(mass_center);

    Eigen::Vector3f center(position_OBB.x, position_OBB.y, position_OBB.z);

    Eigen::Quaternionf quat(rotational_matrix_OBB);
    Eigen::Vector3f eulerAngle = quat.toRotationMatrix().eulerAngles(2,1,0);
    float yaw = eulerAngle(0);

    Eigen::Vector3f box_dim;
	box_dim = max_point_OBB.getVector3fMap() - min_point_OBB.getVector3fMap();

    //output bbox
    Eigen::Vector3f p1, p2, p3, p4, p5, p6, p7, p8;
    p1 << center(0) + 0.5 * cos(yaw) * box_dim(0) - 0.5 * sin(yaw) * box_dim(1),
                center(1) + 0.5 * sin(yaw) * box_dim(0) + 0.5 * cos(yaw) * box_dim(1),
                minPt.z;
    p2 <<  center(0) + 0.5 * cos(yaw) * box_dim(0) + 0.5 * sin(yaw) * box_dim(1),
                 center(1) + 0.5 * sin(yaw) * box_dim(0) - 0.5 * cos(yaw) * box_dim(1),
                 minPt.z;
    p3 << center(0) - 0.5 * cos(yaw) * box_dim(0) + 0.5 * sin(yaw) * box_dim(1),
                 center(1) - 0.5 * sin(yaw) * box_dim(0) - 0.5 * cos(yaw) * box_dim(1),
                minPt.z;
    p4 << center(0) - 0.5 * cos(yaw) * box_dim(0) - 0.5 * sin(yaw) * box_dim(1),
                center(1) - 0.5 * sin(yaw) * box_dim(0) + 0.5 * cos(yaw) * box_dim(1),
                minPt.z;
    p5 << center(0) + 0.5 * cos(yaw) * box_dim(0) - 0.5 * sin(yaw) * box_dim(1),
                center(1) + 0.5 * sin(yaw) * box_dim(0) + 0.5 * cos(yaw) * box_dim(1),
                minPt.z + size_height;
    p6 <<  center(0) + 0.5 * cos(yaw) * box_dim(0) + 0.5 * sin(yaw) * box_dim(1),
                 center(1) + 0.5 * sin(yaw) * box_dim(0) - 0.5 * cos(yaw) * box_dim(1),
                 minPt.z + size_height;
    p7 << center(0) - 0.5 * cos(yaw) * box_dim(0) + 0.5 * sin(yaw) * box_dim(1),
                 center(1) - 0.5 * sin(yaw) * box_dim(0) - 0.5 * cos(yaw) * box_dim(1),
                 minPt.z + size_height;
    p8 << center(0) - 0.5 * cos(yaw) * box_dim(0) - 0.5 * sin(yaw) * box_dim(1),
                center(1) - 0.5 * sin(yaw) * box_dim(0) + 0.5 * cos(yaw) * box_dim(1),
                minPt.z + size_height;

    std::string box_path = outputDirName + std::to_string(id) + ".txt";
    std::ofstream out(box_path, std::ostream::app);
    out << p1(0) << "," << p1(1) << "," << p1(2) << std::endl;
    out << p2(0) << "," << p2(1) << "," << p2(2) << std::endl;
    out << p3(0) << "," << p3(1) << "," << p3(2) << std::endl;
    out << p4(0) << "," << p4(1) << "," << p4(2) << std::endl;
    out << p5(0) << "," << p5(1) << "," << p5(2) << std::endl;
    out << p6(0) << "," << p6(1) << "," << p6(2) << std::endl;
    out << p7(0) << "," << p7(1) << "," << p7(2) << std::endl;
    out << p8(0) << "," << p8(1) << "," << p8(2) << std::endl;

}

bool segmentation(pcPtr &cloud,   std::vector<pcl::PointIndices> &cluster_indices) {
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud); 
   
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
 
    ec.setClusterTolerance(0.3); 
    ec.setMinClusterSize(50);
    ec.setMaxClusterSize(25000); 
    ec.setSearchMethod(tree); 
    ec.setInputCloud(cloud); 
    ec.extract(cluster_indices); 
}

int main(int argc, char **argv)
{
    if (argv[1] == NULL)
    {
        fprintf(stderr, "argv[1] must be path to dataset sequence -> /path/to/sequence/00/\n");
        exit(1);
    }
    if (argv[2] == NULL)
    {
        fprintf(stderr, "argv[2] must be output directory -> /path/to/output/directory/\n");
        exit(1);
    }

    std::string datasetDirName = argv[1];
    outputDirName = argv[2];
    skd::SKDAPI skdapi(datasetDirName);

    int timeIdx = 0;

    pcTPtr frame_pc(new pcT());
    std::vector<skd::VelodynePoint> velodynePoints;
    std::string frame_filename;

    for (;;)
    {
        velodynePoints = skdapi.getVelodynePoints(timeIdx);
        if ((int)velodynePoints.size() == 0)
            break;

        pcl::PointCloud<pcl::PointXYZ>::Ptr show_pc(new pcl::PointCloud<pcl::PointXYZ>); 
        for (int i = 0; i < (int)velodynePoints.size(); i++)
        {
            Point_T pt;
            pt.x = velodynePoints[i].x_;
            pt.y = velodynePoints[i].y_;
            pt.z = velodynePoints[i].z_;
            pt.intensity = velodynePoints[i].intensity_ * 255;
            frame_pc->points.push_back(pt);

            pcl::PointXYZ tmp;
            tmp.x = pt.x;
            tmp.y = pt.y;
            tmp.z = pt.z;
            show_pc->points.push_back(tmp);
        }

        for (int i = 0; i < frame_pc->points.size(); i++)
            frame_pc->points[i].curvature = velodynePoints[i].instance_; 

        pcTPtr vehicle_pc(new pcT());
        for (size_t i = 0; i < frame_pc->points.size(); i++) {
            int label = velodynePoints[i].label_;
            if (label == 10) {
                vehicle_pc->points.push_back(frame_pc->points[i]);
            }
        }
       
        std::unordered_map<int, int> instance_map;
        int index = 0;

        for (size_t i = 0; i < vehicle_pc->points.size(); i++) {
            int instance = vehicle_pc->points[i].curvature;
            auto it = instance_map.find(instance);
            if(it == instance_map.end()) {
                instance_map.emplace(instance, index);
                index++;
            }
        }

        std::vector<pcl::PointCloud<pcl::PointXYZ>> clusters_pc;
        for (auto it:instance_map) {
            int target_instance = it.first;
            pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_pc(new pcl::PointCloud<pcl::PointXYZ>);
            for(size_t i = 0; i < vehicle_pc->points.size(); i++) {
                 int instance = vehicle_pc->points[i].curvature;
                 if (instance == target_instance) {
                      pcl::PointXYZ pt;
                      pt.x = vehicle_pc->points[i].x;
                      pt.y = vehicle_pc->points[i].y;
                      pt.z = vehicle_pc->points[i].z;
                     tmp_pc->points.push_back(pt);
                 }
            }

            if(tmp_pc->points.size() != 0) {
                clusters_pc.push_back(*tmp_pc);
            }
        }
        
        for (size_t i = 0; i < clusters_pc.size(); i++) {
            auto tmp_pc = clusters_pc[i];
            get_bbox(tmp_pc, timeIdx);
        }

        std::ostringstream oss;
        oss.setf(std::ios::right);
        oss.fill('0');
        oss.width(6);
        oss << timeIdx;

        frame_filename = outputDirName + oss.str() + ".pcd";
        std::string vehicle_path  = outputDirName + oss.str() + "_vehicle" + ".pcd";
        pcl::io::savePCDFileBinary(vehicle_path, *vehicle_pc);
        write_pcd_file(frame_filename, frame_pc);

        std::vector<skd::VelodynePoint>().swap(velodynePoints);
        pcT().swap(*frame_pc);

        timeIdx++;
    }
    return 1;
}
