#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/flare.h>
#include <pcl/point_cloud.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/common.h>
#include <cmath>
#include <filesystem>
#include <string>

using KdTreePtr = pcl::search::KdTree<pcl::PointXYZ>::Ptr;
using PointCloudPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

PointCloudPtr cloud;
KdTreePtr tree;

//sampled surface for the computation of tangent X axis
PointCloudPtr sampled_cloud;
KdTreePtr sampled_tree;

int main(int argc, char **argv)
{

  std::ifstream file("../../../data/all_pcd_files.txt");
  std::string fileName;
  while (std::getline(file, fileName))
  {

    cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(fileName, *cloud) < 0)
    {
      std::cerr << "Failed to read test file. Please download `bun0.pcd` and pass its path to the test." << std::endl;
      return (-1);
    }

    tree.reset(new pcl::search::KdTree<pcl::PointXYZ>(false));
    tree->setInputCloud(cloud);

    const float sampling_perc = 1.0f;
    const float sampling_incr = 1.0f / sampling_perc;
    sampled_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

    std::vector<int> sampled_indices;
    for (float sa = 0.0f; sa < (float)cloud->size(); sa += sampling_incr)
      sampled_indices.push_back(static_cast<int>(sa));
    copyPointCloud(*cloud, sampled_indices, *sampled_cloud);

    sampled_tree.reset(new pcl::search::KdTree<pcl::PointXYZ>(false));
    sampled_tree->setInputCloud(sampled_cloud);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::PointCloud<pcl::ReferenceFrame> mesh_LRF;

    const float mesh_res = 0.03;

    // Compute normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;

    ne.setRadiusSearch(5.0f * mesh_res);
    ne.setViewPoint(0,0,0);
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);

    ne.compute(*normals);

    // Compute FLARE LRF
    pcl::FLARELocalReferenceFrameEstimation<pcl::PointXYZ, pcl::Normal, pcl::ReferenceFrame> lrf_estimator;

    lrf_estimator.setRadiusSearch(2 * mesh_res);
    lrf_estimator.setTangentRadius(4 * mesh_res);

    lrf_estimator.setInputCloud(cloud);
    lrf_estimator.setSearchSurface(cloud);
    lrf_estimator.setInputNormals(normals);
    lrf_estimator.setSearchMethod(tree);
    lrf_estimator.setSearchMethodForSampledSurface(sampled_tree);
    lrf_estimator.setSearchSampledSurface(sampled_cloud);

    lrf_estimator.compute(mesh_LRF);

    std::ofstream lrf_file;
    // Write LRF for each point
    fileName = fileName.substr(0, fileName.find(".pcd")) + "_lrf.txt";
    std::cout << fileName << std::endl;
    lrf_file.open(fileName);
    lrf_file << "orig_x orig_y orig_z x_0 x_1 x_2 y_0 y_1 y_2 z_0 z_1 z_2\n";
    for (int i = 0; i < cloud->size(); i++)
    {
      lrf_file << cloud->at(i).x << " " << cloud->at(i).y << " " << cloud->at(i).z << " ";
      lrf_file << mesh_LRF.at(i).x_axis[0] << " " << mesh_LRF.at(i).x_axis[1] << " " << mesh_LRF.at(i).x_axis[2] << " ";
      lrf_file << mesh_LRF.at(i).y_axis[0] << " " << mesh_LRF.at(i).y_axis[1] << " " << mesh_LRF.at(i).y_axis[2] << " ";
      lrf_file << mesh_LRF.at(i).z_axis[0] << " " << mesh_LRF.at(i).z_axis[1] << " " << mesh_LRF.at(i).z_axis[2] << "\n";
    }
    lrf_file.close();
  }
  return 0;
}