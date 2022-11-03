#include "open3d/Open3D.h"
#include "processing.h"
#include "utils.h"
#include <iostream>

using namespace open3d;

Eigen::Matrix4d get_transformation_(const geometry::PointCloud& source,
    const geometry::PointCloud& target, Eigen::Matrix6d& InfoMat) {

    int nb_iterations = 30;

    std::vector<double> voxel_sizes = {0.05, 0.05 / 2, 0.05 / 4};
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; ++i) {
        auto source_down = source.VoxelDownSample(voxel_sizes[i]);
        source_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(
                voxel_sizes[i] * 2.0, 30));

        auto target_down = target.VoxelDownSample(voxel_sizes[i]);
        target_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(
                voxel_sizes[i] * 2.0, 30));

        auto result = pipelines::registration::RegistrationGeneralizedICP(
            *source_down, *target_down, 0.06, T,
            pipelines::registration::TransformationEstimationForGeneralizedICP(),
            pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, nb_iterations));
        T = result.transformation_;
    }
    double max_correspondance_dist = 0.02;
    InfoMat = pipelines::registration::GetInformationMatrixFromPointClouds(source, target, max_correspondance_dist, T);
    return T;
}

Eigen::Matrix4d get_transformation(const PCD& source_pcd,
    const PCD& target_pcd, Eigen::Matrix6d& InfoMat) {

    const geometry::PointCloud source = geometry::PointCloud(source_pcd);
    const geometry::PointCloud target = geometry::PointCloud(target_pcd);

    return get_transformation_(source, target, InfoMat);
}

Eigen::MatrixXd merge_point_clouds(std::vector<PCD>& pcds_vec) {
    pipelines::registration::PoseGraph pose_graph;
    Eigen::Matrix4d odometry = Eigen::Matrix4d::Identity();
    pose_graph.nodes_.push_back(pipelines::registration::PoseGraphNode(odometry));

    std::vector<geometry::PointCloud> pcds;
    for(const auto pcd : pcds_vec) {
        pcds.push_back(geometry::PointCloud(pcd));
    }

    size_t memory = 10;
    for(int src_id = 0; src_id < pcds.size(); src_id++) {
        std::cout << "Computing transformations for point cloud " << src_id << std::endl;
        for(int tgt_id = src_id + 1; tgt_id < std::min(src_id + memory, pcds.size()); tgt_id++) {
            Eigen::Matrix6d InfoMat;
            Eigen::Matrix4d T = get_transformation_(pcds[src_id], pcds[tgt_id], InfoMat);
            if(tgt_id == src_id + 1) { //Adjacent frame
                odometry = T * odometry;
                Eigen::Matrix4d inv_odo = odometry.inverse();
                pose_graph.nodes_.push_back(
                    pipelines::registration::PoseGraphNode(inv_odo));
                pose_graph.edges_.push_back( //'certain'
                    pipelines::registration::PoseGraphEdge(src_id, tgt_id, T, InfoMat, false));
            } else {
                pose_graph.edges_.push_back( //'uncertain'
                    pipelines::registration::PoseGraphEdge(src_id, tgt_id, T, InfoMat, true));
            }
        }
    }

    pipelines::registration::GlobalOptimizationLevenbergMarquardt optimization_method;
    pipelines::registration::GlobalOptimizationConvergenceCriteria criteria;
    pipelines::registration::GlobalOptimizationOption option(0.075, 0.25, 1, 0);
    std::cout << "Running Global optimization..." << std::endl;
    pipelines::registration::GlobalOptimization(pose_graph, optimization_method, criteria, option);
    auto pose_graph_prunned = pipelines::registration::CreatePoseGraphWithoutInvalidEdges(pose_graph, option);
    geometry::PointCloud pcd_combined;
    std::cout << "Merging Point Clouds..." << std::endl;
    for(int i = 0; i < pcds.size(); i++) {
        std::cout << '.' << std::flush;
        pcd_combined += pcds[i].Transform(pose_graph.nodes_[i].pose_);
    }
    std::cout << "\nFinished merging" << std::endl;
    //std::shared_ptr<geometry::PointCloud> pcd_combined_down = pcd_combined.VoxelDownSample(0.05);
    //return vec_to_eigen(pcd_combined_down->points_);
    return vec_to_eigen(pcd_combined.points_);
}