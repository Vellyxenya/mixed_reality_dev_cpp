#include "open3d/Open3D.h"
#include "processing.h"
#include "utils.h"
#include <iostream>

using namespace open3d;

double max_correspondance_dist = 0.01;

Eigen::Matrix4d get_transformation_(const geometry::PointCloud& source,
    const geometry::PointCloud& target, Eigen::Matrix6d& InfoMat) {

    int nb_iterations = 300;

    double voxel_size = 0.02;
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 4; ++i) {
        auto source_down = source.VoxelDownSample(voxel_size);
        source_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(
                voxel_size * 2.0, 30));

        auto target_down = target.VoxelDownSample(voxel_size);
        target_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(
                voxel_size * 2.0, 30));

        auto loss = pipelines::registration::TukeyLoss(0.1);
        auto kernel = loss.k_;
        auto result = pipelines::registration::RegistrationGeneralizedICP(
            *source_down, *target_down, max_correspondance_dist, T,
            pipelines::registration::TransformationEstimationForGeneralizedICP(kernel),
            pipelines::registration::ICPConvergenceCriteria(1e-7, 1e-7, nb_iterations));
        T = result.transformation_;
        voxel_size /= 2;
    }
    InfoMat = pipelines::registration::GetInformationMatrixFromPointClouds(source, target, max_correspondance_dist, T);
    return T;
}


// Eigen::Matrix4d get_transformation_with_initial_T(const geometry::PointCloud& source,
//     const geometry::PointCloud& target, Eigen::Matrix6d& InfoMat, const Eigen::Matrix4d* initialT) {

//     int nb_iterations = 50;

//     std::vector<double> voxel_sizes = {0.02, 0.01, 0.01 / 2, 0.01 / 4};
//     double voxel_size = 0.02;
//     Eigen::Matrix4d T = (initialT == nullptr) ? Eigen::Matrix4d::Identity() : *initialT;
//     for (int i = 0; i < 4; ++i) {
//         auto source_down = source.VoxelDownSample(voxel_size);
//         source_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(
//                 voxel_size * 2.0, 30));

//         auto target_down = target.VoxelDownSample(voxel_size);
//         target_down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(
//                 voxel_size * 2.0, 30));

//         auto loss = pipelines::registration::TukeyLoss(0.5);
//         auto kernel = loss.k_;
//         auto result = pipelines::registration::RegistrationGeneralizedICP(
//             *source_down, *target_down, 0.02, T,
//             pipelines::registration::TransformationEstimationForGeneralizedICP(kernel),
//             pipelines::registration::ICPConvergenceCriteria(1e-7, 1e-6, nb_iterations));
//         T = result.transformation_;
//         voxel_size /= 2;
//     }
//     InfoMat = pipelines::registration::GetInformationMatrixFromPointClouds(source, target, max_correspondance_dist, T);
//     return T;
// }

Eigen::Matrix4d get_transformation(const PCD& source_pcd,
    const PCD& target_pcd, Eigen::Matrix6d& InfoMat) {

    const geometry::PointCloud source = geometry::PointCloud(source_pcd);
    const geometry::PointCloud target = geometry::PointCloud(target_pcd);

    return get_transformation_(source, target, InfoMat);
}

Eigen::MatrixXd merge_point_clouds(std::vector<PCD>& pcds_vec, 
    std::vector<PCD>& colors_vec, 
    std::vector<Eigen::MatrixXd>& list_cumulative_pcds,
    std::vector<Eigen::MatrixXd>& partial_pcds) {

    pipelines::registration::PoseGraph pose_graph;
    Eigen::Matrix4d odometry = Eigen::Matrix4d::Identity();
    geometry::PointCloud pcd_combined;
    std::vector<Eigen::Matrix4d> odometries;

    pose_graph.nodes_.push_back(pipelines::registration::PoseGraphNode(odometry));
    odometries.push_back(odometry);

    std::vector<geometry::PointCloud> pcds;
    for(int i = 0; i < pcds_vec.size(); i++) { //const auto pcd_ : pcds_vec) {
        auto pcd = geometry::PointCloud(pcds_vec[i]);
        pcd.colors_ = colors_vec[i];
        pcds.push_back(pcd);
    }

    bool use_global_optimization = false;
    if(use_global_optimization) {
        size_t memory = 50;
        for(int src_id = 0; src_id < pcds.size(); src_id++) {
            std::cout << "Computing transformations for point cloud " << src_id << std::endl;
            //for(int tgt_id = src_id + 1; tgt_id < std::min(src_id + memory, pcds.size()); tgt_id++) {
            int offset = 1;
            for(int tgt_id = src_id + offset; tgt_id + offset < pcds.size() - 1; offset*=2) {
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
            //Want to make sure that there is at least one frame connected to them all so that we get a connected pose graph
            if(src_id != pcds.size() - 1) {
                Eigen::Matrix6d InfoMat;
                Eigen::Matrix4d T = get_transformation_(pcds[src_id], pcds[pcds.size()-1], InfoMat);
                pose_graph.edges_.push_back( //'uncertain'
                    pipelines::registration::PoseGraphEdge(src_id, pcds.size()-1, T, InfoMat, true));
                std::cout << odometry << std::endl;
            }
        }
        pipelines::registration::GlobalOptimizationLevenbergMarquardt optimization_method;
        pipelines::registration::GlobalOptimizationConvergenceCriteria criteria; //TODO probably need to tweak
        //TODO probably need to tweak
        double edge_prune_threshold = 0.025;
        double preference_loop_closure = 2;
        int reference_node = 0;
        auto option = pipelines::registration::GlobalOptimizationOption(
            max_correspondance_dist, edge_prune_threshold, preference_loop_closure, reference_node);

        std::cout << "Running Global optimization..." << std::endl;
        pipelines::registration::GlobalOptimization(pose_graph, optimization_method, criteria, option);
        //auto pose_graph_prunned = pipelines::registration::CreatePoseGraphWithoutInvalidEdges(pose_graph, option);

        std::cout << "Merging Point Clouds..." << std::endl;
        for(int i = 0; i < pcds.size(); i++) {
            std::cout << '.' << std::flush;
            pcd_combined += pcds[i].Transform(pose_graph.nodes_[i].pose_);
            list_cumulative_pcds.push_back(vec_to_eigen(pcd_combined.VoxelDownSample(0.01)->points_));
        }
        std::cout << "\nFinished merging" << std::endl;
        std::shared_ptr<geometry::PointCloud> pcd_combined_down = pcd_combined.VoxelDownSample(0.01);
        Eigen::MatrixXd MergedPoints = vec_to_eigen(pcd_combined_down->points_);
        return MergedPoints;
    } else {
        //std::vector<Eigen::MatrixXd> partial_pcds;
        int nb_children_pcds = 1;
        for(int src_id = 0; src_id < pcds.size()-1; src_id++) {
            //std::cout << "Transforming and integrating Point Cloud " << src_id << std::endl;
            Eigen::Matrix6d InfoMat;
            Eigen::Matrix4d T = get_transformation_(pcds[src_id], pcds[src_id+1], InfoMat);
            Eigen::Matrix4d new_odometry = T * odometry;
            double norm = (new_odometry - odometry).squaredNorm();
            odometry = new_odometry;
            //std::cout << "norm: " << norm << std::endl;

            if(norm > 0.003) {
                pcd_combined = pcds[src_id];
                std::cout << nb_children_pcds << std::endl;
                if(nb_children_pcds >= 10) {
                    partial_pcds.push_back(vec_to_eigen(pcd_combined.points_));
                }
                nb_children_pcds = 1;
                odometry = Eigen::Matrix4d::Identity();
                continue;
            }

            Eigen::Matrix4d inv_odo = odometry.inverse();
            pcd_combined += pcds[src_id].Transform(inv_odo);
            pcd_combined = *pcd_combined.VoxelDownSample(0.01);
            //std::cout << pcd_combined.points_.size() << std::endl;
            list_cumulative_pcds.push_back(vec_to_eigen(pcd_combined.points_));
            nb_children_pcds++;
        }
        if(nb_children_pcds >= 10) {
            partial_pcds.push_back(vec_to_eigen(pcd_combined.points_));
        }
        return list_cumulative_pcds[list_cumulative_pcds.size()-1];
        /*Eigen::Matrix4d prev_to_base = Eigen::Matrix4d::Identity();
        pcd_combined = pcds[0];
        for(int src_id = 1; src_id < pcds.size(); src_id++) {
            std::cout << "Transforming and integrating Point Cloud " << src_id << std::endl;
            Eigen::Matrix6d InfoMat;
            Eigen::Matrix4d T_current_to_prev = get_transformation_(pcds[src_id], pcds[src_id-1], InfoMat);

            Eigen::Matrix4d T_current_to_base = prev_to_base * T_current_to_prev;
            auto pc_in_base = pcds[src_id].Transform(T_current_to_base);
            Eigen::Matrix4d T_fine_tune = get_transformation_(pc_in_base, pcd_combined, InfoMat);

            prev_to_base = T_fine_tune * T_current_to_base;

            auto pc_transformed = pcds[src_id].Transform(prev_to_base);
            auto registration_res = pipelines::registration::EvaluateRegistration(pcds[src_id], pcd_combined, max_correspondance_dist, prev_to_base);
            std::cout << registration_res.inlier_rmse_ << " " << registration_res.fitness_ << std::endl;
            if(registration_res.fitness_ < 0.9) {
                continue;
            } else {
                std::cout << "Adding new point cloud" << std::endl;
            }
            pcd_combined += pc_transformed;
            pcd_combined = *pcd_combined.VoxelDownSample(0.01);
            std::cout << pcd_combined.points_.size() << std::endl;
            list_cumulative_pcds.push_back(vec_to_eigen(pcd_combined.points_));
        }
        return list_cumulative_pcds[list_cumulative_pcds.size()-1];*/
    }
}