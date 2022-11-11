#include "open3d/Open3D.h"
#include "processing.h"
#include "utils.h"
#include <iostream>

using namespace open3d;

using std::cout;
using std::endl;

double max_correspondance_dist = 0.01;
int minimum_children = 10;

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

void run_global_optimization(std::vector<geometry::PointCloud>& partial_pcds, Eigen::MatrixXd& ThePointCloud) {
    //Setup the pose graph
    pipelines::registration::PoseGraph pose_graph;
    for(int src_id = 0; src_id < partial_pcds.size(); src_id++) {
        std::cout << "Computing transformations for point cloud " << src_id << std::endl;
        pose_graph.nodes_.push_back(
            pipelines::registration::PoseGraphNode(Eigen::Matrix4d::Identity()));
        for(int tgt_id = src_id + 1; tgt_id < partial_pcds.size(); tgt_id++) {
            Eigen::Matrix6d InfoMat;
            Eigen::Matrix4d T = get_transformation_(partial_pcds[src_id], partial_pcds[tgt_id], InfoMat);
            bool uncertain = true;
            pose_graph.edges_.push_back(
                pipelines::registration::PoseGraphEdge(src_id, tgt_id, T, InfoMat, uncertain, 0.7));
        }
    }
    //Setup optimization parameters
    pipelines::registration::GlobalOptimizationLevenbergMarquardt optimization_method;
    pipelines::registration::GlobalOptimizationConvergenceCriteria criteria; //TODO probably need to tweak
    double edge_prune_threshold = 0.025;
    double preference_loop_closure = 6;
    int reference_node = 0;
    auto option = pipelines::registration::GlobalOptimizationOption(
        max_correspondance_dist, edge_prune_threshold, preference_loop_closure, reference_node);

    //Run global optimization
    std::cout << "Running Global optimization..." << std::endl;
    pipelines::registration::GlobalOptimization(pose_graph, optimization_method, criteria, option);
    //auto pose_graph_prunned = pipelines::registration::CreatePoseGraphWithoutInvalidEdges(pose_graph, option);

    std::cout << "Merging Into global point cloud..." << std::endl;
    geometry::PointCloud pcd_combined;
    for(int i = 0; i < partial_pcds.size(); i++) {
        std::cout << '.' << std::flush;
        Eigen::Matrix4d Transformation = pose_graph.nodes_[i].pose_;
        cout << "Transformation:\n" << Transformation << endl;
        pcd_combined += partial_pcds[i].Transform(Transformation);
    }
    cout << "Combined " << pcd_combined.points_.size() << endl;
    std::shared_ptr<geometry::PointCloud> pcd_combined_down = pcd_combined.VoxelDownSample(0.005);
    ThePointCloud = vec_to_eigen(pcd_combined_down->points_);
}

Eigen::MatrixXd merge_point_clouds(std::vector<PCD>& pcds_vec, 
    std::vector<PCD>& colors_vec, 
    std::vector<Eigen::MatrixXd>& list_cumulative_pcds,
    std::vector<Eigen::MatrixXd>& partial_pcds) {
    Eigen::Matrix4d odometry = Eigen::Matrix4d::Identity();
    geometry::PointCloud pcd_combined;
    std::vector<Eigen::Matrix4d> odometries;
    
    odometries.push_back(odometry);

    std::vector<geometry::PointCloud> pcds;
    for(int i = 0; i < pcds_vec.size(); i++) {
        auto pcd = geometry::PointCloud(pcds_vec[i]);
        pcd.colors_ = colors_vec[i];
        pcds.push_back(pcd);
    }

    std::vector<Eigen::Matrix4d> partial_odometries;
    std::vector<Eigen::Matrix4d> partial_Ts;
    int nb_children_pcds = 1;
    for(int src_id = 0; src_id < pcds.size()-1; src_id++) {
        Eigen::Matrix6d InfoMat;
        Eigen::Matrix4d T = get_transformation_(pcds[src_id], pcds[src_id+1], InfoMat);
        Eigen::Matrix4d new_odometry = T * odometry;
        double norm = (new_odometry - odometry).squaredNorm();
        odometry = new_odometry;
        //std::cout << "norm: " << norm << std::endl;

        if(norm > 0.003) {
            pcd_combined = pcds[src_id];
            std::cout << nb_children_pcds << std::endl;
            if(nb_children_pcds >= minimum_children) {
                partial_pcds.push_back(vec_to_eigen(pcd_combined.points_));
                partial_odometries.push_back(odometry);
            }
            nb_children_pcds = 1;
            continue;
        }

        Eigen::Matrix4d inv_odo = odometry.inverse();
        pcd_combined += pcds[src_id].Transform(inv_odo);
        pcd_combined = *pcd_combined.VoxelDownSample(0.01);
        list_cumulative_pcds.push_back(vec_to_eigen(pcd_combined.points_));
        nb_children_pcds++;
    }
    if(nb_children_pcds >= minimum_children) {
        partial_pcds.push_back(vec_to_eigen(pcd_combined.points_));
    }

    //TODO: Clean every point cloud
    cout << "Putting all clouds in the same frame..." << endl;
    std::vector<geometry::PointCloud> transformed_pcds;
    geometry::PointCloud final_pcd_combined;
    int nb_merges = partial_pcds.size();
    for(int i = 0; i < nb_merges; i++) {
        cout << '.' << std::flush;
        Eigen::Matrix4d Transformation = partial_odometries[i].inverse();
        auto transformed_pc = geometry::PointCloud(eigen_to_vec(partial_pcds[i])).Transform(Transformation);
        transformed_pcds.push_back(transformed_pc);
        final_pcd_combined += transformed_pc;
        cout << "final_pcd_combined.size(): " << final_pcd_combined.points_.size() << endl;
    }
    cout << "\nFinished merging" << endl;
    //Eigen::MatrixXd MergedPoints = vec_to_eigen(final_pcd_combined.points_);
    //partial_pcds.push_back(MergedPoints); //TODO put in separate datastructure

    Eigen::MatrixXd ThePointCloud;
    run_global_optimization(transformed_pcds, ThePointCloud);
    partial_pcds.push_back(ThePointCloud);

    return list_cumulative_pcds[list_cumulative_pcds.size()-1];
}