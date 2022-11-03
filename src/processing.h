#ifndef PROCESSING_H
#define PROCESSING_H

typedef std::vector<Eigen::Vector3d> PCD;

Eigen::Matrix4d get_transformation(const PCD& source_pcd,
    const PCD& target_pcd, Eigen::Matrix<double, 6, 6>& InfoMat);

Eigen::MatrixXd merge_point_clouds(std::vector<PCD>& pcds_vec, std::vector<Eigen::MatrixXd>& list_cumulative_pcds);


#endif