#ifndef PROCESSING_H
#define PROCESSING_H

Eigen::Matrix4d get_transformation(const std::vector<Eigen::Vector3d>& source_vec,
    const std::vector<Eigen::Vector3d>& target_vec);

#endif