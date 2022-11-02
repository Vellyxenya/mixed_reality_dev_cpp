#include "open3d/Open3D.h"
#include "processing.h"
#include <iostream>

using namespace open3d;

Eigen::Matrix4d get_transformation(const std::vector<Eigen::Vector3d>& source_vec,
    const std::vector<Eigen::Vector3d>& target_vec) {

    geometry::PointCloud source = geometry::PointCloud(source_vec);
    geometry::PointCloud target = geometry::PointCloud(target_vec);

    int nb_iterations = 30;

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; ++i) {
        auto result = pipelines::registration::RegistrationGeneralizedICP(
            source, target, 0.07, T,
            pipelines::registration::TransformationEstimationForGeneralizedICP(),
            pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, nb_iterations));
        T = result.transformation_;
    }
    return T;
}