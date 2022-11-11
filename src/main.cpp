#include <igl/read_triangle_mesh.h>
#include <igl/readTGF.h>
#include <igl/readDMAT.h>
#include <igl/opengl/glfw/Viewer.h>

#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/slice.h>
#include <imgui/imgui.h>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <limits.h>
#include <math.h>
#include <map>

#include "utils.h"
#include "loading.h"
#include "processing.h"

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

#define NB_DEPTH_PIXELS (512 * 512)

//Timestamps datastructures
vector<long> timestamps;
vector<long> pv_timestamps;
vector<long> human_timestamps;

//Transformation matrices
vector<Eigen::MatrixXf> rig2world_matrices;
vector<Eigen::MatrixXf> pv2world_matrices;
Eigen::MatrixXf cam2rig;

//RGB Images
map<long, RGBImage> rgb_images;

//Depth Images
std::vector<DepthImage> depth_images;

//Joints data
vector<vector<MatrixXf>> list_joints_left;
vector<vector<MatrixXf>> list_joints_right;

//Intrinsics data
vector<std::pair<float, float>> focals;
float intrinsics_ox, intrinsics_oy;
int intrinsics_width, intrinsics_height;

//Lookup table
Eigen::MatrixXf lut;

std::vector<string> paths;
Eigen::MatrixXi E; //Edges between joints

Eigen::VectorXf ones;
Viewer viewer;
string folder;

int l = 0; //frame counter
int nb_frames;
bool is_first_frame = true;
float min_dist_to_joints = 0.007;

//Params
bool visualize_partials = true;

//Temp data
vector<Eigen::Vector3d> prev_points_vec;
vector<Eigen::Matrix4d> transformations;

//All point clouds, and corresponding colors
vector<PCD> point_clouds, color_clouds;

Eigen::MatrixXf to_homogeneous(const Eigen::MatrixXf& points);
bool callback_pre_draw(Viewer& viewer);
bool valid_color(const RowVector3d& color);
void apply_transformation(const Eigen::MatrixXf& T, const Eigen::MatrixXf& points, Eigen::MatrixXf& out, int out_dim);
int closest_timestamp_idx(long ts, vector<long> timestamps);
void compute_joints_positions(const Eigen::MatrixXf& world2rig, const vector<MatrixXf>& joints, MatrixXd& djoints);
void compute_joints_positions(const Eigen::MatrixXf& rig2world_matrix,
  const vector<MatrixXf>& joints_left, const vector<MatrixXf>& joints_right,
  MatrixXd& djoints_left, MatrixXd& djoints_right);
void depth_map_to_pc_px(const DepthImage& depth_map, const Eigen::MatrixXf& cam_calibration, 
  float clamp_min, float clamp_max, Eigen::MatrixXf& pointsCam);
void project_on_pv(const Eigen::MatrixXf& pointsWorld, const Eigen::MatrixXf& pv2world, 
  float focalx, float focaly, float principalx, float principaly, int pvWidth, int pvHeight,
  const vector<vector<RowVector3d>>& rgbImage, Eigen::MatrixXd& colors, Eigen::MatrixXi& rgb_indices);

bool point_too_close_to_hands(
  const Eigen::MatrixXd& joints_left,
  const Eigen::MatrixXd& joints_right,
  const Eigen::RowVector3f& point) {
    ///////////////
    return false;
    ///////////////
  Eigen::RowVector3d dpoint = point.cast<double>();
  for(int i = 0; i < joints_left.rows(); i++) {
    if((joints_left.row(i) - dpoint).squaredNorm() < min_dist_to_joints) {
      return true;
    }
  }
  for(int i = 0; i < joints_right.rows(); i++) {
    if((joints_right.row(i) - dpoint).squaredNorm() < min_dist_to_joints) {
      return true;
    }
  }
  return false;
}

void discard_points_too_close_to_the_hands(
  const Eigen::MatrixXd& joints_left,
  const Eigen::MatrixXd& joints_right,
  const Eigen::MatrixXf& pointsRig_all,
  Eigen::MatrixXf& pointsRig) {
  pointsRig.resize(pointsRig_all.rows(), 3);
  int k = 0;
  for(int i = 0; i < pointsRig_all.rows(); i++) {
    if(!point_too_close_to_hands(joints_left, joints_right, pointsRig_all.row(i)))
      pointsRig.row(k++) = pointsRig_all.row(i);
  }
  pointsRig = pointsRig.block(0, 0, k, 3);
}

/**
 * @brief Main Processing function. Takes in all necessary data-structures and outputs
 * the processed and colored point cloud
 * 
 * @param depthImage 
 * @param rgbImage 
 * @param rig2world 
 * @param pv2world 
 * @param cam2rig 
 * @param joints_left 
 * @param joints_right 
 * @param lut 
 * @param focals 
 * @param intrinsics_ox 
 * @param intrinsics_oy 
 * @param intrinsics_width 
 * @param intrinsics_height 
 * @param djointsleft Output joint positions (left)
 * @param djointsright Output joint positions (right)
 * @param ddata_only_rgb Output points
 * @param dcolors_only_rgb colors corrseponding to the output points
 */
void process(
  const DepthImage& depthImage,
  const RGBImage& rgbImage,
  const Eigen::MatrixXf& rig2world,
  const Eigen::MatrixXf& pv2world,
  const Eigen::MatrixXf& cam2rig,
  const std::vector<Eigen::MatrixXf>& joints_left,
  const std::vector<Eigen::MatrixXf>& joints_right,
  const Eigen::MatrixXf& lut,
  const std::pair<float, float>& focals,
  const float intrinsics_ox, const float intrinsics_oy, 
  const int intrinsics_width, const int intrinsics_height,
  Eigen::MatrixXd& djointsleft,
  Eigen::MatrixXd& djointsright,
  Eigen::MatrixXd& ddata_only_rgb,
  Eigen::MatrixXd& dcolors_only_rgb) {

  compute_joints_positions(rig2world, 
    joints_left, joints_right, djointsleft, djointsright);

  Eigen::MatrixXf pointsCam;
  depth_map_to_pc_px(depthImage, lut, 0.1, 0.7, pointsCam);

  //Transform points from cam coordinatse to rig coordinates
  Eigen::MatrixXf pointsRig_all;
  apply_transformation(cam2rig, pointsCam, pointsRig_all, 3);

  Eigen::MatrixXf pointsRig;
  discard_points_too_close_to_the_hands(djointsleft, djointsright, pointsRig_all, pointsRig);

  //Transform points from rig coordinates to world
  Eigen::MatrixXf pointsWorld;
  apply_transformation(rig2world, pointsRig, pointsWorld, 3);

  //Project points from world coordinates to RGB
  Eigen::MatrixXd colors;
  Eigen::MatrixXi rgb_indices;
  project_on_pv(pointsWorld, pv2world, 
    focals.first, focals.second, intrinsics_ox, intrinsics_oy, 
    intrinsics_width, intrinsics_height, rgbImage, colors, rgb_indices);

  Eigen::MatrixXd V = pointsRig.cast<double>(); //Cast to double so that libigl is happy
  igl::slice(V, rgb_indices, 1, ddata_only_rgb);
  igl::slice(colors, rgb_indices, 1, dcolors_only_rgb);
}

int visualized_partial = 0;
bool gathering_data = true;
std::vector<Eigen::MatrixXd> list_cumulative_pcds;
std::vector<Eigen::MatrixXd> partial_pcds;
//Compute what to display on the next frame
bool callback_pre_draw(Viewer& viewer) {
  if(gathering_data) {
    if (l == nb_frames)
      return false;
    //Compute correct timestamps
    long depth_timestamp = timestamps[l];

    int pv_timestamp_idx = closest_timestamp_idx(depth_timestamp, pv_timestamps);
    long pv_timestamp = pv_timestamps[pv_timestamp_idx];

    int human_timestamp_idx = closest_timestamp_idx(depth_timestamp, human_timestamps);
    long human_timestamp = human_timestamps[human_timestamp_idx];

    //TODO the rig2world_matrices[l] does not necessarily have the right timestamp..
    Eigen::MatrixXd JointsLeft;
    Eigen::MatrixXd JointsRight;
    Eigen::MatrixXd Points;
    Eigen::MatrixXd Colors;
    process(
      depth_images[l],
      rgb_images[pv_timestamp],
      rig2world_matrices[l],
      pv2world_matrices[pv_timestamp_idx],
      cam2rig,
      list_joints_left[human_timestamp_idx],
      list_joints_right[human_timestamp_idx],
      lut,
      focals[l],
      intrinsics_ox, intrinsics_oy,
      intrinsics_width, intrinsics_height,
      JointsLeft,
      JointsRight,
      Points,
      Colors);

    viewer.data().clear();
    viewer.data().add_points(Points, Colors);
    //TODO calling set_edges twices overrides the first one
    viewer.data().set_edges(JointsLeft, E, Eigen::RowVector3d(0, 0, 1));
    viewer.data().set_edges(JointsRight, E, Eigen::RowVector3d(1, 0, 0));

    int minimum_points = 1000; //TODO probably increase this
    int nb_warmup_frames = 50;
    if(Points.rows() > minimum_points && l >= nb_warmup_frames) { //valid frame
      // cout << "min: " << Points.colwise().minCoeff() << endl;
      // cout << "max: " << Points.colwise().maxCoeff() << endl;
      PCD points_vec = eigen_to_vec(Points);
      PCD colors_vec = eigen_to_vec(Colors);
      point_clouds.push_back(points_vec);
      color_clouds.push_back(colors_vec);
      if(is_first_frame) {
        viewer.core().align_camera_center(Points);
        is_first_frame = false;
      } else {
        //TODO For now the processing is done offline but should be moved here at some point

        /*Eigen::Matrix<double, 6, 6> InfoMat;
        Eigen::Matrix4d T = get_transformation(points_vec, prev_points_vec, InfoMat);
        transformations.push_back(T);*/
      }
      prev_points_vec = points_vec;
    }
      
    ++l;
    cout << "Frame " << l << endl;

    //Visualize merged point cloud
    if(l == nb_frames) {
      Eigen::MatrixXd final_point_cloud = merge_point_clouds(point_clouds, color_clouds, list_cumulative_pcds, partial_pcds);
      viewer.data().clear();
      viewer.data().add_points(final_point_cloud, Eigen::RowVector3d(0, 0, 1));
      gathering_data = false;
      l = 0;
    }
  } else {
    if(visualize_partials) {
      if(partial_pcds.size() <= 0) {
        cout << "No partial pcds found" << endl;
        return false;
      }
      l++;
      if(l >= 90) {
        l = 0;
        visualized_partial++;
        if(visualized_partial >= partial_pcds.size())
          visualized_partial = 0;
        bool last_partial = visualized_partial == partial_pcds.size() - 1;
        viewer.data().clear();
        viewer.data().add_points(partial_pcds[partial_pcds.size()-1], last_partial ? Eigen::RowVector3d(0, 0.5, 0) : Eigen::RowVector3d(0.5, 0, 0));
      }
    } else {
      if(list_cumulative_pcds.size() <= 0) {
        cout << "Trying to visualize cumulative point cloud but no point cloud is available" << endl;
        return false;
      }
      viewer.data().clear();
      viewer.data().add_points(list_cumulative_pcds[l], Eigen::RowVector3d(0, 0, 0.5));
      l++;
      if(l >= list_cumulative_pcds.size())
        l = 0;
    }
  }
    
  return false;
}

void depth_map_to_pc_px(const DepthImage& depth_map, 
  const Eigen::MatrixXf& cam_calibration, float clamp_min, float clamp_max,
  Eigen::MatrixXf& pointsCam) {
  clamp_min *= 1000;
  clamp_max *= 1000;

  Eigen::MatrixXf image(NB_DEPTH_PIXELS, 3);
  vector<int> indices_to_remove;
  vector<bool> remove = vector<bool>(NB_DEPTH_PIXELS, false);
  int k = 0;
  int total_removed = 0;
  for(int j = 0; j < depth_map.size(); j++) {
    for(int i = 0; i < depth_map[0].size(); i++) {
      float d = depth_map[j][i];
      if(d < clamp_min || d > clamp_max) {
        d = 0;
        indices_to_remove.push_back(k);
        remove[k] = true;
        total_removed++;
      }
      float a0 = cam_calibration(k, 0);
      float a1 = cam_calibration(k, 1);
      float a2 = cam_calibration(k, 2);
      image.row(k) = Eigen::RowVector3f(d * a0, d * a1, d * a2) / 1000;
      k++;
    }
  }

  int remaining_points = NB_DEPTH_PIXELS - total_removed;
  pointsCam.resize(remaining_points, 3);
  k = 0;
  for(int i = 0; i < NB_DEPTH_PIXELS; i++) {
    if(remove[i])
      continue;
    pointsCam.row(k) = image.row(i);
    k++;
  }
}

int closest_timestamp_idx(long ts, vector<long> timestamps) {
  //TODO do binary search
  long shortest_time = LONG_MAX;
  int index = 0;
  int i = 0;
  for(long l : timestamps) {
    long dt = abs(l - ts);
    if(dt < shortest_time) {
      shortest_time = dt;
      index = i;
    }
    i++;
  }
  return index;
}

Eigen::MatrixXf to_homogeneous(const Eigen::MatrixXf& points) {
  Eigen::MatrixXf P = points;
  P.conservativeResize(points.rows(), points.cols()+1);
  P.col(P.cols()-1) = ones.head(points.rows());
  return P;
}

void apply_transformation(const Eigen::MatrixXf& T, const Eigen::MatrixXf& points, Eigen::MatrixXf& out, int out_dim) {
  Eigen::MatrixXf P = to_homogeneous(points);
  Eigen::MatrixXf temp = (T * P.transpose()).transpose();
  out = (out_dim == 3) ? temp.block(0, 0, temp.rows(), 3) : temp;
}

// RowVector3d hand_color_1(181.0/255, 174.0/255, 150.0/255);
// RowVector3d hand_color_2(79.0/255, 71.0/255, 74.0/255);
// RowVector3d hand_color_3(107.0/255, 81.0/255, 43.0/255);
bool valid_color(const RowVector3d& color) {
  bool not_too_white = color.squaredNorm() < 0.9;
  bool not_too_black = color.squaredNorm() > 0.01;
  bool not_colorful = color.squaredNorm() > 0.8 || color.squaredNorm() < 0.2;
  bool reddish = color.x() > (color.y() + 0.01) && color.x() > (color.z() + 0.01);
  bool dark_reddish = color.squaredNorm() < 0.2 && color.x() > (color.y() + 0.01) && color.x() > (color.z() + 0.01);
  bool greenish = color.y() > (color.x() + 0.01) && color.y() > (color.z() + 0.01); //greenish
  //return not_too_white && not_too_black && greenish;
  return not_colorful && !dark_reddish;
}

void project_on_pv(const Eigen::MatrixXf& pointsWorld, const Eigen::MatrixXf& pv2world, 
  float focalx, float focaly, float principalx, float principaly, int pvWidth, int pvHeight,
  const vector<vector<RowVector3d>>& rgbImage, Eigen::MatrixXd& colors, Eigen::MatrixXi& rgb_indices) {

  Eigen::MatrixXf Intrinsic(3, 3);
  Intrinsic << focalx, 0, principalx, 0, focaly, principaly, 0, 0, 1;

  Eigen::MatrixXf pointsWorld_h = to_homogeneous(pointsWorld);

  Eigen::MatrixXf world2pv = pv2world.inverse();
  Eigen::MatrixXf pointsPV = (world2pv * pointsWorld_h.transpose()).transpose().block(0, 0, pointsWorld_h.rows(), 3);
  
  Eigen::MatrixXf pointsPixel_h = (Intrinsic * pointsPV.transpose()).transpose(); //Retrieve homogeneous coordinates in pixel space

  //This assumes we have homogeneous coordinates in pointsPixel_h (points x 3), (x, y, w) format:
  colors.resize(pointsPixel_h.rows(), 3);
  vector<int> has_rgb_indices;
  for(int i = 0; i < pointsPixel_h.rows(); i++) {
    float x_h = pointsPixel_h(i, 0);
    float y_h = pointsPixel_h(i, 1);
    float w = pointsPixel_h(i, 2);
    int x = (int)(x_h / w);
    int y = (int)(y_h / w);
    if (x >= 0 && x < pvWidth && y >= 0 && y < pvHeight) {
      RowVector3d color = rgbImage[y][pvWidth - x];
      if(valid_color(color)) {
        colors.row(i) = color;
        has_rgb_indices.push_back(i);
      } else {
        colors.row(i) = RowVector3d(0, 0, 0); //No color
      }
    } else {
      colors.row(i) = RowVector3d(0, 0, 0); //No color
    }
  }

  rgb_indices.resize(has_rgb_indices.size(), 1);
  int fill_idx = 0;
  for(int i : has_rgb_indices) {
    rgb_indices(fill_idx++) = i;
  }
}

void compute_joints_positions(const Eigen::MatrixXf& world2rig, const vector<MatrixXf>& joints, MatrixXd& djoints) {
  Eigen::MatrixXd Joints(26, 3);
  int i = 0;
  for(MatrixXf Joint : joints) {
    RowVector3f joint_point = (world2rig * Joint).block(0, 3, 3, 1).transpose();
    Joints.row(i) = RowVector3d(joint_point(0), joint_point(1), joint_point(2));
    i++;
  }
  djoints = Joints; //TODO simplify this
}

void compute_joints_positions(const Eigen::MatrixXf& rig2world_matrix,
    const vector<MatrixXf>& joints_left, const vector<MatrixXf>& joints_right,
    MatrixXd& djoints_left, MatrixXd& djoints_right) {

  Eigen::MatrixXf world2rig = rig2world_matrix.inverse();

  compute_joints_positions(world2rig, joints_left, djoints_left);
  compute_joints_positions(world2rig, joints_right, djoints_right);
}

void read_data() {
  nb_frames = read_paths(folder, paths);

  Eigen::MatrixXf Ext;
  read_extrinsics(folder, Ext);
  cam2rig = Ext.inverse();

  read_rig2world(folder, rig2world_matrices, timestamps);

  read_lut(folder, lut);

  read_pv_meta(folder, pv2world_matrices, pv_timestamps, focals, intrinsics_ox, intrinsics_oy, 
    intrinsics_width, intrinsics_height);

  vector<VectorXf> list_gaze_origins;
  vector<VectorXf> list_gaze_directions;
  vector<float> list_gaze_distances;
  vector<MatrixXf> list_head_data;
  read_human_data(folder, human_timestamps, list_joints_left, list_joints_right,
    list_gaze_origins, list_gaze_directions, list_gaze_distances, list_head_data);

  //Read RGB Image
  cout << "Reading RGB data" << endl;
  for(long pv_timestamp : pv_timestamps) {
    cout << "." << flush;
    rgb_images[pv_timestamp] = read_rgb(folder, pv_timestamp, intrinsics_width, intrinsics_height);
  }

  cout << "\nReading Depth data" << endl;
  for(const auto path : paths) {
    cout << "." << flush;
    depth_images.push_back(read_pgm(path));
  }
  cout << endl;

  size_t max_frames = 400;
  nb_frames = min(max_frames, depth_images.size());

  viewer.data().clear();
}

void initialize() {
  //Used for padding in homogeneous coordinates
  ones.resize(NB_DEPTH_PIXELS);
  for(int i = 0; i < NB_DEPTH_PIXELS; i++) {
    ones(i) = 1;
  }

  //Joints edges
  E.resize(24, 2);
  E << 
  1, 2,
  2, 3,
  3, 4,
  4, 5,

  1, 6,
  6, 7,
  7, 8,
  8, 9,
  9, 10,

  1, 11,
  11, 12,
  12, 13,
  13, 14,
  14, 15,

  1, 16,
  16, 17,
  17, 18,
  18, 19,
  19, 20,

  1, 21,
  21, 22,
  22, 23,
  23, 24,
  24, 25;
}

int main(int argc, char *argv[]) {
  if(argc < 2) {
    std::cout << "You must specify a folder name inside ../data/" << std::endl;
    std::cout << "e.g. you can run ./clouds greenbox" << std::endl;
  }
  string sub_folder = argv[1];
  folder = "../data/" + sub_folder + "/"; //Set data path here

  initialize();

  read_data();

  //Setup visualizer
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  viewer.plugins.push_back(&menu);

  menu.callback_draw_viewer_menu = [&]() {
    menu.draw_viewer_menu();

    if (ImGui::CollapsingHeader("Skeletal Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
      if(ImGui::Checkbox("visualize_partials", &visualize_partials)) {
        
      }
    }
  };

  viewer.callback_pre_draw = callback_pre_draw;
  viewer.core().set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
  viewer.data().point_size = 2;
  viewer.core().is_animating = true;
  Vector4f color(1, 1, 1, 1);
  viewer.core().background_color = color * 0.5f;
  viewer.launch();
}