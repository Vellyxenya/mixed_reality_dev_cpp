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

#include "utils.h"
#include "loading.h"

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

std::vector<DepthData> all_depths;
std::vector<string> paths;
std::vector<Eigen::MatrixXd> ddata;
std::vector<Eigen::MatrixXd> dcolors;
std::vector<Eigen::MatrixXd> djointsleft;
std::vector<Eigen::MatrixXd> djointsright;
Eigen::MatrixXd V;
Eigen::MatrixXi E;

Eigen::VectorXf ones;
Viewer viewer;
string folder;

int l; //frame counter
int nb_frames;

bool callback_pre_draw(Viewer& viewer);

void render_frame() {
  viewer.data().clear();
  viewer.data().add_points(ddata[l], dcolors[l]);
  //TODO calling set_edges twices overrides the first one
  viewer.data().set_edges(djointsleft[l], E, Eigen::RowVector3d(0, 0, 1));
  viewer.data().set_edges(djointsright[l], E, Eigen::RowVector3d(1, 0, 0));
}

//Compute what to display on the next frame
bool callback_pre_draw(Viewer& viewer) {
  //Update frame count
  ++l;
  l %= nb_frames;
  render_frame();
  return false;
}

void depth_map_to_pc_px(Eigen::MatrixXf& D, const DepthData& depth_map, 
  const Eigen::MatrixXf& cam_calibration, float clamp_min = 0, float clamp_max = 1) {
  clamp_min *= 1000;
  clamp_max *= 1000;

  Eigen::MatrixXf image(512*512, 3);
  vector<int> indices_to_remove;
  vector<bool> remove = vector<bool>(512*512, false);
  int k = 0;
  int total_removed = 0;
  for(int j = 0; j < depth_map.size(); j++) {
    for(int i = 0; i < depth_map[0].size(); i++) {
      float d = depth_map[j][i]; // / 65536.0;
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

  int remaining_points = 512 * 512 - total_removed;
  D.resize(remaining_points, 3);
  k = 0;
  for(int i = 0; i < 512*512; i++) {
    if(remove[i])
      continue;
    D.row(k) = image.row(i);
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

RowVector3d hand_color_1(181.0/255, 174.0/255, 150.0/255);
RowVector3d hand_color_2(79.0/255, 71.0/255, 74.0/255);
RowVector3d hand_color_3(107.0/255, 81.0/255, 43.0/255);
bool valid_color(const RowVector3d& color) {
  /*float tolerance = 0.6;
  return (color - hand_color_1).squaredNorm() > tolerance || (color - hand_color_2).squaredNorm() > tolerance
   || (color - hand_color_3).squaredNorm() > tolerance;*/
  return color.y() > (color.x() + 0.01) && color.y() > (color.z() + 0.01);
}

void project_on_pv(const Eigen::MatrixXf& pointsWorld, const Eigen::MatrixXf& pv2world, 
  float focalx, float focaly, float principalx, float principaly, int pvWidth, int pvHeight,
  const vector<vector<RowVector3d>>& rgbImage, Eigen::MatrixXd& colors, vector<int>& has_rgb_indices) {

  Eigen::MatrixXf Intrinsic(3, 3);
  Intrinsic << focalx, 0, principalx, 0, focaly, principaly, 0, 0, 1;

  Eigen::MatrixXf pointsWorld_h = to_homogeneous(pointsWorld);

  Eigen::MatrixXf world2pv = pv2world.inverse();
  Eigen::MatrixXf pointsPV = (world2pv * pointsWorld_h.transpose()).transpose().block(0, 0, pointsWorld_h.rows(), 3);
  
  Eigen::MatrixXf pointsPixel_h = (Intrinsic * pointsPV.transpose()).transpose(); //Retrieve homogeneous coordinates in pixel space
  //This assumes we have homogeneous coordinates in pointsPixel_h (points x 3), (x, y, w) format:
  colors.resize(pointsPixel_h.rows(), 3);
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
}

void compute_joints_positions(const Eigen::MatrixXf& world2rig, const vector<MatrixXf>& joints, vector<MatrixXd>& djoints) {
  Eigen::MatrixXd Joints(26, 3);
  int i = 0;
  for(MatrixXf Joint : joints) {
    RowVector3f joint_point = (world2rig * Joint).block(0, 3, 3, 1).transpose();
    Joints.row(i) = RowVector3d(joint_point(0), joint_point(1), joint_point(2));
    i++;
  }
  djoints.push_back(Joints);
}

void compute_joints_positions(const Eigen::MatrixXf& rig2world_matrix,
    const vector<MatrixXf>& joints_left, const vector<MatrixXf>& joints_right,
    vector<MatrixXd>& djoints_left, vector<MatrixXd>& djoints_right) {

  Eigen::MatrixXf world2rig = rig2world_matrix.inverse();

  compute_joints_positions(world2rig, joints_left, djoints_left);
  compute_joints_positions(world2rig, joints_right, djoints_right);
}

void visualize_raw_data() {
  nb_frames = read_paths(folder, paths);

  Eigen::MatrixXf Ext;
  read_extrinsics(folder, Ext);
  Eigen::MatrixXf cam2rig = Ext.inverse();

  vector<Eigen::MatrixXf> rig2world_matrices;
  vector<long> timestamps;
  read_rig2world(folder, rig2world_matrices, timestamps);

  Eigen::MatrixXf lut;
  read_lut(folder, lut);

  vector<Eigen::MatrixXf> pv2world_matrices;
  vector<long> pv_timestamps;
  vector<std::pair<float, float>> focals;
  float intrinsics_ox, intrinsics_oy;
  int intrinsics_width, intrinsics_height;
  read_pv_meta(folder, pv2world_matrices, pv_timestamps, focals, intrinsics_ox, intrinsics_oy, 
    intrinsics_width, intrinsics_height);

  vector<long> human_timestamps;
  vector<vector<MatrixXf>> list_joints_left;
  vector<vector<MatrixXf>> list_joints_right;
  vector<VectorXf> list_gaze_origins;
  vector<VectorXf> list_gaze_directions;
  vector<float> list_gaze_distances;
  vector<MatrixXf> list_head_data;
  read_human_data(folder, human_timestamps, list_joints_left, list_joints_right,
    list_gaze_origins, list_gaze_directions, list_gaze_distances, list_head_data);

  Eigen::MatrixXf D;
  Eigen::MatrixXf pointsRig;
  Eigen::MatrixXf pointsWorld;
  Eigen::MatrixXd V;
  Eigen::MatrixXd colors;

  Eigen::MatrixXd ddata_only_rgb;
  Eigen::MatrixXd dcolors_only_rgb;

  int max_frames = 300;
  int kk = 0;
  for (auto path : paths) {
    cout << "path " << kk << endl;
    long depth_timestamp = timestamps[kk];

    int pv_timestamp_idx = closest_timestamp_idx(depth_timestamp, pv_timestamps);
    long pv_timestamp = pv_timestamps[pv_timestamp_idx];

    int human_timestamp_idx = closest_timestamp_idx(depth_timestamp, human_timestamps);
    long human_timestamp = human_timestamps[human_timestamp_idx];

    DepthData data = read_pgm(path);
    all_depths.push_back(data);

    //TODO the rig2world matrix does not necessarily have the right timestamp..
    compute_joints_positions(rig2world_matrices[kk], 
      list_joints_left[human_timestamp_idx], list_joints_right[human_timestamp_idx],
      djointsleft, djointsright);

    depth_map_to_pc_px(D, data, lut, 0.2, 0.9);

    //Read RGB Image
    vector<vector<RowVector3d>> rgbImage;
    read_rgb(folder, pv_timestamp, intrinsics_width, intrinsics_height, rgbImage);

    //Transform points from cam coordinatse to rig coordinates
    apply_transformation(cam2rig, D, pointsRig, 3);

    //Transform points from rig coordinates to world
    apply_transformation(rig2world_matrices[kk], pointsRig, pointsWorld, 3);

    //Project points from world coordinates to RGB
    vector<int> has_rgb_indices;
    project_on_pv(pointsWorld, pv2world_matrices[pv_timestamp_idx], 
      focals[kk].first, focals[kk].second, intrinsics_ox, intrinsics_oy, 
      intrinsics_width, intrinsics_height, rgbImage, colors, has_rgb_indices);
    
    V = pointsRig.cast<double>(); //Cast to double so that libigl is happy
    
    Eigen::MatrixXi rgb_indices(has_rgb_indices.size(), 1);
    int fill_idx = 0;
    for(int i : has_rgb_indices) {
      rgb_indices(fill_idx++) = i;
    }
    igl::slice(V, rgb_indices, 1, ddata_only_rgb);
    igl::slice(colors, rgb_indices, 1, dcolors_only_rgb);

    ddata.push_back(ddata_only_rgb);
    dcolors.push_back(dcolors_only_rgb);

    kk++;

    //Debugging
    if(kk == max_frames)
      break;
  }
  //Debugging
  nb_frames = max_frames;

  viewer.data().clear();
  viewer.data().add_points(V, Eigen::RowVector3d(0, 0, 0));
  viewer.core().align_camera_center(V);
}

void initialize() {
  //Used for padding in homogeneous coordinates
  ones.resize(512 * 512);
  for(int i = 0; i < 512 * 512; i++) {
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
  //folder = "../data/raw_ahat/";
  folder = "../data/greenbox/";

  initialize();

  visualize_raw_data();

  //Setup visualizer
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  viewer.plugins.push_back(&menu);

  menu.callback_draw_viewer_menu = [&]() {
    menu.draw_viewer_menu();
  };

  viewer.callback_pre_draw = callback_pre_draw;
  viewer.core().set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
  viewer.data().point_size = 2;
  viewer.core().is_animating = true;
  Vector4f color(1, 1, 1, 1);
  viewer.core().background_color = color * 0.5f;
  viewer.launch();
}