#include <igl/read_triangle_mesh.h>
#include <igl/readTGF.h>
#include <igl/readDMAT.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

#include "Colors.h"

#include <math.h>
#include <imgui/imgui.h>

#include <string>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <algorithm>
#include <fstream> // ifstream
#include <sstream> // stringstream
#include <limits.h>

namespace fs = std::filesystem;

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

std::vector<string> paths;
std::vector<Eigen::MatrixXd> ddata;
std::vector<Eigen::MatrixXd> dcolors;
std::vector<Eigen::MatrixXd> djointsleft;
std::vector<Eigen::MatrixXd> djointsright;
Eigen::MatrixXd V, N;
Eigen::MatrixXi F;
Eigen::MatrixXi E;

Eigen::VectorXf ones;

Viewer viewer;

typedef std::vector<std::vector<unsigned short int>> DepthData;

std::vector<DepthData> all_depths;

string folder;

//bools
bool animate = true; //Animate or pause the animation

//ints
int L; //Nb frames
int l; //frame counter

int nb_frames;

void draw_viewer_menu_minimal(Viewer* viewer);
bool callback_mouse_down(Viewer& viewer, int button, int modifier);
bool callback_mouse_move(Viewer& viewer, int mouse_x, int mouse_y);
bool callback_mouse_up(Viewer& viewer, int button, int modifier);
bool callback_pre_draw(Viewer& viewer);
bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);

void render_frame() {
  viewer.data().clear();
  viewer.data().set_points(ddata[l], dcolors[l]);
  //TODO calling set_edges twices overrides the first one
  viewer.data().set_edges(djointsleft[l], E, Eigen::RowVector3d(0, 0, 1));
  viewer.data().set_edges(djointsright[l], E, Eigen::RowVector3d(1, 0, 0));
  //viewer.data().add_points(ddata[l], Eigen::RowVector3d(0, 0, 0));
}

//Compute what to display on the next frame
bool callback_pre_draw(Viewer& viewer) {
  viewer.core().is_animating = animate;
  if(animate) {
    //Update frame count
    ++l;
    l %= nb_frames;
    render_frame();
  }

  return false;
}

bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers) {
  return false;
}

bool callback_mouse_down(Viewer& viewer, int button, int modifier) {
  return false;
}

bool callback_mouse_move(Viewer& viewer, int mouse_x, int mouse_y) {
  return false;
}

bool callback_mouse_up(Viewer& viewer, int button, int modifier) {
  return false;
}

bool string_contains(string s, char c) {
  return s.find(c) != std::string::npos;
}

void visualize_pcds() {
  for (const auto & entry : fs::directory_iterator(folder)) {
    paths.push_back(entry.path());
  }
  std::sort(paths.begin(), paths.end());

  Eigen::MatrixXd V_temp;
  for (auto path : paths) {
    igl::readPLY(path, V_temp, F, N);
    ddata.push_back(V_temp);
  }

  l = 0; //Initialize frame count
  L = paths.size(); //Nb frames

  render_frame();
  viewer.core().align_camera_center(V_temp);
}

inline void endian_swap(unsigned short int& x) {
    x = (x>>8) | (x<<8);
}

DepthData read_pgm(string pgm_file_path) {  
  int row = 0, col = 0, num_of_rows = 0, num_of_cols = 0;
  stringstream ss;    
  ifstream infile(pgm_file_path, ios::binary);

  string inputLine = "";

  getline(infile, inputLine);      // read the first line : P5
  if(inputLine.compare("P5") != 0) cerr << "Version error" << endl;

  ss << infile.rdbuf();   //read the third line : width and height
  ss >> num_of_cols >> num_of_rows;

  int max_val;  //maximum intensity value : 255
  ss >> max_val;
  ss.ignore();

  unsigned short int pixel;

  DepthData data(num_of_rows, std::vector<unsigned short int>(num_of_cols));

  for (row = 0; row < num_of_rows; row++) {
    for (col = 0; col < num_of_cols; col++) {
      ss.read((char*)&pixel, 2);
      endian_swap(pixel);
      data[row][col] = pixel;
    }
  }

  return data;
}

void read_extrinsics(Eigen::MatrixXf& Ext) {
  ifstream infile(folder+"Depth AHaT_extrinsics.txt");
  string inputLine = "";
  Ext.resize(4, 4);
  int j = 0, i = 0;
  while(getline(infile, inputLine, ',')){
    float f = std::stof(inputLine);
    Ext(j, i) = f;
    i++;
    if(i == 4) {
      i = 0;
      j++;
    }
  }
}

void read_lut(Eigen::MatrixXf& mat) {
  ifstream infile(folder+"Depth AHaT_lut.bin", ios::binary);
  float f;
  int y = 0;
  int x = 0;
  mat.resize(512 * 512, 3);
  while (infile.read(reinterpret_cast<char*>(&f), sizeof(float))) {
    mat(y, x) = f;
    x++;
    if(x >= 3) {
      x = 0;
      y++;
    }
  }
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

  Eigen::MatrixXf points_only(remaining_points, 3);
  k = 0;
  for(int i = 0; i < 512*512; i++) {
    if(remove[i])
      continue;
    points_only.row(k) = image.row(i);
    k++;
  }
  
  D = points_only;
}

void apply_transformation(const Eigen::MatrixXf& T, const Eigen::MatrixXf& points, Eigen::MatrixXf& out, int out_dim) {
  MatrixXf P = points;
  P.conservativeResize(points.rows(), points.cols()+1);
  P.col(P.cols()-1) = ones.head(points.rows());
  Eigen::MatrixXf temp = (T * P.transpose()).transpose();
  if(out_dim == 3)
    out = temp.block(0, 0, temp.rows(), 3);
  else
    out = temp;
}

void read_rig2world(vector<Eigen::MatrixXf>& rig2world_vec, vector<long>& timestamps) {
  ifstream infile(folder+"Depth AHaT_rig2world.txt");
  string input_line = "";
  for(int k = 0; k < nb_frames; k++) {
    Eigen::MatrixXf M(4, 4);
    int j = 0, i = 0;
    getline(infile, input_line);
    stringstream line(input_line);
    string timestamp;
    string next_val;
    getline(line, timestamp, ',');
    timestamps.push_back(stol(timestamp));
    while(getline(line, next_val, ',')){
      float f = std::stof(next_val);
      M(j, i) = f;
      i++;
      if(i == 4) {
        i = 0;
        j++;
      }
    }
    rig2world_vec.push_back(M);
  }
}

void read_pv_meta(vector<Eigen::MatrixXf>& pv2world_matrices, vector<long>& timestamps, 
  vector<std::pair<float, float>>& focals, float& intrinsics_ox, float& intrinsics_oy,
  int& intrinsics_width, int& intrinsics_height) {
  ifstream infile(folder+"2022-10-26-120549_pv.txt");
  string input_line = "";

  getline(infile, input_line, ',');
  intrinsics_ox = stof(input_line);
  getline(infile, input_line, ',');
  intrinsics_oy = stof(input_line);
  getline(infile, input_line, ',');
  intrinsics_width = stof(input_line);
  getline(infile, input_line);
  intrinsics_height = stof(input_line);

  while(true) {
    if(!getline(infile, input_line)) break;
    Eigen::MatrixXf M(4, 4);
    
    stringstream line(input_line);
    string next_val;
    getline(line, next_val, ',');
    timestamps.push_back(stol(next_val));
    getline(line, next_val, ',');
    int focalx = stof(next_val);
    getline(line, next_val, ',');
    int focaly = stof(next_val);

    focals.push_back(std::pair<float, float>(focalx, focaly));

    for(int j = 0; j < 4; j++) {
      for(int i = 0; i < 4; i++) {
        getline(line, next_val, ',');
        M(j, i) = std::stof(next_val);
      }
    }
    pv2world_matrices.push_back(M);
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

void read_rgb(long pv_timestamp, int width, int height, vector<vector<RowVector3d>>& colors) {
  string file_path = folder+"PV/"+to_string(pv_timestamp)+".bytes";
  ifstream infile(file_path, ios::binary);
  unsigned int val;
  int y = 0;
  int x = 0;
  Eigen::MatrixXf RGB;
  colors = vector<vector<RowVector3d>>(height, vector<RowVector3d>(width));
  RGB.resize(height, width);
  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {
      infile.read(reinterpret_cast<char*>(&val), sizeof(unsigned int));
      int r = (val & 0x00FF0000) >> 16;
      int g = (val & 0x0000FF00) >> 8;
      int b = (val & 0x000000FF);
      colors[y][x] = RowVector3d(r, g, b) / 255.0;
    }
  }
}

Eigen::MatrixXf to_homogeneous(const Eigen::MatrixXf& points) {
  MatrixXf P = points;
  P.conservativeResize(points.rows(), points.cols()+1);
  P.col(P.cols()-1) = ones.head(points.rows());
  return P;
}

void project_on_pv(const Eigen::MatrixXf& pointsWorld, const Eigen::MatrixXf& pv2world, 
  float focalx, float focaly, float principalx, float principaly, int pvWidth, int pvHeight,
  const vector<vector<RowVector3d>>& rgbImage, Eigen::MatrixXd& colors) {

  Eigen::MatrixXf Intrinsic(3, 3);
  Intrinsic << focalx, 0, principalx, 0, focaly, principaly, 0, 0, 1;

  Eigen::MatrixXf pointsWorld_h = to_homogeneous(pointsWorld);

  Eigen::MatrixXf world2pv = pv2world.inverse();
  Eigen::MatrixXf pointsPV = (world2pv * pointsWorld_h.transpose()).transpose().block(0, 0, pointsWorld_h.rows(), 3);
  
  Eigen::MatrixXf pointsPixel_h = (Intrinsic * pointsPV.transpose()).transpose(); //Retrieve homogeneous coordinates in pixel space
  //TODO this assumes we have homogeneous coordinates in pointsPixel_h (points x 3), (x, y, w) format:
  colors.resize(pointsPixel_h.rows(), 3);
  for(int i = 0; i < pointsPixel_h.rows(); i++) {
    float x_h = pointsPixel_h(i, 0); //TODO is this the right ordering?
    float y_h = pointsPixel_h(i, 1);
    float w = pointsPixel_h(i, 2);
    int x = (int)(x_h / w);
    int y = (int)(y_h / w);
    if (x >= 0 && x < pvWidth && y >= 0 && y < pvHeight) {
      colors.row(i) = rgbImage[y][pvWidth - x];
    } else {
      colors.row(i) = RowVector3d(0, 0, 0); //No color
      //TODO may want to mark these points as useless, i.e. discard them
    }
  }
}

void read_human_data(vector<long>& timestamps, vector<vector<MatrixXf>>& list_joints_left,
  vector<vector<MatrixXf>>& list_joints_right,
  vector<VectorXf>& list_gaze_origins,
  vector<VectorXf>& list_gaze_directions,
  vector<float>& list_gaze_distances,
  vector<MatrixXf>& list_head_data) {

  ifstream infile(folder+"2022-10-26-120549_head_hand_eye.csv");
  string input_line = "";
  const int joint_count = 26;

  while(true) {
    if(!getline(infile, input_line)) break;
    stringstream line(input_line);
    string next_val;

    //Read timestamp
    getline(line, next_val, ',');
    timestamps.push_back(stol(next_val));

    //Read head data
    Eigen::MatrixXf Head(4, 4);
    for(int j = 0; j < 4; j++) {
      for(int i = 0; i < 4; i++) {
        getline(line, next_val, ',');
        Head(j, i) = std::stof(next_val);
      }
    }
    list_head_data.push_back(Head);
    
    //Read left hand data
    getline(line, next_val, ',');
    bool left_hand_available = std::stoi(next_val) == 1;
    vector<Eigen::MatrixXf> joints_left;
    for(int joint = 0; joint < joint_count; joint++) {
      Eigen::MatrixXf Joint(4, 4);
      for(int j = 0; j < 4; j++) {
        for(int i = 0; i < 4; i++) {
          getline(line, next_val, ',');
          Joint(j, i) = std::stof(next_val);
        }
      }
      joints_left.push_back(Joint);
    }
    list_joints_left.push_back(joints_left);

    //Read right hand data
    getline(line, next_val, ',');
    bool right_hand_available = std::stoi(next_val) == 1;
    vector<Eigen::MatrixXf> joints_right;
    for(int joint = 0; joint < joint_count; joint++) {
      Eigen::MatrixXf Joint(4, 4);
      for(int j = 0; j < 4; j++) {
        for(int i = 0; i < 4; i++) {
          getline(line, next_val, ',');
          Joint(j, i) = std::stof(next_val);
        }
      }
      joints_right.push_back(Joint);
    }
    list_joints_right.push_back(joints_right);

    //Read gaze data
    getline(line, next_val, ',');
    bool gaze_available = std::stoi(next_val) == 1;
    Eigen::VectorXf GazeOrigin(4);
    for(int i = 0; i < 4; i++) {
      getline(line, next_val, ',');
      GazeOrigin(i) = std::stof(next_val);
    }
    list_gaze_origins.push_back(GazeOrigin);

    Eigen::VectorXf GazeDirection(4);
    for(int i = 0; i < 4; i++) {
      getline(line, next_val, ',');
      GazeDirection(i) = std::stof(next_val);
    }
    list_gaze_directions.push_back(GazeDirection);

    getline(line, next_val, ',');
    float gaze_distance = std::stof(next_val);
    list_gaze_distances.push_back(gaze_distance);
  }
}

void compute_joints_positions(Eigen::MatrixXf& rig2world_matrix,
    vector<MatrixXf>& joints_left, vector<MatrixXf>& joints_right) {

  Eigen::MatrixXf world2rig = rig2world_matrix.inverse();

  Eigen::MatrixXd JointsLeft(26, 3);
  int i = 0;
  for(MatrixXf Joint : joints_left) {
    RowVector3f joint_point = (world2rig * Joint).block(0, 3, 3, 1).transpose();
    JointsLeft.row(i) = RowVector3d(joint_point(0), joint_point(1), joint_point(2));
    i++;
  }
  djointsleft.push_back(JointsLeft);

  Eigen::MatrixXd JointsRight(26, 3);
  i = 0;
  for(MatrixXf Joint : joints_right) {
    RowVector3f joint_point = (world2rig * Joint).block(0, 3, 3, 1).transpose();
    JointsRight.row(i) = RowVector3d(joint_point(0), joint_point(1), joint_point(2));
    i++;
  }
  djointsright.push_back(JointsRight);
}

void visualize_raw_data() {
  for (const auto & entry : fs::directory_iterator(folder+"Depth AHaT")) {
    string path = entry.path();
    if(path[path.length()-7] == '_')
      continue;
    paths.push_back(entry.path());
  }
  std::sort(paths.begin(), paths.end());
  nb_frames = paths.size();

  Eigen::MatrixXf Ext;
  read_extrinsics(Ext);
  Eigen::MatrixXf cam2rig = Ext.inverse();
  cout << cam2rig << endl;

  vector<Eigen::MatrixXf> rig2world_matrices;
  vector<long> timestamps;
  read_rig2world(rig2world_matrices, timestamps);

  Eigen::MatrixXf lut;
  read_lut(lut);

  vector<Eigen::MatrixXf> pv2world_matrices;
  vector<long> pv_timestamps;
  vector<std::pair<float, float>> focals;
  float intrinsics_ox, intrinsics_oy;
  int intrinsics_width, intrinsics_height;
  read_pv_meta(pv2world_matrices, pv_timestamps, focals, intrinsics_ox, intrinsics_oy, 
    intrinsics_width, intrinsics_height);

  vector<long> human_timestamps;
  vector<vector<MatrixXf>> list_joints_left;
  vector<vector<MatrixXf>> list_joints_right;
  vector<VectorXf> list_gaze_origins;
  vector<VectorXf> list_gaze_directions;
  vector<float> list_gaze_distances;
  vector<MatrixXf> list_head_data;
  read_human_data(human_timestamps, list_joints_left, list_joints_right,
    list_gaze_origins, list_gaze_directions, list_gaze_distances, list_head_data);

  Eigen::MatrixXf D;
  Eigen::MatrixXf pointsRig;
  Eigen::MatrixXf pointsWorld;
  Eigen::MatrixXd V;
  Eigen::MatrixXd colors;

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
      list_joints_left[human_timestamp_idx], list_joints_right[human_timestamp_idx]);

    depth_map_to_pc_px(D, data, lut, 0, 1);

    //Read RGB Image
    vector<vector<RowVector3d>> rgbImage;
    read_rgb(pv_timestamp, intrinsics_width, intrinsics_height, rgbImage);

    //Transform points from cam coordinatse to rig coordinates
    apply_transformation(cam2rig, D, pointsRig, 3);

    //Transform points from rig coordinates to world
    // cout << "pointsRig: " << pointsRig.rows() << " " << pointsRig.cols() << endl;
    // cout << "rig2world_matrices[kk]: " << rig2world_matrices[kk].rows() << " " << rig2world_matrices[kk].cols() << endl;
    apply_transformation(rig2world_matrices[kk], pointsRig, pointsWorld, 3);

    //Project points from world coordinates to RGB
    //cout << "pointsWorld: " << pointsWorld.rows() << " " << pointsWorld.cols() << endl;
    //cout << focals[kk].first << " " << focals[kk].second << " " << intrinsics_ox << " " << intrinsics_oy << endl; 
    project_on_pv(pointsWorld, pv2world_matrices[pv_timestamp_idx], 
      focals[kk].first, focals[kk].second, intrinsics_ox, intrinsics_oy, 
      intrinsics_width, intrinsics_height, rgbImage, colors);
    dcolors.push_back(colors);

    //Cast to double so that libigl is happy
    V = pointsRig.cast<double>();
    ddata.push_back(V);
    kk++;

    //Debugging
     if(kk == 20)
       break;
  }
  //Debugging
  nb_frames = 20;

  viewer.data().clear();
  viewer.data().add_points(V, Eigen::RowVector3d(0, 0, 0));
  viewer.core().align_camera_center(V);
}

int main(int argc, char *argv[]) {

  //folder = "../data/raw_ahat/";
  folder = "../data/umbrella/";

  ones.resize(512 * 512);
  for(int i = 0; i < 512 * 512; i++) {
    ones(i) = 1;
  }

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

  visualize_raw_data();

  //visualize_pcds();

  igl::opengl::glfw::imgui::ImGuiMenu menu;
  viewer.plugins.push_back(&menu);

  menu.callback_draw_viewer_menu = [&]() {
    menu.draw_viewer_menu();
  };

  viewer.callback_pre_draw = callback_pre_draw;

  viewer.core().set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
  viewer.data().point_size = 0.8f;
  Vector4f color(1, 1, 1, 1);
  viewer.core().background_color = color * 0.5f;
  //viewer.core().background_color.setOnes();
  viewer.launch();
}