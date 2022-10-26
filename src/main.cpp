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

namespace fs = std::filesystem;

using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

std::vector<string> paths;
std::vector<Eigen::MatrixXd> ddata;
Eigen::MatrixXd V, N;
Eigen::MatrixXi F;

Eigen::VectorXf ones;

Viewer viewer;

typedef std::vector<std::vector<unsigned short int>> DepthData;

std::vector<DepthData> all_depths;

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
  viewer.data().add_points(ddata[l], Eigen::RowVector3d(0, 0, 0));
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
  for (const auto & entry : fs::directory_iterator("../data/ahat")) {
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
  ifstream infile("../data/raw_ahat/Depth AHaT_extrinsics.txt");
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

void read_rig2world(vector<Eigen::MatrixXf>& rig2world_vec, vector<string>& timestamps) {
  ifstream infile("../data/raw_ahat/Depth AHaT_rig2world.txt");
  string inputLine = "";
  for(int k = 0; k < nb_frames; k++) {
    Eigen::MatrixXf M(4, 4);
    int j = 0, i = 0;
    getline(infile, inputLine);
    stringstream line(inputLine);
    string timestamp;
    string next_val;
    getline(line, timestamp, ',');
    timestamps.push_back(timestamp);
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

void read_lut(Eigen::MatrixXf& mat) {
  ifstream infile("../data/raw_ahat/Depth AHaT_lut.bin", ios::binary);
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
  cout << cam_calibration.rows() << " " << cam_calibration.cols() << endl;
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
  cout << points.rows() << " " << points.cols() << endl;
  MatrixXf P = points;
  P.conservativeResize(points.rows(), points.cols()+1);
  P.col(P.cols()-1) = ones.head(points.rows());
  Eigen::MatrixXf temp = (T * P.transpose()).transpose();
  if(out_dim == 3)
    out = temp.block(0, 0, temp.rows(), 3);
  else
    out = temp;
}

void visualize_raw_data() {
  for (const auto & entry : fs::directory_iterator("../data/raw_ahat/pgm")) {
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
  vector<string> timestamps;
  read_rig2world(rig2world_matrices, timestamps);

  Eigen::MatrixXf lut;
  read_lut(lut);

  Eigen::MatrixXf D;
  Eigen::MatrixXf transformed;
  Eigen::MatrixXd V;

  int kk = 0;
  for (auto path : paths) {
    DepthData data = read_pgm(path); //TODO change path
    all_depths.push_back(data);

    depth_map_to_pc_px(D, data, lut, 0.2, 1);

    apply_transformation(cam2rig, D, transformed, 3); //TODO change index here
    V = transformed.cast<double>();
    ddata.push_back(V);
    kk++;
    // if(kk == 40)
    //   break;
  }

  //nb_frames = 40;

  viewer.data().clear();
  viewer.data().add_points(V, Eigen::RowVector3d(0, 0, 0));
  viewer.core().align_camera_center(V);
}

int main(int argc, char *argv[]) {

  ones.resize(512 * 512);
  for(int i = 0; i < 512 * 512; i++) {
    ones(i) = 1;
  }

  visualize_raw_data();

  //visualize_pcds();

  igl::opengl::glfw::imgui::ImGuiMenu menu;
  viewer.plugins.push_back(&menu);

  menu.callback_draw_viewer_menu = [&]() {
    menu.draw_viewer_menu();
  };

  viewer.callback_pre_draw = callback_pre_draw;

  viewer.core().set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);
  viewer.data().point_size = 1;
  viewer.launch();
}