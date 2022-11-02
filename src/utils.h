#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <Eigen/Dense>

bool string_contains(std::string s, char c) {
  return s.find(c) != std::string::npos;
}

inline void endian_swap(unsigned short int& x) {
    x = (x>>8) | (x<<8);
}

#endif