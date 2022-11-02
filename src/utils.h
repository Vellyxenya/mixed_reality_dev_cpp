#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <filesystem>

bool string_contains(const std::string s, const char c) {
  return s.find(c) != std::string::npos;
}

bool ends_with(std::string& s, const std::string& ending) {
    if (s.length() >= ending.length()) {
        return s.compare(s.length() - ending.length(), ending.length(), ending) == 0;
    } else {
        return false;
    }
}

string find_file_ending_with(const string& folder, const string& ending) {
  for (const auto & entry : std::filesystem::directory_iterator(folder)) {
    string path = entry.path();
    if(ends_with(path, ending)) {
        return path;
    }
  }
  throw std::runtime_error("Did not find any file ending with '" + ending + "'.");
}

inline void endian_swap(unsigned short int& x) {
    x = (x>>8) | (x<<8);
}

#endif