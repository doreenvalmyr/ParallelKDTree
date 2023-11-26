#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>

using namespace std;

// Function to check if a string is a positive integer
bool isPositiveInteger(const string& s) {
  for (char c : s) {
    if (!isdigit(c)) {
      return false;
    }
  }
  return true;
}

#endif
