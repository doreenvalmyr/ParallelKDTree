#include <cstddef>
#include <vector>
#include <cmath>
#include <iostream>

class KDNode
{
public:
  KDNode* left;
  KDNode* right;
  std::vector<double> features;
  int label;
  
  // Constructor to initialize KDNode with features
  KDNode() : left(nullptr), right(nullptr), label(0) {}
};
