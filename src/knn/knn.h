#ifndef KNN_H
#define KNN_H

#include <iostream>
#include "../kdTree/kdTree.h"

using namespace std;

struct DistanceNode {
  double distance;
  const KDNode* node;

  // Custom comparison function for sorting in the priority queue
  // Higher priority given to smaller distances from target point
  static bool compareByDistance(const DistanceNode& a, const DistanceNode& b) {
    return a.distance > b.distance;
  }
};

class KNN {
public:
  // Constructor
  

  vector<DataPoint> kNNSearch(const KDTree& kdTree, const vector<double>& target, int k);
};

#endif
