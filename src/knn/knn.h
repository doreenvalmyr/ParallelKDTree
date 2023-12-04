#ifndef KNN_H
#define KNN_H

#include <iostream>
#include "../kdTree/kdTree.h"

using namespace std;

struct DistanceNode {
  double distance;
  const KDNode* node;
};

class KNN {
public:
  vector<DataPoint> nearestNeighbors;

  void kNNSearch(const KDTree& kdTree, const vector<double>& target, int k);

  void printNearestNeighbors(vector<DataPoint>& nearestNeighbors) {
    cout << "\nList of " << nearestNeighbors.size() << " nearest neighbors:" << endl;
    for (auto neighbor : nearestNeighbors) {
      int numFeatures = neighbor.features.size();
      for (int i = 0; i < numFeatures; i++) {
        cout << neighbor.features[i] << " ";
      }
      cout << neighbor.label << endl;
    }
    cout << "\n";
  }
};

#endif
