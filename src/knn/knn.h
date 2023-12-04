#ifndef KNN_H
#define KNN_H

#include <iostream>
#include <map>
#include "../kdTree/kdTree.h"

using namespace std;

struct DistanceNode {
  double distance;
  const KDNode* node;
};

class KNN {
public:
  vector<DataPoint> nearestNeighbors;
  int targetLabel;

  void kNNSearch(const KDTree& kdTree, const vector<double>& target, int k);

  void printNearestNeighbors() {
    cout << "\nList of " << nearestNeighbors.size() << " nearest neighbors:" << endl;
    for (auto neighbor : nearestNeighbors) {
      int numFeatures = neighbor.features.size();
      for (int i = 0; i < numFeatures; i++) {
        cout << neighbor.features[i] << " ";
      }
      cout << neighbor.label << endl;
    }
  }

  // Determine the target point's label based on its nearest neighbors
  void findTargetLabel() {
    map<int,int> labelCounts;

    // Count occurrences of each label among nearest neighbors
    for (auto neighbor : nearestNeighbors) {
      labelCounts[neighbor.label]++;
    }

    // Find the label with the highest count
    int maxCount = 0;
    for (auto keyVal : labelCounts) {
      if (keyVal.second > maxCount) {
        maxCount = keyVal.second;
        targetLabel = keyVal.first;
      }
    }

    cout << "\nPredicted label for the target point: " << targetLabel << endl;
  }
};

#endif
