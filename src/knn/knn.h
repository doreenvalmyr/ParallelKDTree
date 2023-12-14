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

struct DistanceNode2 {
  double distance;
  int label;
};

// Calculate Euclidean distance between two points
// Generalizable to points with any number of features
double calculateDistance(vector<double> point1, vector<double> point2) {
  // Throw exception for invalid input
  if (point1.size() != point2.size()) {
    throw invalid_argument("Points must have the same number of features");
  }

  double distance = 0.0;

  for (size_t i = 0; i < point1.size(); ++i) {
    double diff = point1[i] - point2[i];
    distance += diff * diff;
  }

  return sqrt(distance);
}

// Insert a node into the nearest neighbor vector in the correct position
// Nearest neighbor vector is sorted in ascending order of distances
void insertAndSortNeighbors(vector<DistanceNode>& nearestNeighbors, DistanceNode& neighbor, size_t k) {
  // Use lower_bound to find the position to insert the new element
  auto it = std::lower_bound(nearestNeighbors.begin(),
                            nearestNeighbors.end(),
                            neighbor,
                            [](const DistanceNode& a, const DistanceNode& b) {
    return a.distance < b.distance;
  });

  nearestNeighbors.insert(it, neighbor);

  // Remove the last element if there are more than k neighbors
  if (nearestNeighbors.size() > k) {
    nearestNeighbors.pop_back();
  }
}

// Search KDTree for nearest neighbors
void kNNSearchIterative(const KDNode* root, const vector<double>& target, size_t k,
                        vector<DistanceNode>& nearestNeighbors) {
  if (root == nullptr) {
    return;
  }

  stack<pair<const KDNode*, int>> nodeStack;
  nodeStack.push({root, 0});

  while (!nodeStack.empty()) {
    const KDNode* currentNode = nodeStack.top().first;
    int depth = nodeStack.top().second;
    nodeStack.pop();

    if (currentNode == nullptr) {
      continue;
    }

    int axis = depth % target.size();

    double distance;
    try {
      distance = calculateDistance(target, currentNode->features);
    } catch (const invalid_argument& error) {
      cerr << "Exception caught: " << error.what() << endl;
      continue;
    }

    DistanceNode neighbor = {distance, currentNode};
    insertAndSortNeighbors(nearestNeighbors, neighbor, k);

    if (target[axis] < currentNode->features[axis]) {
      nodeStack.push({currentNode->right.get(), depth + 1});
      nodeStack.push({currentNode->left.get(), depth + 1});
    } else {
      nodeStack.push({currentNode->left.get(), depth + 1});
      nodeStack.push({currentNode->right.get(), depth + 1});
    }
  }
}

// Parse target point (vector of features)
vector<double> parseInputVector(const std::string& input) {
  istringstream iss(input);
  double feature;
  vector<double> point;
  while (iss >> feature) {
    point.push_back(feature);
  }
  return point;
}

class KNN {
public:
  vector<DataPoint> nearestNeighbors;
  int targetLabel;

  // void kNNSearch(const KDTree& kdTree, const vector<double>& target, int k);

  void kNNSearchParallelOpenMP(const KDTree& kdTree, const vector<double>& target, int k);
  
  void kNNSearchParallelMPI(const vector<DataPoint>& data, const vector<double>& target, int k, int rank, int nproc);

  // Find k nearest neighbors of target point
  void kNNSearch(const KDTree& kdTree, const vector<double>& target, int k) {
    vector<DistanceNode> nearestNeighborsVector;

    // Add k nearest neighbors to result using kdTree
    kNNSearchIterative(kdTree.root.get(), target, (size_t)k, nearestNeighborsVector);

    // Collect the results from the priority queue
    while (!nearestNeighborsVector.empty()) {
      DistanceNode point = nearestNeighborsVector.back();
      nearestNeighbors.push_back({ point.node->features, point.node->label });
      nearestNeighborsVector.pop_back();
    }
  }

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
