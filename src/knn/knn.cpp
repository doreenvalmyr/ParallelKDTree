#include <iostream>
#include <unistd.h>
#include <vector>
#include <queue>
#include <cmath>
#include "../kdTree/kdTree.h"
#include "../utils.h"
#include "knn.h"

using namespace std;
using KNNPriorityQueue = priority_queue<DistanceNode, vector<DistanceNode>, decltype(&DistanceNode::compareByDistance)>;

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

// Search KD Tree recursively to determine k nearest neighbors
void kNNSearchRecursive(const KDNode* currentNode, const vector<double>& target, int k,
                        KNNPriorityQueue& nearestNeighbors, int depth) {
  if (currentNode == nullptr) {
    return;
  }
  cout << "kNNSearchRecursive 1" << endl;

  int axis = depth % target.size();

  // Calculate distance of target point to current node
  double distance;
  try {
    double distance = calculateDistance(target, currentNode->features);
  } catch (const invalid_argument& error) {
    cerr << "Exception caught: " << error.what() << endl;
  }

  cout << "kNNSearchRecursive 2" << endl;
  cout << "Distance: " << distance << endl;

  // Update nearest neighbors queue
  DistanceNode neighbor = {distance, currentNode};
  nearestNeighbors.push(neighbor);
  cout << distance << endl;
  cout << "HEY 1" << endl;
  if (nearestNeighbors.size() > k) {
    cout << "HELLO 1" << endl;
    nearestNeighbors.pop();  // Keep only k nearest neighbors in the queue
  }

  cout << "kNNSearchRecursive 3" << endl;

  // Recursively search the side of the splitting plane that contains the target point
  if (target[axis] < currentNode->features[axis]) {
    kNNSearchRecursive(currentNode->left.get(), target, k, nearestNeighbors, depth + 1);
  } else {
    kNNSearchRecursive(currentNode->right.get(), target, k, nearestNeighbors, depth + 1);
  }

  cout << "kNNSearchRecursive 4" << endl;

  // Check if we need to search the other side (if there could be closer points)
  if (nearestNeighbors.size() < k || abs(target[axis] - currentNode->features[axis]) < nearestNeighbors.top().distance) {
    if (target[axis] < currentNode->features[axis]) {
      kNNSearchRecursive(currentNode->right.get(), target, k, nearestNeighbors, depth + 1);
    } else {
      kNNSearchRecursive(currentNode->left.get(), target, k, nearestNeighbors, depth + 1);
    }
  }

  cout << "kNNSearchRecursive 5" << endl;
}

// Find k nearest neighbors of target point
vector<DataPoint> KNN::kNNSearch(const KDTree& kdTree, const vector<double>& target, int k) {
  cout << "kNNSearch 1" << endl;
  // Create priority that is a minheap based on distance
  KNNPriorityQueue nearestNeighbors;
  cout << "kNNSearch 2" << endl;

  // Add k nearest neighbors to queue using kdTree
  kNNSearchRecursive(kdTree.root.get(), target, k, nearestNeighbors, 0);

  cout << "kNNSearch 3" << endl;

  // Collect the results from the priority queue
  vector<DataPoint> result;
  while (!nearestNeighbors.empty()) {
    result.push_back({ nearestNeighbors.top().node->features, nearestNeighbors.top().node->label });
    nearestNeighbors.pop();
  }

  cout << "kNNSearch 4" << endl;

  return result;
}

vector<double> parseInputVector(const std::string& input) {
  istringstream iss(input);
  double feature;
  vector<double> point;
  char space;
  while (iss >> feature) {
    point.push_back(feature);
  }
  return point;
}

int main(int argc, char *argv[]) {
  int k, d = -1;
  string filename = "";
  int opt;
  vector<double> target;

  // Parse command-line arguments
  while ((opt = getopt(argc, argv, "hk:i:d:t:")) != -1) {
    switch (opt) {
      case 'h':
        cout << "Usage: " << argv[0] << " [-k value] [-i value]" << endl;
        cout << "Options:" << endl;
        cout << "  -k value       Number of neighbors to consider" << endl;
        cout << "  -i value       Input dataset" << endl;
        cout << "  -d value       Number of feature to consider in dataset" << endl;
        cout << "  -t value       Target point" << endl;
        return 0;
      case 'k':
        if (isPositiveInteger(optarg)) {
            k = stoi(optarg);
        } else {
            cout << "Invalid value for k, k = " << optarg << endl;
            return 0;
        }
        break;
      case 'i':
        filename = optarg;
        break;
      case 'd':
        if (isPositiveInteger(optarg)) {
            d = stoi(optarg);
        } else {
            cout << "Invalid value for d, d = " << optarg << endl;
            return 0;
        }
        break;
      case 't':
        target = parseInputVector(optarg);
        break;
      default:
        cout << "Usage: " << argv[0] << " -k <k_value> -i <i_value> -d <d_value>" << endl;
        return 0;
    }
  }

  if (k == -1 || d == -1 || filename == "" || target.size() == 0) {
    cout << "Not enough arguments provided." << endl;
    return 0;
  }

  KDTree kdTree;
  vector<DataPoint> data = kdTree.parseInput(filename, kdTree.dimensions);
  kdTree.buildKDTree(data, 0, d);

  KNN knn;
  vector<DataPoint> nearestNeighbors = knn.kNNSearch(kdTree, target, k);

  cout << "Nearest neighbor size: " << nearestNeighbors.size() << endl;

  return 0;
}
