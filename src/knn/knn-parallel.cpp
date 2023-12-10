#include <iostream>
#include <unistd.h>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include "knn.h"
#include "../kdTree/kdTree.h"
#include "../utils.h"
#include "../timing.h"

using namespace std;

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

// Search KD Tree recursively to determine k nearest neighbors
void kNNSearchRecursive(const KDNode* currentNode, const vector<double>& target, size_t k,
                        vector<DistanceNode>& nearestNeighbors, int depth) {
  
  if (currentNode == nullptr) {
    return;
  }

  int axis = depth % target.size();

  // Calculate distance of target point to current node
  double distance;
  try {
    distance = calculateDistance(target, currentNode->features);
  } catch (const invalid_argument& error) {
    cerr << "Exception caught: " << error.what() << endl;
    return;
  }

  // Insert the new DistanceNode into the vector in ascending order
  DistanceNode neighbor = {distance, currentNode};
  insertAndSortNeighbors(nearestNeighbors, neighbor, k);

  // Recursively search the side of the splitting plane that contains the target point
  if (target[axis] < currentNode->features[axis]) {
    kNNSearchRecursive(currentNode->left.get(), target, k, nearestNeighbors, depth + 1);
  } else {
    kNNSearchRecursive(currentNode->right.get(), target, k, nearestNeighbors, depth + 1);
  }

  // Check if we need to search the other side (if there could be closer points)
  if (nearestNeighbors.size() < k || abs(target[axis] - currentNode->features[axis]) < nearestNeighbors.back().distance) {
    if (target[axis] < currentNode->features[axis]) {
      kNNSearchRecursive(currentNode->right.get(), target, k, nearestNeighbors, depth + 1);
    } else {
      kNNSearchRecursive(currentNode->left.get(), target, k, nearestNeighbors, depth + 1);
    }
  }
}

// Find k nearest neighbors of target point
void KNN::kNNSearch(const KDTree& kdTree, const vector<double>& target, int k) {
  vector<DistanceNode> nearestNeighborsVector;

  // Add k nearest neighbors to result using kdTree
  kNNSearchRecursive(kdTree.root.get(), target, (size_t)k, nearestNeighborsVector, 0);

  // Collect the results from the priority queue
  while (!nearestNeighborsVector.empty()) {
    DistanceNode point = nearestNeighborsVector.back();
    nearestNeighbors.push_back({ point.node->features, point.node->label });
    nearestNeighborsVector.pop_back();
  }
}

void kNNSearchRecursiveParallel(const KDNode* currentNode, const vector<double>& target, size_t k,
                        vector<DistanceNode>& nearestNeighbors, int depth) {
  
  if (currentNode == nullptr) {
    return;
  }

  int axis = depth % target.size();

  // Calculate distance of target point to current node
  double distance;
  try {
    distance = calculateDistance(target, currentNode->features);
  } catch (const invalid_argument& error) {
    cerr << "Exception caught: " << error.what() << endl;
    return;
  }

  DistanceNode neighbor = {distance, currentNode};

  // Only one thread should moodify the nearestNeighbors vector at a time
  #pragma omp critical
  {
    // Insert the new DistanceNode into the vector in ascending order
    insertAndSortNeighbors(nearestNeighbors, neighbor, k);
  }

  // Parallelize the recursive calls
  #pragma omp parallel
  {
    #pragma omp single nowait
    {
      // Recursive call for the branch containing the target point
      if (target[axis] < currentNode->features[axis]) {
        #pragma omp task
        kNNSearchRecursiveParallel(currentNode->left.get(), target, k, nearestNeighbors, depth + 1);
      } else {
        #pragma omp task
        kNNSearchRecursiveParallel(currentNode->right.get(), target, k, nearestNeighbors, depth + 1);
      }

      // Recursive call for the other branch if necessary
      if (nearestNeighbors.size() < k || abs(target[axis] - currentNode->features[axis]) < nearestNeighbors.back().distance) {
        if (target[axis] < currentNode->features[axis]) {
          #pragma omp task
          kNNSearchRecursiveParallel(currentNode->right.get(), target, k, nearestNeighbors, depth + 1);
        } else {
          #pragma omp task
          kNNSearchRecursiveParallel(currentNode->left.get(), target, k, nearestNeighbors, depth + 1);
        }
      }
    }
  }
}

// Find k nearest neighbors of target point
void KNN::kNNSearchParallel(const KDTree& kdTree, const vector<double>& target, int k) {
  vector<DistanceNode> nearestNeighborsVector;

  // Add k nearest neighbors to result using kdTree
  kNNSearchRecursiveParallel(kdTree.root.get(), target, (size_t)k, nearestNeighborsVector, 0);

  // Collect the results from the priority queue
  while (!nearestNeighborsVector.empty()) {
    DistanceNode point = nearestNeighborsVector.back();
    nearestNeighbors.push_back({ point.node->features, point.node->label });
    nearestNeighborsVector.pop_back();
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

  Timer sequentialTimer;
  KNN seqKnn;
  seqKnn.kNNSearch(kdTree, target, k);
  double sequentialTime = sequentialTimer.elapsed();

  Timer parallelTimer;
  KNN parallelKnn;
  parallelKnn.kNNSearchParallel(kdTree, target, k);
  double parallelTime = parallelTimer.elapsed();

  parallelKnn.printNearestNeighbors();

  parallelKnn.findTargetLabel();

  printf("\nTotal simulation time for KNN sequential search: %.6fs\n", sequentialTime);
  printf("\nTotal simulation time for KNN parallel search: %.6fs\n", parallelTime);
  printf("\nSpeedup: %.6fs\n", sequentialTime/parallelTime);

  return 0;
}
