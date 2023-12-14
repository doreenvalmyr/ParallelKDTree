#include <iostream>
#include <unistd.h>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <stack>
#include <omp.h>
#include "knn.h"
#include "../kdTree/kdTree.h"
#include "../utils.h"
#include "../timing.h"

using namespace std;

void kNNSearchIterativeParallel(const KDNode* root, const vector<double>& target, size_t k,
                                 vector<DistanceNode>& nearestNeighbors) {
  if (root == nullptr) {
    return;
  }

  size_t targetSize = target.size();

  stack<pair<const KDNode*, int>> nodeStack;
  nodeStack.push({root, 0});

  #pragma omp parallel shared(nearestNeighbors)
  {
    #pragma omp single nowait
    while (!nodeStack.empty()) {
      const KDNode* currentNode;
      int depth;

      #pragma omp critical
      {
        currentNode = nodeStack.top().first;
        depth = nodeStack.top().second;
        nodeStack.pop();
      }

      int axis = depth % targetSize;

      double distance;
      try {
        distance = calculateDistance(target, currentNode->features);
      } catch (const invalid_argument& error) {
        cerr << "Exception caught: " << error.what() << endl;
        continue;
      }

      DistanceNode neighbor = {distance, currentNode};
      #pragma omp critical
      insertAndSortNeighbors(nearestNeighbors, neighbor, k);
        
      #pragma omp critical
      {
        if (target[axis] < currentNode->features[axis]) {
          auto right = currentNode->right.get();
          auto left = currentNode->left.get();
          if (right != nullptr) nodeStack.push({right, depth + 1});
          if (left != nullptr) nodeStack.push({left, depth + 1});
        } else {
          auto right = currentNode->right.get();
          auto left = currentNode->left.get();
          if (left != nullptr) nodeStack.push({left, depth + 1});
          if (right != nullptr) nodeStack.push({right, depth + 1});
        }
      }
    }
  }
}

// Find k nearest neighbors of target point (parallel implementation)
void KNN::kNNSearchParallelOpenMP(const KDTree& kdTree, const vector<double>& target, int k) {
  vector<DistanceNode> nearestNeighborsVector;

  kNNSearchIterativeParallel(kdTree.root.get(), target, static_cast<size_t>(k), nearestNeighborsVector);

  // Collect the results from the priority queue
  while (!nearestNeighborsVector.empty()) {
    DistanceNode point = nearestNeighborsVector.back();
    nearestNeighbors.push_back({point.node->features, point.node->label});
    nearestNeighborsVector.pop_back();
  }
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

  // Run sequential knn search
  Timer sequentialTimer;
  KNN seqKnn;
  seqKnn.kNNSearch(kdTree, target, k);
  double sequentialTime = sequentialTimer.elapsed();

  // Run parallel knn search
  Timer parallelTimer;
  KNN parallelKnn;
  parallelKnn.kNNSearchParallelOpenMP(kdTree, target, k);
  double parallelTime = parallelTimer.elapsed();

  parallelKnn.printNearestNeighbors();

  parallelKnn.findTargetLabel();

  printf("\nTotal simulation time for KNN sequential search: %.6fs\n", sequentialTime);
  printf("\nTotal simulation time for KNN parallel search: %.6fs\n", parallelTime);
  printf("\nSpeedup: %.6f\n", sequentialTime/parallelTime);

  return 0;
}
