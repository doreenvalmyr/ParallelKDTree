#include <iostream>
#include <vector>
#include <algorithm>
#include "kdTree.h"
#include <omp.h>

#define MAX_PARALLEL_DEPTH 3
using namespace std;

// Function to build a KD-tree using OpenMP
unique_ptr<KDNode> buildKDTreeImpl(vector<DataPoint>& data, int depth, int k) {

  if (data.empty()) {
    return nullptr;
  }

  // Choose axis based on depth for balanced tree construction
  int axis = depth % k;

  // Sort and choose median as pivot element along axis
  int median = data.size() / 2;
  nth_element(data.begin(), data.begin() + median, data.end(),
              [axis](const DataPoint& a, const DataPoint& b) {
                  return a.features[axis] < b.features[axis];
              });

  // Create the root node
  unique_ptr<KDNode> node = make_unique<KDNode>();
  node->features = data[median].features;
  node->label = data[median].label;

 if (depth < MAX_PARALLEL_DEPTH) {
    #pragma omp parallel
    {
      #pragma omp single
      {
        #pragma omp task
        {
          // Build left subtree
          vector<DataPoint> leftData(data.begin(), data.begin() + median);
          node->left = buildKDTreeImpl(leftData, depth + 1, k);
        }

        #pragma omp task
        {
          // Build right subtree
          vector<DataPoint> rightData(data.begin() + median + 1, data.end());
          node->right = buildKDTreeImpl(rightData, depth + 1, k);
        }
      }
    }
    #pragma omp taskwait // Wait for tasks to complete
  } else {
    // Non-parallel execution for deeper levels
    vector<DataPoint> leftData(data.begin(), data.begin() + median);
    node->left = buildKDTreeImpl(leftData, depth + 1, k);

    vector<DataPoint> rightData(data.begin() + median + 1, data.end());
    node->right = buildKDTreeImpl(rightData, depth + 1, k);
  }
 
 // #pragma omp taskwait // Wait for tasks to complete
  return node;
}

// Function to build a KD-tree
void KDTree::buildKDTree(vector<DataPoint>& data, int depth, int k) {
  root = buildKDTreeImpl(data, depth, k);
  dimensions = k;
}
