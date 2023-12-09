#include <iostream>
#include <vector>
#include <algorithm>
#include "kdTree.h"

using namespace std;

atomic<KDNode*> buildKDTreeImpl(vector<DataPoint>& data, int depth, int k) {
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

  // unique_ptr<KDNode> node = make_unique<KDNode>();
  KDNode* node = new KDNode(); // Use raw pointer instead of unique_ptr
  node->features = data[median].features;
  node->label = data[median].label;

  // Construct subtrees
  vector<DataPoint> leftData(data.begin(), data.begin() + median);
  node->left = buildKDTreeImpl(leftData, depth + 1, k).load();
  vector<DataPoint> rightData(data.begin() + median + 1, data.end());
  node->right = buildKDTreeImpl(rightData, depth + 1, k).load();

  return node;
}

// Function to build a KD-tree
void KDTree::buildKDTree(vector<DataPoint>& data, int depth, int k) {
  root.store(buildKDTreeImpl(data, depth, k));
  dimensions = k;
}
