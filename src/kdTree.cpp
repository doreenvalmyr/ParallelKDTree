#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "kdTree.h"
#include <cmath>
#include <limits>

using namespace std;

size_t dimension = std::numeric_limits<int>::max();

// Define a struct to represent a data point
struct DataPoint {
  vector<double> features;
  int label;
};

// Function to check if a string is a positive integer
bool isPositiveInteger(const string& s) {
  for (char c : s) {
    if (!isdigit(c)) {
      return false;
    }
  }
  return true;
}

// Function to parse input file into a vector of strings separated by newlines
vector<DataPoint> parseInput(const string& filename) {
  vector<DataPoint> dataPoints;
  string line;
  ifstream file(filename);
  
  if (file.is_open()) {
    while (getline(file, line)) {
      istringstream iss(line);
      DataPoint dataPoint;
      double content;
      char comma;  // To read and discard the comma separator
      
      // Read features and label into the vector
      while (iss >> content) {
        // Check for the end of the line
        if (iss.peek() == '\n' || iss.peek() == EOF) {
          // Read the label
          dataPoint.label = (int)content;
          break;
        }

        dataPoint.features.push_back(content);
        iss >> comma;
      }

      dimension = fminf(dimension, dataPoint.features.size());

      // Add data point
      dataPoints.push_back(dataPoint);
    }
    file.close();
  } else {
    cout << "Unable to open file " << filename << endl;
  }

  cout << "Parsed " << dataPoints.size() << " data points from " << filename << endl;
  return dataPoints;
}

// Function to build a KD-tree
KDNode* buildKDTree(vector<DataPoint>& data, int depth, int k) {
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

  KDNode* node = new KDNode();
  node->features = data[median].features;
  node->label = data[median].label;

  // Construct subtrees
  vector<DataPoint> leftData(data.begin(), data.begin() + median);
  node->left = buildKDTree(leftData, depth + 1, k);
  vector<DataPoint> rightData(data.begin() + median + 1, data.end());
  node->right = buildKDTree(rightData, depth + 1, k);

  return node;
}

// Print KD-tree in-order
void printKDTree(KDNode* root) {
  if (root == nullptr) {
    return;
  }

  // Traverse left subtree
  printKDTree(root->left);

  // Print information for the current node
  std::cout << "Features: ";
  for (double feature : root->features) {
    std::cout << feature << " ";
  }
  std::cout << "| Label: " << root->label << std::endl;

  // Traverse right subtree
  printKDTree(root->right);
}

int main(int argc, char *argv[]) {
  int k;
  string filename = "";
  int opt;

  while ((opt = getopt(argc, argv, "hk:i:")) != -1) {
    switch (opt) {
      case 'h':
        cout << "Usage: " << argv[0] << " [-k value] [-i value]" << endl;
        cout << "Options:" << endl;
        cout << "  -k value       Number of dimension" << std::endl;
        cout << "  -i value       Input dataset" << std::endl;
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
      default:
        cout << "Usage: ./kdTree -k <number of dimensions>" << endl;
        return 0;
    }
  }

  // Open the file and parse input into a vector of strings separated by newlines
  vector<DataPoint> input = parseInput(filename);
    
  if (k > dimension) {
      cout << "Value given for k is greater than the number of features in the data set" << endl;
      cout << "     Setting k to the number of dimensions in the data set" << endl;
      k = dimension;
  }

  // Use the input vector to build the kd-tree
  KDNode *root = buildKDTree(input, 0, k);

  printKDTree(root);

  return 0;
}
