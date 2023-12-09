#ifndef KDTREE_H
#define KDTREE_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstddef>
#include <vector>
#include <cmath>
#include <iostream>
#include <memory>
#include <atomic>


using namespace std;

// Define a struct to represent a data point
struct DataPoint {
  vector<double> features;
  int label;
  int threadId;
};

class KDNode
{
public:
  vector<double> features;
  int label;
  atomic<KDNode*> left;
  atomic<KDNode*> right;
  
  // Constructor to initialize KDNode with features
  KDNode() : label(0), left(nullptr), right(nullptr) {}
};

class KDTree {
public:
  atomic<KDNode*> root;
  size_t dimensions; // To store the dimensionality of the data

  // Constructor
  KDTree() : root(nullptr), dimensions(0) {}

  void buildKDTree(vector<DataPoint>& data, int depth, int k);

  // Function to parse input file into a vector of strings separated by newlines
  vector<DataPoint> parseInput(const string& filename, size_t &dimension) {
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
          if (isspace(iss.peek()) || iss.peek() == '\n' || 
                iss.peek() == '\r' || iss.peek() == EOF) {
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

  void insertLockFree(const DataPoint& dataPoint, int depth, int k) {
    KDNode* node = new KDNode();
    node->features = dataPoint.features;
    node->label = dataPoint.label;
    insertRecursiveLockFree(root, node, depth, k);
  }

  void printKDTree(KDNode* node) {
    if (node == nullptr) {
      return;
    }

    // Traverse left subtree
    printKDTree(node->left.load());

    // Print information for the current node
    cout << "Features: ";
    for (double feature : node->features) {
      cout << feature << " ";
    }
    cout << "| Label: " << node->label << endl;

    // Traverse right subtree
    printKDTree(node->right.load());
  }
  
private:
  void insertRecursiveLockFree(atomic<KDNode*>& current, KDNode* node, int depth, int k) {
    KDNode* expected = nullptr;
    if (!current.load() && current.compare_exchange_weak(expected, node)) {
      return;
    }

    // Determine dimension for comparison
    int dim = depth % k;
    if (node->features[dim] < current.load()->features[dim]) {
      insertRecursiveLockFree(current.load()->left, node, depth + 1, k);
    } else {
      insertRecursiveLockFree(current.load()->right, node, depth + 1, k);
    }
  }
};

#endif
