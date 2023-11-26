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

using namespace std;

// Define a struct to represent a data point
struct DataPoint {
  std::vector<double> features;
  int label;
  int threadId;
};

class KDNode
{
public:
  std::vector<double> features;
  int label;
  std::unique_ptr<KDNode> left;
  std::unique_ptr<KDNode> right;
  
  // Constructor to initialize KDNode with features
  KDNode() : left(nullptr), right(nullptr), label(0) {}
};

class KDTree {
public:
  std::unique_ptr<KDNode> root;
  size_t dimensions; // To store the dimensionality of the data

  // Constructor
  KDTree() : root(nullptr), dimensions(0) {}

  void buildKDTree(std::vector<DataPoint>& data, int depth, int k);

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
  
private:
  // Print KD-tree in-order
  void printKDTree(unique_ptr<KDNode>& root) {
    if (root == nullptr) {
      return;
    }

    // Traverse left subtree
    printKDTree(root->left);

    // Print information for the current node
    cout << "Features: ";
    for (double feature : root->features) {
      cout << feature << " ";
    }
    cout << "| Label: " << root->label << endl;

    // Traverse right subtree
    printKDTree(root->right);
  }
};

#endif
