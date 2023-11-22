#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "kdTree.h"
using namespace std;
#define DIMENSIONS 6
// Define a struct to represent a data point
struct DataPoint {
    double feature1;
    double feature2;
    double feature3;
    double feature4;
    double feature5;
    double feature6;
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
            char comma;  // To read and discard the comma separator
            iss >> dataPoint.feature1 >> comma
                >> dataPoint.feature2 >> comma
                >> dataPoint.feature3 >> comma
                >> dataPoint.feature4 >> comma
                >> dataPoint.feature5 >> comma
                >> dataPoint.feature6 >> comma
                >> dataPoint.label;

            // Optional: Print the parsed data for verification
            // cout << "Parsed: " << dataPoint.feature1 << ", " << dataPoint.feature2 << ", " << dataPoint.label << endl;

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
kdNode<DIMENSIONS>* buildKDTree(vector<DataPoint>& data, int depth, int k) {
    if (data.empty()) {
        return nullptr;
    }

    // Choose axis based on depth for balanced tree construction
    int axis = depth % k;

    // Sort and choose median as pivot element
    int median = data.size() / 2;
    nth_element(data.begin(), data.begin() + median, data.end(),
                [axis](const DataPoint& a, const DataPoint& b) {
                    return a.feature1 < b.feature1; // Change the axis as needed
                });

    // Create node and construct subtrees
    kdNode<DIMENSIONS>* node = new kdNode<DIMENSIONS>();
    node->coordinates[0] = data[median].feature1;
    node->coordinates[1] = data[median].feature2;
    node->coordinates[2] = data[median].feature3;
    node->coordinates[3] = data[median].feature4;
    node->coordinates[4] = data[median].feature5;
    node->coordinates[5] = data[median].feature6;
    node->label = data[median].label;
    node->left = buildKDTree(vector<DataPoint>(data.begin(), data.begin() + median), depth + 1);
    node->right = buildKDTree(vector<DataPoint>(data.begin() + median + 1, data.end()), depth + 1);

    return node;
}

// Function to print KD-tree for debugging
void printKDTree(kdNode<DIMENSIONS>* root) {
    if (root != nullptr) {
        cout << "Feature1: " << root->coordinates[0] << ", Feature2: " << root->coordinates[1] << ", ..., Label: " << root->coordinates[DIMENSIONS - 1] << endl;
        printKDTree(root->left);
        printKDTree(root->right);
    }
}

int main(int argc, char *argv[]) {
    int k = -1;
    string filename = "";

    int opt;
    while ((opt = getopt(argc, argv, "hk:i:")) != -1) {
        switch (opt) {
            case 'h':
                cout << "Usage: ./kdTree -k <number of dimensions> -i <input file>" << endl;
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

    cout << "Filename: " << filename << endl;
    cout << "k: " << k << "\n" << endl;

    // Open the file and parse input into a vector of strings separated by newlines
    vector<DataPoint> input = parseInput(filename);
    if (k > DIMENSIONS) {
        cout << "k is greater than the number of dimensions in the data set" << endl;
        cout << "setting k to the number of dimensions in the data set" << endl;
        k = DIMENSIONS;
    }

    // Now we can use the input vector to build the kd-tree

    kdNode<DIMENSIONS> *root = buildKDTree(input, 0, k);

    printKDTree(root);



    

    return 0;
}
