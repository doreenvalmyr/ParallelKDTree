#include <iostream>
#include <unistd.h>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <stack>
#include "knn.h"
#include "../kdTree/kdTree.h"
#include "../utils.h"
#include "../timing.h"

using namespace std;

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

  Timer totalSimulationTimer;
  KNN knn;
  knn.kNNSearch(kdTree, target, k);
  double totalSimulationTime = totalSimulationTimer.elapsed();

  knn.printNearestNeighbors();
  
  knn.findTargetLabel();

  printf("\nTotal simulation time for KNN search: %.6fs\n", totalSimulationTime);

  return 0;
}
