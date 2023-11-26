#include <vector>
#include <unistd.h>
#include <limits>
#include "kdTree.h"
#include "../timing.h"
#include "../utils.h"

using namespace std;

size_t dimension = numeric_limits<int>::max();

int main(int argc, char *argv[]) {
  int k;
  string filename = "";
  int opt;

  while ((opt = getopt(argc, argv, "hk:i:")) != -1) {
    switch (opt) {
      case 'h':
        cout << "Usage: " << argv[0] << " [-k value] [-i value]" << endl;
        cout << "Options:" << endl;
        cout << "  -k value       Number of dimension" << endl;
        cout << "  -i value       Input dataset" << endl;
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

  KDTree myKDTree;

  // Open the file and parse input into a vector of strings separated by newlines
  vector<DataPoint> input = myKDTree.parseInput(filename, dimension);
    
  if (k > dimension) {
      cout << "Value given for k is greater than the number of features in the data set" << endl;
      cout << "     Setting k to the number of dimensions in the data set" << endl;
      cout << "Dimensions are " << dimension << endl;
      k = dimension;
  }
  
  // Use the input vector to build the kd-tree
  Timer totalSimulationTimer;
  myKDTree.buildKDTree(input, 0, k);
  double totalSimulationTime = totalSimulationTimer.elapsed();

  printf("Total simulation time: %.6fs\n", totalSimulationTime);

  return 0;
}
