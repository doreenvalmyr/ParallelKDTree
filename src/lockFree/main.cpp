#include <vector>
#include <unistd.h>
#include <limits>
#include "kdTree.h"
#include "../timing.h"
#include "../utils.h"
#include <atomic>
#include <thread>

using namespace std;

size_t dimension = numeric_limits<int>::max();

// Define a function for threads to execute
void threadInsertion(KDTree& tree, DataPoint dataPoint, int depth, int k) {
    tree.insertLockFree(dataPoint, depth, k);
}

int main(int argc, char *argv[]) {
  size_t k;
  string filename = "";
  int opt;
  const int numThreads = 5;  // Can adjust this number

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

  // Create and launch threads
    std::thread threads[numThreads];
    for (int i = 0; i < numThreads; ++i) {
        DataPoint dp;
        dp.label = i % 2;  // Just for example
        dp.features = std::vector<double>(k, 0); // Fill with k zeros (just for example)
        threads[i] = std::thread(threadInsertion, std::ref(myKDTree), dp, 0, k);
    }

    // Wait for all threads to complete
    for (int i = 0; i < numThreads; ++i) {
        threads[i].join();
    }

  printf("Total simulation time: %.6fs\n", totalSimulationTime);

  myKDTree.printKDTree(myKDTree.root.load());

  return 0;
}
