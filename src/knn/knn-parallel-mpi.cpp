#include <iostream>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <stack>
#include <omp.h>
#include "mpi.h"
#include "knn.h"
#include "../kdTree/kdTree.h"
#include "../utils.h"
#include "../timing.h"

using namespace std;

void insertAndSortNeighbors2(vector<DistanceNode2>& nearestNeighbors, DistanceNode2& neighbor, size_t k) {
  // Use lower_bound to find the position to insert the new element
  auto it = std::lower_bound(nearestNeighbors.begin(),
                            nearestNeighbors.end(),
                            neighbor,
                            [](const DistanceNode2& a, const DistanceNode2& b) {
    return a.distance < b.distance;
  });

  nearestNeighbors.insert(it, neighbor);

  // Remove the last element if there are more than k neighbors
  if (nearestNeighbors.size() > k) {
    nearestNeighbors.pop_back();
  }
}

// Function to distribute data among MPI processes
std::vector<DataPoint> distributeData(int rank, int size, const std::vector<DataPoint>& allData) {
  int dataSize = allData.size();
  int localSize = dataSize / size;

  // Calculate the starting index and local size for each process
  int startIndex = rank * localSize;
  int endIndex = startIndex + localSize;
  if (rank == size - 1) {
    endIndex = dataSize;
  }

  // Allocate memory for the local data
  std::vector<DataPoint> localData(allData.begin() + startIndex, allData.begin() + endIndex);
  
  return localData;
}

void kNNSearchMPI(const KDNode* root, const vector<double>& target, size_t k,
                                 vector<DistanceNode2>& neighbors) {
  if (root == nullptr) {
    return;
  }

  size_t targetSize = target.size();

  stack<pair<const KDNode*, int>> nodeStack;
  nodeStack.push({root, 0});

  while (!nodeStack.empty()) {
    const KDNode* currentNode = nodeStack.top().first;
    int depth = nodeStack.top().second;
    nodeStack.pop();

    if (currentNode == nullptr) {
      continue;
    }

    int axis = depth % targetSize;

    double distance;
    try {
      distance = calculateDistance(target, currentNode->features);
    } catch (const invalid_argument& error) {
      cerr << "Exception caught: " << error.what() << endl;
      continue;
    }

    DistanceNode2 neighbor = {distance, currentNode->label};
    insertAndSortNeighbors2(neighbors, neighbor, k);

    if (target[axis] < currentNode->features[axis]) {
      nodeStack.push({currentNode->right.get(), depth + 1});
      nodeStack.push({currentNode->left.get(), depth + 1});
    } else {
      nodeStack.push({currentNode->left.get(), depth + 1});
      nodeStack.push({currentNode->right.get(), depth + 1});
    }
  }
}

// Find k nearest neighbors of target point (parallel implementation)
void KNN::kNNSearchParallelMPI(const vector<DataPoint>& data, const vector<double>& target, int k, int rank, int size) {
  // Distribute data among processes
  vector<DataPoint> localData = distributeData(rank, size, data);

  // Build local KDTree
  KDTree localKDTree;
  localKDTree.buildKDTree(localData, 0, data[0].features.size());
  
  vector<DistanceNode2> nearestNeighborsVector;
  kNNSearchMPI(localKDTree.root.get(), target, static_cast<size_t>(k), nearestNeighborsVector);

  MPI_Datatype MPI_DISTANCENODE;
  MPI_Type_contiguous(sizeof(DistanceNode2), MPI_BYTE, &MPI_DISTANCENODE);
  MPI_Type_commit(&MPI_DISTANCENODE);

  int sizes[size] = {0};
  int sizeToSend = nearestNeighborsVector.size();
  MPI_Gather(&sizeToSend, 1, MPI_INT, &sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Compute displacements for all the particles (prefix sum)
  int displacements[size];
  int allNeighborsSum = 0;
  for (int i = 0; i < size; i++) {
    displacements[i] = allNeighborsSum;
    allNeighborsSum += sizes[i];
  }

  // Gather nearest neighbors from all processes
  vector<DistanceNode2> allNearestNeighbors(allNeighborsSum);
  MPI_Gatherv(nearestNeighborsVector.data(), sizeToSend, MPI_DISTANCENODE,
             allNearestNeighbors.data(), sizes, displacements,
             MPI_DISTANCENODE, 0, MPI_COMM_WORLD);

  // On process rank 0, combine the results
  if (rank == 0) {
    std::sort(allNearestNeighbors.begin(), allNearestNeighbors.end(),
          [](const DistanceNode2& a, const DistanceNode2& b) {
              return a.distance > b.distance;
          });

    vector<DataPoint> nearestNeighborsLocal;
    while (!allNearestNeighbors.empty()) {
      DistanceNode2 distanceNode = allNearestNeighbors.back();
      vector<double> emptyVector;
      DataPoint datapoint = {emptyVector, distanceNode.label};
      nearestNeighbors.push_back(datapoint);
      if (nearestNeighbors.size() >= static_cast<size_t>(k)) break;
      allNearestNeighbors.pop_back();
    }
  }
}

int main(int argc, char *argv[]) {
  int k, d = -1;
  string filename = "";
  int opt;
  vector<double> target;

  MPI_Init(&argc, &argv);
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

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

  // Run sequential knn search
  Timer sequentialTimer;
  KNN seqKnn;
  kdTree.buildKDTree(data, 0, d);
  seqKnn.kNNSearch(kdTree, target, k);
  double sequentialTime = sequentialTimer.elapsed();

  // Run parallel knn search
  Timer parallelTimer;
  KNN parallelKnn;
  parallelKnn.kNNSearchParallelMPI(data, target, k, rank, nproc);
  double parallelTime = parallelTimer.elapsed();

  if (rank == 0) {
    printf("\nSequential KNN");
    seqKnn.findTargetLabel();

    printf("\nParallel KNN");
    parallelKnn.findTargetLabel();

    printf("\nTotal simulation time for KNN sequential search: %.6fs", sequentialTime);
    printf("\nTotal simulation time for KNN parallel search: %.6fs", parallelTime);
    printf("\nSpeedup: %.6f\n", sequentialTime/parallelTime);

    std::ofstream results_file;
    results_file.open("knn_timings.txt", std::ios_base::app); // Appending to the file

    results_file << "Sequential: " << " Time: " << sequentialTime << " seconds" << endl;
    results_file << "Parallel: " << "Processes: " << nproc << ", Time: " << parallelTime << " seconds" << endl;
    results_file << "Speedup: " << (sequentialTime/ parallelTime) << "\n" << endl;
  }

  MPI_Finalize();

  return 0;
}
