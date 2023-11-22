#include <cstddef>  // For size_t
#include <vector>
#include <cmath>
#include <iostream>

template <size_t Dimensions>
// kdNode class
class kdNode
{
public:
    kdNode* left;
    kdNode* right;
    int label;
    // Additional member variables for each dimension
    double coordinates[Dimensions];
    
    // Constructor to initialize coordinates
    kdNode() {
        for (size_t i = 0; i < Dimensions; ++i) {
            coordinates[i] = 0.0;  // You can initialize to your desired default value
        }
    }
};



