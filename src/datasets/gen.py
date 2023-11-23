import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Number of data points
num_points = 500000

# Generate random data
data = []
for _ in range(num_points):
    # Generate random values for each feature
    features = np.random.randint(1, 11, size=9)
    
    # Generate a random class label (1 or 2)
    label = np.random.randint(1, 3)
    
    # Combine features and label into a single data point
    data_point = np.concatenate((features, [label]))
    
    # Convert data point to a string and append to the dataset
    data.append(','.join(map(str, data_point)))

# Save the generated data to a file
with open('large-dataset.csv', 'w') as file:
    file.write('\n'.join(data))

print("Generated data and saved to 'large-dataset.csv'")
