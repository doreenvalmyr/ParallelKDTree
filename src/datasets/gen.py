import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Number of data points
num_points = 2000000

# Generate random data
data = []
for _ in range(num_points):
    # Generate random values for each feature
    features = np.random.randint(1, 11, size=10)
    
    # Generate a random class label
    label = np.random.randint(1, 6)
    
    # Combine features and label into a single data point
    data_point = np.concatenate((features, [label]))
    
    # Convert data point to a string and append to the dataset
    data.append(','.join(map(str, data_point)))

# Save the generated data to a file
with open('very-large-dataset.csv', 'w') as file:
    file.write('\n'.join(data))

print("Generated data and saved to 'very-large-dataset.csv'")
