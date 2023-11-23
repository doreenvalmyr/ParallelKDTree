import csv

# Read the dataset and store it in a list of lists
with open('first-dataset.csv', 'r') as file:
    data = [line.strip().split(',') for line in file]

# Remove the first column (index 0) from each row
for row in data:
    del row[0]

# Write the modified dataset to a new file
with open('modified_dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
