import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import math

# Get the path to the survivorData directory
data_dir = os.path.join(os.path.dirname(__file__), '..', 'survivorData')

# List of CSV file names
csv_files = [
    'tribe_colours.csv'
]

# Create a dictionary to store the DataFrames
dataframes = {}

# Loop through each CSV file and read its data into a DataFrame
for csv_file in csv_files:
    # Specify the relative path to the CSV file
    file_path = os.path.join(data_dir, csv_file)

    # Read the data from the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Store the DataFrame in the dictionary
    dataframes[csv_file] = df

# Convert tribe_status column to category type
dataframes['tribe_colours.csv']['tribe_status'] = dataframes['tribe_colours.csv']['tribe_status'].astype('category')

# Prepare the data for training
X = dataframes['tribe_colours.csv']['tribe_status'].values.reshape(-1, 1)
y = dataframes['tribe_colours.csv']['tribe_colour']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
logreg_model = LogisticRegression()

# Train the model
logreg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate Hamming distance for all predictions
hamming_distances = []

for idx in range(len(y_pred)):
    hamming_dist = sum(el1 != el2 for el1, el2 in zip(y_pred[idx], y_test.iloc[idx]))
    hamming_distances.append(hamming_dist)

# Calculate average Hamming distance for incorrect predictions
avg_hamming_distance = np.mean(hamming_distances)

# The Hamming distance measures the dissimilarity between two strings of equal length. In the context of hex color values, 
# each character represents a component of the color (e.g., red, green, and blue), and the Hamming distance calculates 
# the number of positions at which the predicted and actual color values differ.

# Since hex color values consist of six characters (e.g., #RRGGBB), the Hamming distance for hex color values can range from 0 to 6.

# Print average Hamming distance for incorrect predictions
print("Average Hamming Distance for Incorrect Predictions:", avg_hamming_distance)

# Calculate Euclidean distance for color similarity
euclidean_distances = []
for idx in range(len(y_pred)):
    predicted_color = y_pred[idx][1:]  # Remove the "#" character from the predicted color
    actual_color = y_test.iloc[idx][1:]  # Remove the "#" character from the actual color
    
    # Convert hex color values to RGB tuples
    predicted_rgb = tuple(int(predicted_color[i:i+2], 16) for i in (0, 2, 4))
    actual_rgb = tuple(int(actual_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Calculate Euclidean distance between RGB tuples
    distance = math.sqrt(sum((p - a) ** 2 for p, a in zip(predicted_rgb, actual_rgb)))
    euclidean_distances.append(distance)

# Calculate average Euclidean distance for all predictions
avg_euclidean_distance = np.mean(euclidean_distances)

# The Euclidean distance measures the spatial or geometric distance between two colors in the RGB color space.
# In the RGB color space, each color is represented by three components: red (R), green (G), and blue (B). 
# The Euclidean distance calculates the straight-line distance between two colors in this three-dimensional space.

# Print average Euclidean distance for all predictions
print("Average Euclidean Distance for All Predictions:", avg_euclidean_distance)


