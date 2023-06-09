import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the path to the survivorData directory
data_dir = os.path.join(os.path.dirname(__file__), '..', 'survivorData')

# List of CSV file names
csv_files = [
    'challenge_description.csv',
    'challenge_results.csv',
    'tribe_mapping.csv',
    'castaway_details.csv'
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

# Count the number of TRUE values in each column from the 3rd column onward
true_counts = (dataframes['challenge_description.csv'].iloc[:, 2:] == True).sum()

# Create a bar chart showing the number of TRUE values in each column
true_counts.plot(kind='bar')

print(true_counts)

# Add labels and title
plt.xlabel('Column')
plt.ylabel('Number of challanges')
plt.title('Number of each type of challange')

# Rotate x-axis labels by 90 degrees
plt.xticks(rotation=90)

# Show the plot
plt.show()