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

    # Print the head of the DataFrame
    print(f'{csv_file}:')
    print(df.describe())

# Merge challenge_results and castaway_details dataframes on castaway_id
merged_df = pd.merge(dataframes['challenge_results.csv'], dataframes['castaway_details.csv'], on='castaway_id')

# Group data by challenge type and gender and count number of challenges won by each group
grouped_df = merged_df.groupby(['challenge_type', 'gender'])['challenge_id'].count().unstack()

# Create a stacked bar chart
grouped_df.plot.bar(stacked=True)

# Get the challenge types from challenge_description.csv
challenge_types = dataframes['challenge_description.csv'].columns[2:]

# Set the xtick labels to the challenge types
plt.xticks(range(len(challenge_types)), challenge_types, rotation=90)

# Add labels and title
plt.xlabel('Challenge Type')
plt.ylabel('Number of Challenges Won')
plt.title('Number of Challenges Won by Gender')

# Show the plot
plt.show()