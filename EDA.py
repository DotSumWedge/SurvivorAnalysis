import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import math
import os

# Get the path to the survivorData directory
data_dir = os.path.join(os.path.dirname(__file__), '..', 'survivorData')

# List of CSV file names
csv_files = [
    'advantage_movement.csv',
    'boot_mapping.csv',
    'castaways.csv',
    'castaway_details.csv',
    'challenge_description.csv',
    'challenge_results.csv',
    'confessionals.csv',
    'jury_votes.csv',
    'screen_time.csv',
    'season_palettes.csv',
    'season_summary.csv',
    'survivor_auction.csv',
    'tribe_colours.csv',
    'tribe_mapping.csv',
    'viewers.csv',
    'vote_history.csv'
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
    # print(f'{csv_file}:')
    # print(df.head())

# Loop through each DataFrame in the dataframes dictionary
for csv_file, df in dataframes.items():
    # Print the name of the CSV file
    print(csv_file)
    
    # Show the number of rows in the DataFrame
    print('Number of rows:', len(df))
    
    # Show the number of unique values and data type for each column of the DataFrame
    print(pd.concat([df.nunique(), df.dtypes], axis=1))
    
    # Print a separator
    print('-' * 40)

# Create a violin plot of the age column from the castaways.csv data
sns.violinplot(x=dataframes['castaways.csv']['age'])
plt.show()

print("Ben has a big brain")