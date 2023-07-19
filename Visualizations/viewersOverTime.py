import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the path to the survivorData directory
data_dir = os.path.join(os.path.dirname(__file__), '..', 'survivorData')

# List of CSV file names
csv_files = [
    'viewers.csv'
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

# Create a line plot of viewers over time
fig, ax1 = plt.subplots()
sns.lineplot(data=dataframes['viewers.csv'], x='episode_date', y='viewers', ax=ax1)

# Create a secondary y-axis on the right side of the chart
ax2 = ax1.twinx()

# Create a line plot of imdb_rating over time on the secondary y-axis
sns.lineplot(data=dataframes['viewers.csv'], x='episode_date', y='imdb_rating', ax=ax2, color='g')

# Set the y-axis labels
ax1.set_ylabel('Viewers')
ax2.set_ylabel('IMDB Rating')

# Set the title of the chart
plt.title('Viewers and IMDB rating for each episode over time')

plt.show()