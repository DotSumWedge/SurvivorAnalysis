import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    
print(dataframes['tribe_colours.csv']['tribe_colour'])

dataframes['tribe_colours.csv']['tribe_status'] = dataframes['tribe_colours.csv']['tribe_status'].astype('category')

# assuming your DataFrame is named df
colors = dataframes['tribe_colours.csv']['tribe_colour']
n_colors = len(colors)

fig, ax = plt.subplots()
for i, color in enumerate(colors):
    ax.add_patch(plt.Rectangle((i/n_colors, 0), 1/n_colors, 1, color=color))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.show()