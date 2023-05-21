import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the path to the survivorData directory
data_dir = os.path.join(os.path.dirname(__file__), '..', 'survivorData')

# List of CSV file names
csv_files = [
    'season_palettes.csv'
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
    
# assuming your DataFrame is named df
colors = dataframes['season_palettes.csv']['palette']
colors = colors.dropna() # Remove any nan values from the colors variable

# Get the unique season values and sort them in ascending order
seasons = dataframes['season_palettes.csv']['season'].unique()
seasons.sort()

# Set the spacing between different season values
spacing = 0.9

fig, ax = plt.subplots()
for season in seasons:
    # Get the colors for the current season
    season_colors = dataframes['season_palettes.csv'][dataframes['season_palettes.csv']['season'] == season]['palette']
    season_colors = season_colors.dropna() # Remove any nan values from the season_colors variable
    n_colors = len(season_colors)
    
    for i, color in enumerate(season_colors):
        ax.add_patch(plt.Rectangle((i/n_colors + (season-1)*(1+spacing), 0), 1/n_colors, 1, color=color))
ax.set_xlim(0, len(seasons)*(1+spacing)-spacing)
ax.set_ylim(0, 1)
ax.axis('off')
plt.show()