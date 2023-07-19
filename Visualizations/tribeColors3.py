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

colors = dataframes['tribe_colours.csv']['tribe_colour']
n_colors = len(colors)

# Create a crosstab to count the number of occurrences of each color within each tribe_status category
ct = pd.crosstab(dataframes['tribe_colours.csv']['tribe_status'], dataframes['tribe_colours.csv']['tribe_colour'])

# Plot the crosstab as a stacked bar chart
ax = ct.plot(kind='bar', stacked=True, color=sns.color_palette(dataframes['tribe_colours.csv']['tribe_colour'].unique()))

# Hide the legend
ax.legend().set_visible(False)

# Adjust the bottom margin of the plot
plt.gcf().subplots_adjust(bottom=0.2)

# Show the plot
plt.show()