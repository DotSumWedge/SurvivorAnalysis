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

df = dataframes['tribe_colours.csv'][dataframes['tribe_colours.csv']['version_season'] == 'AU01']
# Create a count plot
sns.countplot(data=df, x='tribe_status', hue='tribe_colour', palette=sns.color_palette(df['tribe_colour'].unique()))

# Show the plot
plt.show()