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

# Remove the last 2 characters of challange_id so it can be used a primary key
dataframes['challenge_description.csv']['challenge_id'] = dataframes['challenge_description.csv']['challenge_id'].str[:-2]

# Merge challenge_results and castaway_details dataframes on castaway_id
castaway_results = pd.merge(dataframes['challenge_results.csv'], dataframes['castaway_details.csv'], on='castaway_id')

# Merge challenge_results and challenge_description dataframes on challenge_id
challenges = pd.merge(castaway_results, dataframes['challenge_description.csv'], on='challenge_id')

# Count the number of challenge categories won grouped by gender but with gender on the x axis
challenges_counts_grouped = (challenges.iloc[:, 29:] == True).groupby(challenges['gender']).sum()

# Change the column name from 'race_y' to 'race'
challenges_counts_grouped = challenges_counts_grouped.rename(columns={'race_y': 'race'})

# Create a dictionary of colors for each challenge type
colors = {
    'water': '#1E90FF', # dodger blue
    'fire': '#FF4500', # orange red
    'memory': '#FFDAB9', # peach puff
    'knowledge': '#4B0082', # indigo
    'food': '#FF69B4', # hot pink
    'balance': '#00BFFF', # deep sky blue
    'turn_based': '#8B008B', # dark magenta
    'strength': '#8B0000', # dark red
    'endurance': '#228B22', # forest green
    'precision': '#00FF7F', # spring green
    'race': '#FFD700', # gold
    'puzzle': '#FFA500' # orange
}

# Create a stacked bar chart
bar_chart_axes = challenges_counts_grouped.plot.bar(stacked=True, color=colors)

# Get the gender categories for the x axis labels
genders = dataframes['castaway_details.csv']['gender'].unique().tolist()

# Set the xtick labels to the challenge types
plt.xticks(range(len(genders)), genders, rotation=90)

# Add labels and title
plt.xlabel('Gender')
plt.ylabel('Number of Challenges Won')
plt.title('Number of Challenges Won by Gender')

# Change the order of the legend
handles, labels = bar_chart_axes.get_legend_handles_labels()
bar_chart_axes.legend(reversed(handles), reversed(labels))

# Show the plot
plt.show()