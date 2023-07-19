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

# Count the number of challenge categories won grouped by gender
challenges_counts_grouped = (challenges.set_index('gender').iloc[:, 29:] == True).stack().groupby(level=[0, 1]).sum().unstack(level=0)

print(challenges_counts_grouped)

challenges_counts_grouped = challenges_counts_grouped[['Male', 'Female', 'Non-binary']]

# Create a stacked bar chart
bar_chart_axes = challenges_counts_grouped.plot.bar(stacked=True, color={'Male': '#87CEEB', 'Female': 'pink', 'Non-binary': 'green'})

# Get the challenge types from challenge_description.csv
genders = dataframes['challenge_description.csv'].columns[2:]

# Set the xtick labels to the challenge types
plt.xticks(range(len(genders)), genders, rotation=90)

# Add labels and title
plt.xlabel('Challenge Category')
plt.ylabel('Number of Challenges Won')
plt.title('Number of Challenges Won by Gender')

# Change the order of the legend
handles, labels = bar_chart_axes.get_legend_handles_labels()
bar_chart_axes.legend(reversed(handles), reversed(labels))

# Show the plot
plt.show()