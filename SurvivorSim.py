import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import DataConversionWarning
from xgboost import XGBClassifier
from scipy import stats

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys
import warnings
import math
import os
import random

warnings.filterwarnings("ignore", category=DataConversionWarning)

print("Loaded libs")

# Get the current working directory
current_dir = os.getcwd()

# Get the path to the survivorData directory
data_dir = os.path.join(current_dir, '..', 'survivorData2')

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
    
# Convert tribe_status column to category type
dataframes['tribe_colours.csv']['tribe_status'] = dataframes['tribe_colours.csv']['tribe_status'].astype('category')

# Convert result to a categorical variable
dataframes['castaways.csv']['result'] = pd.Categorical(dataframes['castaways.csv']['result'])
dataframes['castaways.csv']['result'] = dataframes['castaways.csv']['result'].cat.codes

# Merge challenge_results and castaway_details dataframes on castaway_id
castawayAll = pd.merge(dataframes['castaways.csv'], dataframes['castaway_details.csv'], on='castaway_id', how ="left")

# Male = 1, Female = 2, Non-binary = 3
castawayAll['genderNumber'] = np.where(castawayAll['gender'] == 'Male', 1,
                                     np.where(castawayAll['gender'] == 'Female', 2,
                                              np.where(castawayAll['gender'] == 'Non-binary', 3, 0)))

# Binary on if the person was the sole survivor
castawayAll['won'] = np.where(castawayAll['result'] == 'Sole Survivor', 1, 0)

castawayAll = castawayAll.dropna(subset=['age'])

# Read the challenge_results.csv file
challenge_results = dataframes['challenge_results.csv']

# Filter out the rows where challenge_type is "Immunity" and the result is "Won"
immunity_wins = challenge_results[(challenge_results['challenge_type'] == 'Immunity') & (challenge_results['result'] == 'Won')]

# Group by 'castaway_id' and count the number of immunity wins
immunity_counts = immunity_wins.groupby('castaway_id').size()

# Convert Series to DataFrame
immunity_counts_df = immunity_counts.reset_index(name='immunityWins')

# Set 'castaway_id' as the index for easier merging later
immunity_counts_df.set_index('castaway_id', inplace=True)

# Set 'castaway_id' as the index for 'castawayAll' dataframe for easier merging
castawayAll.set_index('castaway_id', inplace=True)

# Merge 'immunity_counts_df' into 'castawayAll'
castawayAll = castawayAll.merge(immunity_counts_df, how='left', left_index=True, right_index=True)

# Fill NaN values with 0 - assuming that contestants who didn't win any immunity challenges are not present in the immunity_counts_df DataFrame
castawayAll['immunityWins'].fillna(0, inplace=True)

# Split dataframe into a list of dataframes grouped by column name
def split_dataframe(df, column_name):
    groups = df.groupby(column_name)
    result = [group for _, group in groups]
    return result

# Call split_dataframe on column name "version_season"
season_split = split_dataframe(castawayAll, 'version_season')

# Give each contestant which order they were eliminated from the show
for df in season_split:
    df['orderOut'] = range(1, len(df) + 1)

def person_prediction(model, remaining_contestants, x_train_current, y_train_current):
    """
    Use the specified model to make a prediction on the next person to be eliminated from the game.
    ...
    """
    
    # Train the model
    model.fit(x_train_current, y_train_current)

    # Predict the order for the remaining contestants
    x_test_current = remaining_contestants[['age', 'genderNumber', 'immunityWins']]
    predicted_order = model.predict(x_test_current)

    # Find the contestants with the smallest predicted order
    min_predicted_order = np.min(predicted_order)
    contestants_with_min_order = np.where(predicted_order == min_predicted_order)[0]

    # Randomly select one contestant from those tied with the smallest predicted order
    person_predicted = np.random.choice(contestants_with_min_order)

    return person_predicted

def models_predict(remaining_contestants, x_train_current, y_train_current, models):
    """
    Use each model in the models list to make a prediction on the next person to be eliminated from the game.
    Returns a list of predictions, one for each model.
    """
    
    predictions = []
    for model_name, model in models:
        prediction = person_prediction(model, remaining_contestants, x_train_current, y_train_current)
        predictions.append((model_name, prediction))

    return predictions


def generate_test_challenge_features(current_season_version, current_season, current_episode_number, challenge_results):
    """
    Adds data from the current episode/version/season
    """
    
    # Filter the challenge results to include only rows up to the current episode, from the current version season, and from the current season
    challenge_results_current = challenge_results[(challenge_results['version_season'] == current_season_version) &
                                                  (challenge_results['season'] == current_season) &
                                                  (challenge_results['episode'] <= current_episode_number)]

    # Generate a DataFrame of the number of wins for each castaway
    challenge_results_current = challenge_results_current[challenge_results_current['result'] == 'Won']\
                        .groupby('castaway_id')['result'].count().reset_index()\
                        .rename(columns={'result': 'immunityWins'})

    # Set 'castaway_id' as the index
    challenge_results_current.set_index('castaway_id', inplace=True)

    return challenge_results_current

def fix_train_challenge_features(x_train, y_train, x_test, current_season_version, current_season):
    """
    Removes contestants in the testing set from the training set so the model doesn't have data from the future
    """
    
    # Identify contestants in x_train who are also in the current test set
    test_contestants = x_test.index if 'castaway_id' not in x_test.columns else x_test['castaway_id']
    
    # Create a mask to identify rows in x_train where the contestant is in the test set
    # and the version_season is the same but the season is not in the future
    mask = ~(
        x_train.index.isin(test_contestants) & 
        ((x_train['version_season'] != current_season_version) | 
        (x_train['version_season'] == current_season_version) & (x_train['season'] > current_season))
    )

    # Apply the mask to x_train and y_train
    x_train = x_train[mask]
    y_train = y_train[mask]
    
    # Remove the 'version_season' and 'season' columns from x_train
    x_train = x_train.drop(columns=['version_season', 'season'])
    
    return x_train, y_train

def calculate_current_episode_number(current_season_version, current_season, current_order_out):
    
    # Filter the dataframe to only the rows that match the current season version, season, and orderOut
    df = dataframes['castaways.csv'][
        (dataframes['castaways.csv']['version_season'] == current_season_version) & 
        (dataframes['castaways.csv']['season'] == current_season) &
        (dataframes['castaways.csv']['order'] == current_order_out)
    ]
    
    if df.empty:
        # If the DataFrame is empty, return the maximum episode number for the given season and version
        max_episode = dataframes['castaways.csv'][
            (dataframes['castaways.csv']['version_season'] == current_season_version) & 
            (dataframes['castaways.csv']['season'] == current_season)
        ]['episode'].max()
        
        return max_episode
    else:
        return df['episode'].iloc[0]



# Splitting the data into training and testing sets using LeaveOneGroupOut.
multilevel_season_splitter = LeaveOneGroupOut()

# Create a list of group labels corresponding to each DataFrame in season_split
group_labels = [i for i, _ in enumerate(season_split)]

# Create a baseline model that randomly guesses
class RandomClassifier:
    def fit(self, X, y):
        # This classifier doesn't learn anything from the data
        pass

    def predict(self, X):
        # Choose a random index for each sample
        return np.random.choice(X.shape[0], size=X.shape[0])

models = [
    ("Random", RandomClassifier()),
    ("SVM", svm.SVC(decision_function_shape='ovo')),
    ("Random Forest", RandomForestClassifier()),
    ("K Neighbors K=1", KNeighborsClassifier(n_neighbors=1)),
    ("K Neighbors K=3", KNeighborsClassifier(n_neighbors=3)),
    ("K Neighbors K=5", KNeighborsClassifier(n_neighbors=5)),
    ("K Neighbors K=7", KNeighborsClassifier(n_neighbors=7)),
    ("K Neighbors K=9", KNeighborsClassifier(n_neighbors=9)),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Naive Bayes", GaussianNB()),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("AdaBoost", AdaBoostClassifier()),
]

# Create an empty dictionary to store accuracies
accuracies = {model_name: [] for model_name, _ in models}

# Create an empty list to store accuracies
accuracies_support_vector_machine = []

for train_index, test_index in multilevel_season_splitter.split(season_split, groups=group_labels):

    x_train = pd.concat([season_split[i][['age', 'genderNumber', 'immunityWins', 'version_season', 'season']] for i in train_index])
    y_train = pd.concat([season_split[i][['orderOut']] for i in train_index])

    x_test = pd.concat([season_split[i][['age', 'genderNumber']] for i in test_index])
    y_test = pd.concat([season_split[i][['orderOut']] for i in test_index])

    number_starting_contestants = len(x_test)
    
    correct_elimination_predictions = 0
    correct_predictions = {model_name: 0 for model_name, _ in models}
    
    current_season_version = season_split[test_index[0]]['version_season'].iloc[0]
    current_season = group_labels[test_index[0]] + 1
    current_episode_number = 1
    x_train, y_train = fix_train_challenge_features(x_train, y_train, x_test, current_season_version, current_season)

    while len(x_test) > 0:
        
        # Get the current_order_out value from y_test
        current_order_out = y_test['orderOut'].iloc[0]
        
        # Calculate the current episode number
        current_episode_number = calculate_current_episode_number(current_season_version, current_season, current_order_out)
        
        # Update x_test to include updated challenge results
        challenge_features = generate_test_challenge_features(current_season_version, current_season, current_episode_number, challenge_results)

        # Drop the 'immunityWins' column from x_test if it exists
        if 'immunityWins' in x_test.columns:
            x_test.drop('immunityWins', axis=1, inplace=True)
        
        # Merge challenge_features into x_test
        x_test = x_test.merge(challenge_features, how='left', on='castaway_id')
        
        # Replace NaN values in 'immunityWins' with 0
        x_test['immunityWins'].fillna(0, inplace=True)

        # Save run the models and save their predictions
        predictions = models_predict(x_test, x_train, y_train, models) 
        
        # Talley the correct predictions
        for model_name, prediction_person_index in predictions:
            # Check if the prediction is correct
            if prediction_person_index == 0:
                correct_predictions[model_name] += 1        

        # Remove the first element of x_test and y_test
        x_test = x_test.iloc[1:]
        y_test = y_test.iloc[1:]

    # Calculate and print the accuracy of the models
    for model_name, correct_prediction_count in correct_predictions.items():
        # Calculate and print the accuracy of the model
        accuracy = (correct_prediction_count / number_starting_contestants) * 100
        accuracies[model_name].append(accuracy)

for model_name, accuracy_list in accuracies.items():
    print(f"Accuracies for {model_name}: {accuracy_list}")
    average_accuracy = sum(accuracy_list) / len(accuracy_list)
    print(f"Average Accuracy for {model_name}: {average_accuracy}")

# Initialize a dictionary to hold average accuracies
average_accuracies = {}

for model_name, accuracy_list in accuracies.items():
    average_accuracy = sum(accuracy_list) / len(accuracy_list)
    average_accuracies[model_name] = average_accuracy

# Convert the average accuracies dictionary to a pandas DataFrame
average_accuracies_df = pd.DataFrame(list(average_accuracies.items()), columns=['Model', 'AverageAccuracy'])

# Write the DataFrame to a CSV file
average_accuracies_df.to_csv('average_accuracies.csv', index=False)

# Similarly, for each model's list of accuracies:
for model_name, accuracy_list in accuracies.items():
    # Convert the list of accuracies to a pandas DataFrame
    accuracies_df = pd.DataFrame(accuracy_list, columns=['Accuracy'])
    # Write the DataFrame to a CSV file
    accuracies_df.to_csv(f'accuracies_{model_name}.csv', index=False)


def test_generate_test_challenge_features():
    # Create a DataFrame that mimics the structure of `challenge_results`
    challenge_results_test = pd.DataFrame({
        'season': [1, 1, 1, 1, 2, 2],
        'episode': [1, 1, 2, 2, 1, 1],
        'castaway_id': ['A', 'B', 'A', 'B', 'A', 'B'],
        'result': ['Won', 'Lost', 'Won', 'Won', 'Won', 'Lost'],
        'version_season': ['S01', 'S01', 'S01', 'S01', 'S02', 'S02']  # Add this line
    })

    # Test 1: Current season version is 'S01', current season is 1 and current episode is 2
    result = generate_test_challenge_features('S01', 1, 2, challenge_results_test)
    expected_result = pd.DataFrame({
        'castaway_id': ['A', 'B'],
        'immunityWins': [2, 1]
    }).set_index('castaway_id')
    pd.testing.assert_frame_equal(result, expected_result)

test_generate_test_challenge_features()