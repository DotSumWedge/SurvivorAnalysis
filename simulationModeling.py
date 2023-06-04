
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, LeaveOneGroupOut

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os

# Get the path to the survivorData directory
data_dir = os.path.join(os.path.dirname(__file__), '..', 'survivorData')

# List of CSV file names
csv_files = [
    'castaways.csv',
    'castaway_details.csv',
    'challenge_results.csv'
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

# Convert result to a categorical variable
dataframes['castaways.csv']['result'] = pd.Categorical(dataframes['castaways.csv']['result'])
dataframes['castaways.csv']['result'] = dataframes['castaways.csv']['result'].cat.codes

# Merge challenge_results and castaway_details dataframes on castaway_id
castawayAll = pd.merge(dataframes['castaways.csv'], dataframes['castaway_details.csv'], on='castaway_id', how ="left")

castawayAll['genderNumber'] = np.where(castawayAll['gender'] == 'Male', 1,
                                     np.where(castawayAll['gender'] == 'Female', 2,
                                              np.where(castawayAll['gender'] == 'Non-binary', 3, 0)))
castawayAll['won'] = np.where(castawayAll['result'] == 'Sole Survivor', 1, 0)

castawayAll = castawayAll.dropna(subset=['age'])

# Drop rows where 'season' is equal to 44
castawayAll = castawayAll[castawayAll['season'] != 44]

def split_dataframe(df, column_name):
    groups = df.groupby(column_name)
    result = [group for _, group in groups]
    return result

# Call split_dataframe on column name "version_season"
season_split = split_dataframe(castawayAll, 'version_season')

for df in season_split:
    df['orderOut'] = range(1, len(df) + 1)

# Perform leave-one-group-out cross-validation
logo = LeaveOneGroupOut()

# Initialize a list to store the predicted outcomes
predicted_outcomes = []

# Create a list to store the true outcomes
true_outcomes = []

# Create a list of group labels corresponding to each DataFrame in season_split
group_labels = [i for i, _ in enumerate(season_split)]

# Initialize the SVM model
model = svm.SVC()

# Iterate through each test set
for train_index, test_index in logo.split(season_split, groups=group_labels):
    # Get the training and test sets for the current iteration
    train_set = [season_split[i] for i in train_index]
    test_set = [season_split[i] for i in test_index]

    # Initialize the SVM model for each iteration
    model = svm.SVC()

# Iterate through each test set
for train_index, test_index in logo.split(season_split, groups=group_labels):
    # Get the training and test sets for the current iteration
    train_set = [season_split[i] for i in train_index]
    test_set = [season_split[i] for i in test_index]

    # Iterate through each test DataFrame
    for i, test_df in enumerate(test_set):
        X_train = pd.concat(train_set)[['age', 'genderNumber']]
        y_train = pd.concat(train_set)['orderOut']

        # Fit the SVM model on the training data
        model.fit(X_train, y_train)

        # Predict the orderOut values for the test set
        X_test = test_df[['age', 'genderNumber']]
        y_pred = model.predict(X_test)

        # Save the predicted outcomes to the list
        predicted_outcomes.extend(y_pred)

        # Save the true outcomes to the list
        true_outcomes.extend(test_df['orderOut'])

        # Remove the row with the lowest orderOut value from the possible choices
        min_orderOut = min(test_df['orderOut'])
        test_set[i] = test_df[test_df['orderOut'] > min_orderOut]

# Print the predicted outcomes
print("Predicted Outcomes:")
print(predicted_outcomes)

# Calculate the accuracy of the predicted outcomes
accuracy = accuracy_score(true_outcomes, predicted_outcomes)
print("Accuracy:", accuracy)