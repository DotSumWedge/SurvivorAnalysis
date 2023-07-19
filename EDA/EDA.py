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

castaways = pd.read_csv("C:/Users/bejes/OneDrive/School/ML/Code/Project1/survivorData/castaway_details.csv")
castaway_details = pd.read_csv("C:/Users/bejes/OneDrive/School/ML/Code/Project1/survivorData/castaways.csv")
c_results = pd.read_csv("C:/Users/bejes/OneDrive/School/ML/Code/Project1/survivorData/challenge_results.csv")
screentime = pd.read_csv("C:/Users/bejes/OneDrive/School/ML/Code/Project1/survivorData/screen_time.csv")

merged_castaways = pd.merge(castaways, castaway_details, on ='castaway_id', how ="left")
merged_data = pd.merge(c_results, castaways, on ='castaway_id', how ="left")
merged_screen = pd.merge(merged_castaways, screentime, on ='castaway_id', how ="inner")

print("--------------------------------------------------------")
print(merged_data.head(5))
print("--------------------------------------------------------")
print(merged_screen)
print("--------------------------------------------------------")


# Filter the dataset to include only rows where the result is 'Won'
wins_df = merged_data[merged_data['result'] == 'Won']

# Count the number of wins by gender
win_counts = wins_df['gender'].value_counts()

# Create a pie chart
plt.pie(win_counts, labels=win_counts.index, autopct='%1.1f%%')
plt.title('Number of Wins by Gender')

# Display the chart
plt.show()









# Group by gender and type, and count the number of wins
win_counts = wins_df.groupby(['gender', 'challenge_type']).size()

# Create a separate pie chart for each gender
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Iterate over each gender
for i, gender in enumerate(['Male', 'Female']):
    ax = axes[i]
    gender_counts = win_counts.loc[gender]
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
    ax.set_title(f'Number of Wins for {gender} by Type')

# Adjust the layout
plt.tight_layout()

# Display the chart
plt.show()






# Count the occurrences of each gender
gender_counts = castaways['gender'].value_counts()

# Create a pie chart
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
plt.title('Count of Each Gender')

# Display the chart
plt.show()







print("--------------------------------------------------------")
# Group the dataset by 'person_id'
df = merged_screen.groupby('castaway_id')

# Calculate the average completion time for each person
average_screen_time = df['screen_time'].mean()
details = merged_castaways[['castaway_id','full_name_x','gender','result']]
average_screen_time = pd.merge(average_screen_time, details, on ='castaway_id', how ="left")

# Display the average completion time for each person
#print(average_screen_time)
print(average_screen_time.sort_values(by='screen_time', ascending=False) )
print("--------------------------------------------------------")

plt.plot(average_screen_time['result'], average_screen_time['screen_time'], marker='o')




