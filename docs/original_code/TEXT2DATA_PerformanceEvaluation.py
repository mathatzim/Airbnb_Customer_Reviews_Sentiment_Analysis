# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 21:13:39 2025

@author: mathaios
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the data from the Excel file
# We load the dataset from an Excel file using pandas. It is assumed that this dataset has two relevant columns:
# - 'Predicted_Sentiment': Sentiment predicted by the model (either Positive or Negative).
# - 'Actual_Sentiment': Actual sentiment label (either Positive or Negative) that corresponds to each review.
data = pd.read_excel('Reviews_Sentiments.xlsx')  # Load the dataset into a pandas DataFrame

# Step 2: Extract the columns containing predicted and actual sentiment
# In this step, we extract the two columns that contain the predicted sentiment and the actual sentiment.
# These columns will be used for comparing how well the model performed.
# - 'Predicted_Sentiment' column holds the model's prediction for each review's sentiment.
# - 'Actual_Sentiment' column contains the true sentiment (ground truth) for each review.

predicted_sentiment = data['Predicted_Sentiment']  # Model's predicted sentiment labels
actual_sentiment = data['Actual_Sentiment']  # Actual sentiment labels (the ground truth)

# Step 3: Generate the confusion matrix
# The confusion matrix helps assess the performance of a classification model by showing how many instances of 
# each class (positive or negative) were correctly or incorrectly classified.
# The confusion matrix will display the following information:
# - True positives (TP): Reviews that were correctly classified as positive.
# - False positives (FP): Reviews that were incorrectly classified as positive.
# - True negatives (TN): Reviews that were correctly classified as negative.
# - False negatives (FN): Reviews that were incorrectly classified as negative.
cm = confusion_matrix(actual_sentiment, predicted_sentiment)  # Generate confusion matrix

# Step 4: Generate the classification report
# The classification report provides detailed performance metrics such as precision, recall, and F1-score for each class (positive and negative).
# - Precision: The proportion of correctly predicted positive reviews (TP) out of all predicted positive reviews (TP + FP).
# - Recall: The proportion of correctly predicted positive reviews (TP) out of all actual positive reviews (TP + FN).
# - F1-Score: The harmonic mean of precision and recall. It provides a balance between precision and recall.
# The classification report will give a summary of these metrics for both positive and negative sentiments.
class_report = classification_report(actual_sentiment, predicted_sentiment, target_names=['Negative', 'Positive'])  # Generate classification report

# Step 5: (Optional) Visualize the confusion matrix using seaborn heatmap
# A heatmap is used to visually display the confusion matrix. It helps to see the distribution of correct and incorrect classifications.
# - `annot=True` adds the numerical values on each cell in the matrix.
# - `fmt='d'` formats the annotations as integers (since they represent counts of predictions).
# - `cmap='Blues'` sets the color scheme of the heatmap to shades of blue.
# - `xticklabels` and `yticklabels` specify the labels for the axes (Negative and Positive sentiments).
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])

# Add labels and title to the heatmap for clarity
plt.xlabel('Predicted Sentiment')  # Label the x-axis as 'Predicted Sentiment'
plt.ylabel('Actual Sentiment')  # Label the y-axis as 'Actual Sentiment'
plt.title('Confusion Matrix for Sentiment Analysis')  # Title for the heatmap

# Show the plot
plt.show()  # This displays the confusion matrix heatmap

# Step 6: Print the confusion matrix values as well
# Output the raw confusion matrix values. This will give us the exact counts of TP, FP, TN, and FN.
# For example:
# [[TP, FP],
#  [FN, TN]]
print("Confusion Matrix:\n", cm)  # Print the confusion matrix values

# Step 7: Print the classification report
# The classification report contains precision, recall, and F1-score for each class (positive and negative).
# It gives a good overview of the model's performance and its ability to correctly classify reviews into positive and negative categories.
print("\nClassification Report:\n", class_report)  # Print the classification report