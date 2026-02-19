# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 18:39:39 2025

@author: mathaios
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
# The dataset is loaded from an Excel file containing reviews and their corresponding sentiment labels.
data = pd.read_excel('Reviews_Sentiments.xlsx')

# Step 2: Remove duplicate reviews
# The following line removes any duplicate reviews based on the 'Review_Text' column.
# By default, 'drop_duplicates' keeps the first occurrence of each duplicate, and removes all others.
data = data.drop_duplicates(subset='Review_Text', keep='first')  # Preprocessing: Remove duplicate reviews

# Step 3: Drop rows with missing review texts
# In this step, rows with missing values in the 'Review_Text' column are removed.
# This ensures that we only work with complete reviews.
data = data.dropna(subset=['Review_Text'])  # Preprocessing: Remove rows with missing review texts

# Step 4: Data Cleaning Function
# A function is defined to clean the review text.
# This is a basic example of text cleaning, where the review text is converted to lowercase.
def clean_text(text):
    # Convert the text to lowercase
    text = text.lower()
    return text  # Return the cleaned text

# Step 5: Apply the cleaning function to all reviews
# The 'apply()' method is used to apply the 'clean_text' function to each review in the 'Review_Text' column.
# The cleaned reviews are stored in a new column called 'Cleaned_Review_Text'.
data['Cleaned_Review_Text'] = data['Review_Text'].apply(clean_text)  # Preprocessing: Clean the text

# Step 6: Check the size of the dataset
# This is a check to ensure that the dataset contains a reasonable number of reviews after cleaning.
print(f"Dataset contains {len(data)} reviews.")

# Step 7: Define independent variable (X) and dependent variable (y)
# Here, we define X (features) and y (target variable).
# X will contain the cleaned review texts, and y will contain the actual sentiment (positive or negative) of the review.
# The 'Actual_Sentiment' column is assumed to contain a binary classification: 0 for negative and 1 for positive.
X = data['Cleaned_Review_Text']  # Features (review texts)
y = data['Actual_Sentiment']  # Target variable (actual sentiment labels)

# Step 8: Split data into training and testing sets
# We use 'train_test_split' to split the dataset into training and testing sets.
# 90% of the data will be used for training, and 10% will be used for testing.
# The random_state parameter ensures reproducibility of the results.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # Data Preprocessing: Split data

# Step 9: Convert 'Review_Text' into numerical features using TF-IDF Vectorizer
# The TfidfVectorizer converts the text data into numerical form that can be used by machine learning models.
# ngram_range=(1, 3) creates unigrams, bigrams, and trigrams from the review text.
# max_features=200 limits the number of features to the 200 most important ones.
# Fit the vectorizer on the training data and transform both the training and testing data into vectors.
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=200)  # Feature extraction: TF-IDF Vectorization
X_train_tfidf = tfidf.fit_transform(X_train).toarray()  # Fit and transform on the train set
X_test_tfidf = tfidf.transform(X_test).toarray()  # Transform the test set

# Step 10: Feature Selection
# In this case, we have already performed feature selection in the previous step through the TF-IDF vectorizer,
# where we limited the number of features to the top 200. This is a basic form of feature selection.
# You can improve feature selection further by using techniques like Chi-Square, Mutual Information, or feature importance from a model.

# Step 11: Initialize and train the Random Forest model
# We initialize a RandomForestClassifier, which is an ensemble machine learning algorithm.
# We set n_estimators=100 to use 100 decision trees and class_weight='balanced' to address potential class imbalance.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model on the training data
rf_model.fit(X_train_tfidf, y_train)

# Step 12: Make predictions on the test set
# After training the model, we use it to make predictions on the test set.
y_pred = rf_model.predict(X_test_tfidf)

# Step 13: Evaluate the model's performance
# We evaluate the performance of the model using accuracy, confusion matrix, and classification report.

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 14: Display the confusion matrix
# A confusion matrix is a table that shows the actual vs predicted classifications.
# It helps assess how well the model is performing by displaying True Positives, False Positives, True Negatives, and False Negatives.
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Step 15: Display the classification report
# The classification report provides precision, recall, and F1-score metrics for each class (positive and negative).
# These metrics are essential for evaluating the model's performance.
print("\nClassification Report RandomForest:")
print(classification_report(y_test, y_pred))

# Step 16: Visualize the confusion matrix using a heatmap
# A heatmap is used to visualize the confusion matrix in a more understandable way.
# The heatmap shows the number of true positives, false positives, true negatives, and false negatives.
# This visualization helps better interpret the performance of the model.
plt.figure(figsize=(6, 5))  # Set figure size for better clarity
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')  # Add title to the plot
plt.xlabel('Predicted')  # Label for the x-axis
plt.ylabel('Actual')  # Label for the y-axis
plt.show()  # Display the heatmap plot