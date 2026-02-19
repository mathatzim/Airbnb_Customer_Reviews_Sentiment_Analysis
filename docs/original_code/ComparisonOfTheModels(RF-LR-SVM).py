# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 15:26:44 2025

@author: mathaios
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Step 1: Load the dataset from an Excel file containing reviews and sentiment labels
data = pd.read_excel('Reviews_Sentiments.xlsx')

# Step 2: Remove duplicate reviews based on 'Review_Text' column
data = data.drop_duplicates(subset='Review_Text', keep='first')

# Step 3: Remove rows with missing review texts to ensure complete data
data = data.dropna(subset=['Review_Text'])

# Step 4: Define a function to clean the text (convert to lowercase)
def clean_text(text):
    text = text.lower()  # Lowercase conversion
    return text

# Step 5: Apply the text cleaning function to each review in the dataset
data['Cleaned_Review_Text'] = data['Review_Text'].apply(clean_text)

# Step 6: Check the number of reviews remaining after cleaning
print(f"Dataset contains {len(data)} reviews.")

# Step 7: Define features (X) and target (y)
X = data['Cleaned_Review_Text']  # Features: review text
y = data['Actual_Sentiment']  # Target: sentiment (0 for negative, 1 for positive)

# Step 8: Split the data into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Step 9: Convert text data into numerical format using TF-IDF Vectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=200)  # Using unigrams, bigrams, and trigrams
X_train_tfidf = tfidf.fit_transform(X_train).toarray()  # Fit and transform on training data
X_test_tfidf = tfidf.transform(X_test).toarray()  # Transform on test data

# Step 10: Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_tfidf, y_train)

# Step 11: Make predictions on the test set
y_pred = rf_model.predict(X_test_tfidf)

# Step 12: Evaluate Random Forest model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report RandomForest:")
print(classification_report(y_test, y_pred))

# Step 13: Implement Logistic Regression model for comparison
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_tfidf, y_train)

# Step 14: Make predictions using Logistic Regression model
y_pred_log_reg = log_reg.predict(X_test_tfidf)

# Step 15: Evaluate Logistic Regression model
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))
print("\nClassification Report Logistic Regression:")
print(classification_report(y_test, y_pred_log_reg))

# Step 16: Implement Support Vector Machine (SVM) model for comparison
svm_model = SVC(kernel='linear', random_state=42)  # Using linear kernel for classification
svm_model.fit(X_train_tfidf, y_train)

# Step 17: Make predictions using SVM model
y_pred_svm = svm_model.predict(X_test_tfidf)

# Step 18: Evaluate SVM model
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report SVM:")
print(classification_report(y_test, y_pred_svm))