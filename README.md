# RandomForestSVM-SpamDetection
Spam detection with RandomForest and SVM classification 
# Random Forest and SVM Text Spam Detection

This repository contains code for a text spam detection project using Random Forest and SVM classifiers.

## Dataset

The dataset used for this project is stored in the file `spam.tsv`, which is a TSV (Tab-Separated Values) file. The dataset is loaded and processed using pandas in the provided Python script.

## Data Exploration

### Dataset Overview

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('spam.tsv', sep='\t')

# Display basic information about the dataset
print(df.head())
print(df.isna().sum())
print(df['label'].value_counts())

# Code for balancing the dataset (sampling equal numbers of 'ham' and 'spam' records)
# ...

# Display the shape of the balanced dataset
# ...
# Code for visualizing message lengths for 'ham' and 'spam'
# ...
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
# ...
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Create a pipeline with TF-IDF vectorizer and Random Forest classifier
# ...
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Evaluate the model performance
# ...
pip install pandas matplotlib scikit-learn
python spam_detection.py

