# Step 1: Imports

# Import packages
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import sklearn.metrics as metrics

import matplotlib.pyplot as plt

# Import data
df_original = pd.read_csv('.Invistico_Airline.csv')

print(df_original.head())

# Step 2: Data exploration, data cleaning, and model preparation
# Exploring the data
# Checking for missing values
# Encoding the data
# Renaming a column
# Creating the training and testing data

# Explore the data
print(df_original.dtypes)
print(df_original['Class'].unique())

# Check for missing values
print(df_original.isnull().sum())

# Drop the rows with missing values
df_subset = df_original.dropna(axis=0).reset_index(drop=True)

# Encode the data
df_subset['Class'] = df_subset['Class'].map({'Business':3,'Eco Plus':2,'Eco':1})
df_subset['satisfaction'] = df_subset['satisfaction'].map({'satisfied':1,'dissatisfied':0})
df_subset = pd.get_dummies(df_subset)

print(df_subset.dtypes)

# Create the training and testing data
y = df_subset['satisfaction']

X = df_subset.copy()
X = X.drop('satisfaction',axis=1)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# Step 3: Model building
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train,y_train)
dt_pred = decision_tree.predict(X_test)

# Step 4: Results and evaluation
print("Decision Tree")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, dt_pred))
print("Precision:", "%.6f" % metrics.precision_score(y_test, dt_pred))
print("Recall:", "%.6f" % metrics.recall_score(y_test, dt_pred))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, dt_pred))

# Produce a confusion matrix
cm = metrics.confusion_matrix(y_test,dt_pred,labels = decision_tree.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = decision_tree.classes_)
disp.plot()
plt.show()

# Plot the decision tree
plt.figure(figsize=(20,12))
plot_tree(decision_tree, max_depth=2, fontsize=14, feature_names=X.columns)
plt.tight_layout()
plt.show()

importances = decision_tree.feature_importances_
forest_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
plt.tight_layout()
plt.show()