import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

df_heart = pd.read_csv('./Heart.csv')
print(df_heart.head())

df_heart = df_heart.drop("Unnamed: 0",axis=1)

# Converting nominal target into binary encoding
le = LabelEncoder()
df_heart["AHD"] = le.fit_transform(df_heart["AHD"])

print(df_heart.isna().sum())
df_heart = df_heart.dropna(axis=0)

print(df_heart.head())

le = LabelEncoder()

df_heart['ChestPain'] = le.fit_transform(df_heart['ChestPain'])
df_heart['Thal'] = le.fit_transform(df_heart['Thal'])

print(df_heart.head())

# Training
X = df_heart.drop(columns= "AHD",  axis=1)
y = df_heart['AHD']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state=42)

# MinMax Scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Making a simple KNN classifier
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = knn_clf.predict(X_test_scaled)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a figure and axis for the plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(X_test.shape)
print()
print(y_pred.tolist())
print()
print(y_test.tolist())

print('Accuracy: ', accuracy_score(y_pred, y_test))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

