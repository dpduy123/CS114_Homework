import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

data = load_iris()
df = pd.DataFrame(data.data, columns = data.feature_names)
df['species'] = data.target

print(df.head())

X = df.drop(columns = ['species'])
y = df['species']

clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(X,y)
tree.plot_tree(clf_tree, feature_names=data.feature_names,
              filled=True,
              class_names=data.target_names,
              fontsize=7)
plt.tight_layout()
plt.show()
