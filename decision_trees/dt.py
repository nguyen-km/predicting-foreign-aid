from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

path ='/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project/data/final_clean_data.csv'
df = pd.read_csv(path, index_col=2)

X=StandardScaler().fit_transform(df.drop('Obligations', axis =1).drop(df.columns[0],axis =1)._get_numeric_data()) #scale the numeric data
y = df['Aid Level']

X_train, X_test, y_train, y_test = train_test_split(X,y)

dt = tree.DecisionTreeClassifier()

parameters = {'criterion':['gini', 'entropy', 'log_loss'],
              'splitter': ['best', 'random'],
              'max_features': [None,'sqrt', 'log2'],
              'class_weight': [None, 'balanced']}

grid = GridSearchCV(dt, parameters)
grid.fit(X_train, y_train)

print(grid.best_params_) 

dt.set_params(**grid.best_params_)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
tree.plot_tree(dt)

print(confusion_matrix(y_test,y_pred))
