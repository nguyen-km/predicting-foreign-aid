from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt

path ='/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project/data/final_clean_data.csv'
df = pd.read_csv(path, index_col=2)

X =df.drop('Obligations', axis =1).drop(df.columns[0],axis =1)._get_numeric_data()

y = df['Aid Level']
print(X.head())


X_train, X_test, y_train, y_test = train_test_split(X,y)

dt = tree.DecisionTreeClassifier()

parameters = {'criterion':['gini', 'entropy', 'log_loss'], 
              'splitter': ['best', 'random']}

clf = GridSearchCV(dt, parameters)
clf.fit(X_train, y_train)

print(clf.best_params_) 

dt.fit(X_train, y_train)
dt.predict(X_test)
tree.plot_tree(dt)
plt.show()


