from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Data Prep
path ='/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project/data/final_clean_data.csv'
df = pd.read_csv(path, index_col=1)
print(df.head())

X=df._get_numeric_data() #only the numeric data
print(X[0:5])
y = df['Aid Level']

X_train, X_test, y_train, y_test = train_test_split(X,y)


# Single decision Tree
dt = tree.DecisionTreeClassifier()

## Hyperparameter Tuning
param_grid = {'criterion':['gini', 'entropy', 'log_loss'],
              'splitter': ['best', 'random'],
              'max_features': [None,'sqrt', 'log2'], 
              'max_depth' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'class_weight': [None, 'balanced']}

grid = GridSearchCV(dt, param_grid)
grid.fit(X_train, y_train)

print(grid.best_params_) # Output optimal parameters

## fit the model
dt.set_params(**grid.best_params_)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)


## Plotting

### Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred)).plot()
plt.title('Confusion Matrix for Decision Tree')

### Tree
fig = plt.figure(figsize=(25,25))
_ = tree.plot_tree(dt, 
                   feature_names=X.columns.values.tolist(),
                   class_names = dt.classes_,
                   filled=True)


# Random Forrest
rf = RandomForestClassifier()

## Hyper Parameter Tuning
params_rf = {'n_estimators' : [2, 5, 10, 50, 100, 200], # number of trees to estimate
             'criterion' :['gini', 'entropy', 'log_loss'],
             'max_features': [None,'sqrt', 'log2'],
             'n_jobs': [-1], # Run on all availible processors
             'bootstrap': [True, False], # If false, all data used to train each tree
             'class_weight': [None, 'balanced', 'balanced_subsample']}

grid_rf = GridSearchCV(rf, params_rf)
grid_rf.fit(X_train, y_train)
print(grid_rf.best_params_)

# Fit decision tree
rf.set_params(**grid_rf.best_params_)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

#plot Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf)).plot()
plt.title('Confusion Matrix for Random Forest')

plt.show()
