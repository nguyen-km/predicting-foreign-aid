from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project')

# Data Prep
path ='data/final_clean_data.csv'
df = pd.read_csv(path, index_col=1)
print(df.head())

X=df._get_numeric_data() #only the numeric data
print(X[0:5])
y = df['Aid Level']

X_train, X_test, y_train, y_test = train_test_split(X,y)


# Single decision Tree (Untuned)
dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

## Model evaluation
###accuracy
print(accuracy_score(y_test, y_pred))

### Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred), display_labels = dt.classes_).plot()
plt.title('Confusion Matrix for Decision Tree (without tuning)')
plt.savefig('decision_trees/imgs/cm.png', dpi = 300)

### Tree
fig = plt.figure(figsize=(25,25))
_ = tree.plot_tree(dt, 
                   feature_names=X.columns.values.tolist(),
                   class_names = dt.classes_,
                   filled=True)
plt.title('Decision Tree without tuning')
plt.savefig('decision_trees/imgs/dt.png', dpi = 300)


## Single Decision Tree (Tuned)
dt = tree.DecisionTreeClassifier()
param_grid = {'criterion':['gini', 'entropy', 'log_loss'],
              'splitter': ['best', 'random'],
              'max_features': [None,'sqrt', 'log2'], 
              'max_depth' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'class_weight': [None, 'balanced']}

grid = GridSearchCV(dt, param_grid)
grid.fit(X_train, y_train)

print(grid.best_params_) # Output optimal parameters

## fit the model
y_pred_tuned = grid.predict(X_test)

## Model Evaluation
print(accuracy_score(y_test, y_pred_tuned))

### Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred_tuned), display_labels = grid.classes_).plot()
plt.title('Confusion Matrix for Decision Tree w/ Tuning')
plt.savefig('decision_trees/imgs/cm_tuned.png', dpi = 200)

### Tree
fig = plt.figure(figsize=(25,25))
_ = tree.plot_tree(grid.best_estimator_, 
                   feature_names=X.columns.values.tolist(),
                   class_names = grid.classes_,
                   filled=True)
plt.title('Decision Tree w/ Hyperparameter Tuning')
plt.savefig('decision_trees/imgs/dt_tuned.png', dpi = 300)


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
y_pred_rf = grid_rf.predict(X_test)

print(accuracy_score(y_test, y_pred_rf))

#plot Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf), display_labels = grid_rf.classes_).plot()
plt.title('Confusion Matrix for Random Forest')
plt.savefig('decision_trees/imgs/cm_rf.png', dpi = 300)

# plt.show()
