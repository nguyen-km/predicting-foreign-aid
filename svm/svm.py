from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

os.chdir('/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project')

path ='data/final_clean_data.csv'
df = pd.read_csv(path, index_col=1)
print(df.head())

df_numeric = df._get_numeric_data()

X =StandardScaler().fit_transform(df_numeric) #scale the numeric data
y = df['Aid Level']

X_train, X_test, y_train, y_test = train_test_split(X,y)

kernels = ['linear', 'poly', 'rbf']
C = [0.1, 1, 10]

for i in kernels:
    degree_grid = [2, 3, 4] if i=='poly' else [3]
    gamma_grid = ['scale', 'auto'] if i == 'rbf' else ['scale']
    for c in C:
        params = {'C': [c],
                  'kernel': [i],
                  'gamma': gamma_grid,
                  'degree': degree_grid,
                  'shrinking': [True, False],
                  'class_weight': [None, 'balanced'],
                  }

        svm = SVC()
        grid = GridSearchCV(svm, params)
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)
        print('{} kernel with {} cost\n'.format(i, c))
        # print(grid.best_params_)
        print(accuracy_score(y_test, y_pred), '\n')
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels = grid.classes_).plot()
        plt.title('SVM Confusion Matrix ({} kernel with cost = {})'.format(i, c))
        plt.savefig('svm/cm_svm_{}_{}.png'.format(i, c), dpi = 300)
        # plt.show()

