import pandas as pd
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Data
path ='/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project/data/final_clean_data.csv'
df = pd.read_csv(path, index_col=1)
# print(df.head())

X_numeric =StandardScaler().fit_transform(df._get_numeric_data()) #scale the numeric data
y = df['Aid Level']

X_train_numeric, X_test_numeric, y_train, y_test = train_test_split(X_numeric,y)


# GAUSSIAN NAIVE BAYES for quantitative data
quantNB = GaussianNB().fit(X_train_numeric,y_train)
y_pred_quant = quantNB.predict(X_test_numeric)
ConfusionMatrixDisplay(confusion_matrix(y_test,y_pred_quant), display_labels= quantNB.classes_).plot()
plt.title('Confusion Matrix for Gaussian Naive Bayes')


#CATEGORICAL NAIVE BAYES for categorical data
X_cat = df.select_dtypes(exclude=["number"]).drop([df.columns[0], 'Aid Level'], axis = 1)
print(X_cat)

X_cat = OrdinalEncoder().fit_transform(X_cat) # encode data to be numeric 

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat,y)
print(X_train_cat)

# ## hyperparameter tuning
catNB = CategoricalNB().fit(X_train_cat, y_train_cat)
params = {'alpha': [0.001, 0.01, 0.1, 0.2, 0.5, 0.75, 1, 2, 3]}
grid = GridSearchCV(catNB, params)
grid.fit(X_train_cat, y_train_cat)

y_pred_cat= grid.predict(X_test_cat)
ConfusionMatrixDisplay(confusion_matrix(y_test_cat,y_pred_cat), display_labels = grid.classes_).plot()
plt.title('Confusion Matrix for Categorical Naive Bayes')

plt.show()
