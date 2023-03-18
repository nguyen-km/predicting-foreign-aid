import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import os 

class NeuralNetwork:
    def __init__(self, hidden_units=2, features=3, outputs = 1, activation="sigmoid", reinit = False):
        #randomly initialize weights and bias
        self.__outputs = outputs
        self.W1 = np.random.randn(features,hidden_units) # first weight matrix, ncols x hidden units
        self.b = np.random.randn(hidden_units) # bias vector, hidden units x 1
        self.W2 = np.random.randn(hidden_units,self.__outputs) # second weight matrix, hidden units x 1
        self.__hidden_units = hidden_units
        self.H = None
        self.c = np.random.randn(self.__outputs)
        self.__Z = None
        self.__Z2 = None
        self.__activation = activation.strip().lower()
        self.y_hat = None
        self.avgLoss = []
        self.totLoss = []
    # activation functions
    def activation(self, z, deriv = False, softmax = False, activation = 'sigmoid', alpha = 0.01):
        if (softmax):
            exp = np.exp(z)
            return(exp/np.sum(exp, axis=1)[:,None])
        elif (activation == "sigmoid"):
            return 1 / (1+np.exp(-z)) if (deriv==False) else np.multiply(z,1-z) 
        elif (activation == "relu"): # Leaky ReLU
            if (deriv == False):
                val = np.maximum(alpha*z, z)
            else:
                def reluDeriv(x): 
                    return 1 if(x > 0) else alpha
                func_vec = np.vectorize(lambda t: reluDeriv(t)) #Vectorize the derivative of reLU
                val = func_vec(z)
            return val
            
    def fit(self, X, y, eta = 0.1, epochs=1000):

        for i in range(epochs):
            self.y_hat = self.predict(X)
            error = self.y_hat - y # n x o (y^-y)

            #calculate loss categorical cross entropy
            loss = np.mean(-y * np.log(self.y_hat)) ## We need y to place the "1" in the right place
            # print("The current average loss is\n", loss)
            self.avgLoss.append(loss)
            self.totLoss.append(loss)

            d_error = error # n x o derivative for categorical cross entropy and softmax
            H_deriv = self.activation(self.H, deriv=True, activation = self.__activation) # n x h (Activation'(H))
            H_deriv_d_error_w2 = np.multiply((d_error @ self.W2.T), H_deriv)
            
            dW1 = X.T @ H_deriv_d_error_w2
            dW2 = self.H.T @ d_error
            db = np.array(np.multiply(H_deriv.T @ d_error, self.W2).sum(axis=1)) # h x 1
            dc = d_error.sum()
            
            self.W1 = self.W1 - (eta * dW1)
            self.W2 = self.W2 - (eta * dW2)
            self.b = np.squeeze(self.b.reshape(-1,1) - (eta*db.reshape(-1,1)))
            self.c = self.c - (eta*dc)
        return(self.y_hat)
            
    def predict(self, X):
        self.__Z = (X @ self.W1) + self.b # n x h
        self.H = self.activation(self.__Z, activation = self.__activation) # n x h
        self.__Z2 = (self.H @ self.W2) + self.c # (nxh)x(hxo) = n x o
        y_hat = self.activation(self.__Z2, softmax=True)
        self.y_hat = y_hat
        return self.y_hat


os.chdir('/Users/kevnguyen/Library/CloudStorage/GoogleDrive-keng2413@colorado.edu/My Drive/CSCI5622/project')
df = pd.read_csv('data/final_clean_data.csv')
print(df.head())
X = StandardScaler().fit_transform(df._get_numeric_data())
print(X[0:5])

y = OneHotEncoder().fit_transform(df['Aid Level'].values.reshape(-1,1)).toarray()
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

nn= NeuralNetwork(hidden_units = 10, features = X_train.shape[1], outputs=y_train.shape[1], activation = 'relu')
nn.fit(X_train, y_train, eta = 0.01, epochs = 3000)
y_pred_probs = nn.predict(X_test)
# print(y_pred_probs)

y_pred_list = list()
y_test_list = list()
for i in range(len(y_pred_probs)):
    y_pred_list.append(np.argmax(y_pred_probs[i]))
    y_test_list.append(np.argmax(y_test[i]))


print('Truth', y_test_list)
print('Preds', y_pred_list)

from sklearn.metrics import accuracy_score
a =accuracy_score(y_pred_list,y_test_list)
print('Accuracy is:', round(a*100, 2), '%')