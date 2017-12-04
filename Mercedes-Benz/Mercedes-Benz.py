### Mercedes-Benz Green Challenge

# Importing the basic libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the datsets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

### EDA
# Basic data summary info

train.info()
print("-"*40)
test.info()

train.describe()
train.describe(include = ['O'])
test.describe()
test.describe(include = ['O'])

# Outlier treatment - Removing 'y' values > 140
train_outliers = train[train['y'] >= 140]
train = train[train['y'] < 140]

# Label Encoding
from sklearn.preprocessing import LabelEncoder
features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']
for x in features:
    le = LabelEncoder()
    le.fit(list(train[x].values) + list(test[x].values))
    train[x] = le.transform(list(train[x].values))
    test[x] = le.transform(list(test[x].values))
   

# Separating numerical and categorical features
num_feat_train = train.iloc[:, 10::].values
cat_feat_train = train.iloc[:, 2:10].values
num_feat_test = test.iloc[:, 9::].values
cat_feat_test = test.iloc[:, 1:9].values

y_train = train['y'].values

# Removing unimportant features
''' Numerical features which have very few zeros/ones across train and test do not add 
    any valuable information to our models. Hence, they can be done away with.    
'''
num_feat = np.append(num_feat_train, num_feat_test, axis = 0)
col_sum = np.sum(num_feat, axis = 0)
relevant_cols = np.array(np.zeros(col_sum.shape[0]))
for i in np.arange(0, col_sum.shape[0]) :
    relevant_cols[i] = col_sum[i] > 4 and col_sum[i] < 8380
relevant_cols = relevant_cols.astype(bool)
num_feat = num_feat[:, relevant_cols]

num_feat_train = num_feat[0:num_feat_train.shape[0],:]
num_feat_test = num_feat[num_feat_train.shape[0]::,:]


# Combining numerical and categorical features
feat_comb_train = np.append(cat_feat_train, num_feat_train, axis = 1)
feat_comb_test = np.append(cat_feat_test, num_feat_test, axis = 1)

# Adding ID as a feature as well
feat_comb_train = np.append(feat_comb_train, train['ID'].values.reshape(-1,1), axis = 1)
feat_comb_test = np.append(feat_comb_test, test['ID'].values.reshape(-1,1), axis = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
feat_comb_train = sc.fit_transform(feat_comb_train)
feat_comb_test = sc.transform(feat_comb_test)


# Reducing number of numerical features using PCA/MCA
# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
num_feat_train_pca = pca.fit_transform(num_feat_train)
num_feat_test_pca = pca.transform(num_feat_test)
explained_variance = pca.explained_variance_ratio_
plt.plot(np.arange(1, 369), explained_variance.cumsum()) # Plots the explained variance curve
''' From the above curve, 150 seems to be the ideal number of PCA components. It explains
    ~ 97% of variance. Even 100 can be used as it explains ~90% of variance  ''' 
del num_feat_train_pca, num_feat_test_pca
pca = PCA(n_components = 120)
num_feat_train_pca = pca.fit_transform(num_feat_train)
num_feat_test_pca = pca.transform(num_feat_test)

# Adding PCA features to main feature-set
num_feat_train = np.append(num_feat_train, num_feat_train_pca, axis = 1)
num_feat_test = np.append(num_feat_test, num_feat_test_pca, axis = 1)

# Undoing the above step
num_feat_train = num_feat_train[:,0:368]
num_feat_test = num_feat_test[:,0:368]


# Using only a few important features
feat_prac = ['ID', 'X0', 'X5', 'X8', 'X44', 'X91', 'X109', 'X112', 'X118', 'X119', 'X127',
             'X238', 'X285', 'X286', 'X287', 'X311']
df_prac_train = train[feat_prac].values 
df_prac_test = test[feat_prac].values


## Regression models
''' At first, will try only with the numerical features '''
# Model 1 : XGBoost Regression
from xgboost import XGBRegressor
model = XGBRegressor(max_depth = 3, learning_rate = 0.05, n_estimators = 300, nthread = -1,
                     objective = "reg:linear", subsample = 0.6, reg_alpha = 0.05, reg_lambda = 10,
                     colsample_bytree = 0.5, base_score = 0.4)

model.fit(df_prac_train, y_train, eval_metric = "rmse")


# Model 2 : Support Vector Regression
from sklearn.svm import SVR
model = SVR(kernel = 'rbf', C = 20)
model.fit(num_feat_train, y_train)

# Model 3 : Linear/Polynomial Regression
from sklearn.linear_model import ElasticNet
model = ElasticNet(fit_intercept = True, normalize = True, alpha = 0.1, l1_ratio = 1, precompute = True)
model.fit(num_feat_train, y_train)

# Model 4 : Random Forest Regression
from sklearn.ensemble import RandomForestRegressor as RFR
model = RFR(n_estimators = 300, max_depth = 8)
model.fit(num_feat_train_pca, y_train)

    

# Model 5 : ANN
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization, Activation
from keras.optimizers import SGD

# Defining custom R2 metric for ANN
def r2_metric(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


## Initialising the ANN
model_NN = Sequential()

# Adding the input layer and one hidden layer
model_NN.add(Dense(units = 128, kernel_initializer = 'glorot_uniform', input_dim = 342))
model_NN.add(BatchNormalization())
model_NN.add(Activation('relu'))
model_NN.add(Dropout(0.4))

# Adding the second hidden layer
model_NN.add(Dense(units = 64, kernel_initializer = 'glorot_uniform'))
model_NN.add(BatchNormalization())
model_NN.add(Activation('tanh'))
model_NN.add(Dropout(0.3))

# Adding the third hidden layer
model_NN.add(Dense(units = 64, kernel_initializer = 'glorot_uniform'))
model_NN.add(BatchNormalization())
model_NN.add(Activation('tanh'))
model_NN.add(Dropout(0.3))

# Adding the fourth hidden layer
model_NN.add(Dense(units = 16, kernel_initializer = 'glorot_uniform'))
model_NN.add(Activation('relu'))

# Adding the output layer
model_NN.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'linear'))

# Compiling the ANN
model_NN.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = [r2_metric])
model_NN.optimizer.lr = 0.01

## Fitting the ANN to the Training Set
history = model_NN.fit(feat_comb_train, y_train, validation_split = 0.2, batch_size = 32, 
                            epochs = 20)



## Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators' : [100, 300, 501],
               'max_depth' : [3, 6, 8],
               'reg_alpha' : [0.05, 0.01, 0.1],
               'reg_lambda' : [10, 1, 5, 30],
               'learning_rate' : [0.05, 0.1, 0.01]}
             ]

grid_search = GridSearchCV(estimator = model, 
                           param_grid = parameters,
                           scoring = "r2",
                           cv = 5, n_jobs = -1)
grid_search = grid_search.fit(df_prac_train, y_train)
best_metric = grid_search.best_score_
best_params = grid_search.best_params_
grid_search.grid_scores_ # See all scores



## Ensembling - Weighted averaging
y_pred_RF1 = model.predict(num_feat_test)
y_pred_XGB1 = model.predict(num_feat_test)
y_pred_RF2 = model.predict(num_feat_test_pca) # PCA
y_pred_lin = model.predict(num_feat_test)
y_pred_SVR = model.predict(num_feat_test) 
y_pred_SVR2 = model.predict(num_feat_test_pca) # with outliers in train data

ensemble_pred = (0.2 * y_pred_RF1) + (0.15 * y_pred_RF2) + (0.2 * y_pred_XGB1) + (0.1 * y_pred_lin) + (0.2 * y_pred_SVR) + (0.15 * y_pred_SVR2)

# Predicting Test Set Results
y_pred = model_NN.predict(feat_comb_test)
y_pred2 = model.predict(df_prac_test)
ensemble = 0.5 * (y_pred + y_pred2)

# Preparing the submission file
np.savetxt('results.csv', y_pred, fmt = '%.3f')




