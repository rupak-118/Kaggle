### Porto Seguro's Safe Driver Prediction

## Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
# from sklearn.ensemble import ExtraTreesClassifier, AdaBoost Classifier
from sklearn.linear_model import LogisticRegression

## Importing datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


## Basic data exploration
train.describe()
test.describe()
pd.value_counts(train.target, normalize = True)

feat_corr = train.corr()
''' Inferences :
    1) Imbalanced classification problem (~3.6% pos_class)
    2) test has more rows than train. Pseudo-labelling might be required
    3) Missing values have been replaced with -1
    4) Most features are uncorrelated
'''


## Functions built for target encoding
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series = None, tst_series = None, target = None, min_samples_leaf = 1,
                  smoothing = 1, noise_level = 0):
    
    


## Pre-processing the dataset - missing values, Encoding of categorical variables etc.
def preproc(df, df_test, col_to_drop, replace_missing = False, cat_encoding = True, cardinality_limit = 10):
    
    # Dropping target and id columns
    df = df.drop(['target', 'id'], axis = 1)
    df_test = df_test.drop(['id'], axis = 1)
    
    # If any other columns/features have to be dropped
    df = df.drop(col_to_drop, axis = 1)
    df_test = df_test.drop(col_to_drop, axis = 1)
    
    # Replace -1 with np.nan
    if replace_missing == True:
        df.replace(-1, np.nan, inplace = True)
        df_test.replace(-1,np.nan, inplace = True)
    
    if cat_encoding == True:
        # Calculating cardinality of each of the '_cat' features
        feat_cardinality = df.apply(pd.Series.nunique)
        features_df = pd.DataFrame({'features':[a for a in df.columns if a.endswith('cat')]})
        features_df['cardinality'] = features_df.features.map(feat_cardinality)
    
        target_encoding_feat = list(features_df[features_df.cardinality >= cardinality_limit].features)
        ohe_encoding_feat = list(features_df[features_df.cardinality < cardinality_limit].features)
    
        # OneHot Encoding for low cardinality variables
        for col in ohe_encoding_feat :
            dummy_trn = pd.get_dummies(pd.Series(df[col])).rename(columns = lambda x: col + '_' + str(x))
            dummy_trn.drop(dummy_trn.columns[[0]], axis = 1, inplace = True)
            dummy_test = pd.get_dummies(pd.Series(df_test[col])).rename(columns = lambda x: col + '_' + str(x))
            dummy_test.drop(dummy_test.columns[[0]], axis = 1, inplace = True)
            df = pd.concat([df, dummy_trn], axis = 1)
            df_test = pd.concat([df_test, dummy_test], axis = 1)
            df.drop([col], axis = 1, inplace = True)
            df_test.drop([col], axis = 1, inplace = True)
        
        # Target Encoding (with noise) for high cardinality features
    
    
    
    
    
    return df, df_test


## Define the gini metric
''' Gini = (2*AUC - 1)  '''
#def gini(actual, pred, cmpcol = 0, sortcol = 1):
#    assert(len(actual) == len(pred))
#    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
#    all = all[np.lexsort((all[:,2], -1*all[:,1]))]
#    totalLosses = all[:,0].sum()
#    giniSum = all[:,0].cumsum().sum() / totalLosses
#    
#    giniSum -= (len(actual) + 1) / 2.
#    return giniSum / len(actual)

def gini(actual, pred):
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_c[-1] - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


## Preparing train and test datasets
drop_cols = train.columns[train.columns.str.startswith('ps_calc_')]
train_df, test_df = preproc(train, test, drop_cols)
X = train_df.values
X_test = test_df.values
y = train['target'].values 


## Code for oversampling (SMOTE) since the classes are unbalanced
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

#sm = SMOTE(ratio = 'minority')
#enn = EditedNearestNeighbours(ratio = 'not minority')
#sme = SMOTEENN(smote = sm, enn = enn, random_state = 42)

oversampling_ratio = {0 : 600000, 1 : 200000}
sm = SMOTE(ratio = oversampling_ratio)
undersampling_ratio = {0:100000, 1:21000}
enn = EditedNearestNeighbours(undersampling_ratio)

X_res, y_res = enn.fit_sample(X, y)

''' 
SMOTE implementation takes a lot of time on full dataset. Perform feature selection before resampling.
SMOTE is not very effective and might overfit on high-dimensional data.
s = x + {u * (x* - x)}; 0<u<1, s --> synthetic sample, x* --> one of the k-nearest neighbour of x
'''

## Model 1 : Light GBM
import lightgbm as lgb
train_data = lgb.Dataset(X_res, label = y_res)
params = {'num_leaves':151, 'objective': 'binary', 'max_depth':9, 'learning_rate':0.1, 
         'max_bin':222, 'feature_fraction':0.9, 'bagging_fraction':0.7, 'lambda_l1': 10,
         'lambda_l2':1}
params['metric'] = ['auc','binary_logloss']
num_rounds = 150
model_lgb = lgb.train(params, train_data, num_rounds)

## Model 2 : Define XGBoost model
from xgboost import XGBClassifier
model_xgb = XGBClassifier(max_depth = 9, learning_rate = 0.05,
                            n_estimators = 101, objective = "binary:logistic", 
                            gamma = 0, base_score = 0.5, reg_lambda = 5, subsample = 0.8,
                            colsample_bytree = 0.7)

model_xgb.fit(X, y, eval_metric = "auc")
feat_imp = model_xgb.feature_importances_


## K-Fold/StratifiedKFold Cross validation setup
cv_folds = 5
fold_num = 0
skf = StratifiedKFold(n_splits = cv_folds, shuffle = True)
metric_val = np.zeros([cv_folds])
for train_idx, val_idx in skf.split(X,y):
    X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]
    model_xgb.fit(X_train, y_train, eval_metric = "auc")
    metric_val[fold_num]= gini_normalized(y_val, model_xgb.predict_proba(X_val)[:,1])
    fold_num += 1

print("Avg. RMSE for {} folds = {}".format(cv_folds, np.mean(metric_val)))
print("Std. dev. for {} folds = {}".format(cv_folds, np.std(metric_val)))


## Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators' : [51, 100, 301, 500],
               'max_depth' : [6,9],
               'learning_rate' : [0.01, 0.1]}
             ]

grid_search = GridSearchCV(estimator = model_xgb, 
                           param_grid = parameters,
                           scoring = gini_normalized,
                           cv = 5, n_jobs = -1)
grid_search = grid_search.fit(X, y)
best_metric = grid_search.best_score_
best_params = grid_search.best_params_
grid_search.grid_scores_ # See all scores


## Create a LightGBM-compatible metric from Gini
#def gini_xgb(preds, train_data):
#    labels = train_data.get_label()
#    gini_score = gini_normalized(labels, preds)
#    return [('gini', gini_score)]



## Check validation score
#pred_val = model_lgb.predict(X_val)
#pred_trn = model_lgb.predict(X_train)
#gini_normalized(y_val, pred_val)
#gini_normalized(y_train, pred_trn)


''' Resample properly to improve score '''

## Make test predictions
preds = model_xgb.predict_proba(X_test)[:,1]
# Prepare submission file
subm = pd.DataFrame({'id':test['id'].values, 'target':preds})
subm.to_csv('sub07.csv', index=False)












