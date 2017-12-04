## Kaggle Digit Recognizer problem

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print 'Imported Libraries'

# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_train = train.iloc[:, 1::].values
labels_train = train.iloc[:, 0].values
X_test = test.values

print 'Loaded Datasets'

# Plotting sample images in test and train
images_train = X_train.reshape(X_train.shape[0], 28, 28)
images_test = X_test.reshape(X_test.shape[0], 28, 28)
# Train images
plt.figure(figsize = (5, 5))
for i in range(0, 100):
    plt.subplot(10, 10, i+1)
    plt.imshow(images_train[i], interpolation = "none")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
# Test images
plt.figure(figsize = (5, 5))
for i in range(0, 50):
    plt.subplot(5, 10, i+1)
    plt.imshow(images_test[i], interpolation = "none")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()

# Feature Scaling (required for PCA and efficient for NNs)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


print 'Pre-processing done'

''' We'll try different ML models. Particularly, SVM and RF combined along with 
    dimensionality reduction techniques (PCA, LDA etc.) '''


## Dimensionality reduction techniques
''' We check if dimensionality reduction is required by plotting correlation charts between
    different features. PCA and LDA plots are plotted. '''


# Draw a heatmap using seaborn (correlation plot of features)
f, ax = plt.subplots(figsize=(7, 8))
plt.title('Correlation plot of a 100 columns in the MNIST dataset')
sns.heatmap(train.corr(),linewidths=0, square=True, 
            cmap="viridis", xticklabels=False, yticklabels= False, annot=True)
''' Lots of features are fairly correlated. Hence, PCA/LDA is required. '''
# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
plt.plot(np.arange(1, 785), explained_variance.cumsum()) # Plots the explained variance curve
''' From the above curve, 300 seems to be the ideal number of PCA components. It explains
    ~ 94% of variance  ''' 
del X_train_pca, X_test_pca
pca = PCA(n_components = 300)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# LDA
''' For classification tasks LDA usually works better than PCA 
    LDA doesn't require feature scaling. There are 10 classes. Thus, max. no. of 
    n_components can be 9. '''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 9)
X_train_lda = lda.fit_transform(X_train, labels_train)
X_test_lda = lda.transform(X_test)


# Model 1 : ML - SVM with dimensionality reduction (PCA, LDA)
from sklearn.svm import SVC
''' Image classification essentially involves non-linear operations. Hence, kernel SVM is used ''' 
classifier_svm_pca = SVC(kernel = 'rbf', probability = False, C = 10)
classifier_svm_pca.fit(X_train_pca, labels_train)

classifier_svm_lda = SVC(kernel = 'rbf', probability = False, C = 5)
classifier_svm_lda.fit(X_train_lda, labels_train)
''' LDA + SVM gave poor accuracy (~80%) on training data. Hence, re-trained, using higher
    no. of discriminants and larger C value. Grid search is used to arrive at suitable C'''

# Model 2 : ML - RF with dimensionality reduction
from sklearn.ensemble import RandomForestClassifier
classifier_rf_pca = RandomForestClassifier(n_estimators = 50, criterion = "entropy")
classifier_rf_pca.fit(X_train_pca, labels_train)

classifier_rf_lda = RandomForestClassifier(n_estimators = 500, criterion = "entropy", 
                                           min_samples_split = 4, n_jobs = -1)
classifier_rf_lda.fit(X_train, labels_train)


# Model 3 - kNN (k-Nearest Neighbours)
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform', 
                                      metric = 'minkowski', p = 2, n_jobs = -1)
print 'Fitting the model'
classifier_knn.fit(X_train, labels_train)

''' Calculating score :  classifier_knn.score(X_train[0:10], labels_train[0:10]) '''
print 'Starting Grid search'

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_neighbors' : [2, 3, 5, 7, 10]}
             ]
'''{'n_estimators' : [100, 300, 500, 501, 750, 1000]}'''
'''{'C' : [50, 75, 100],
              'kernel' : ['poly'], 
               'degree' : [7,9]}'''
'''{'C' : [0.05, 0.1, 0.5, 1, 5, 10, 50],
               'kernel' : ['rbf', 'linear',]}'''
'''{'C' : [0.1, 0.5, 1, 5, 10]} '''

grid_search = GridSearchCV(estimator = classifier_knn, 
                           param_grid = parameters,
                           scoring = None,
                           cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, labels_train)
best_metric = grid_search.best_score_
best_params = grid_search.best_params_

# Predicting results

labels_pred_svm_pca = classifier_svm_pca.predict(X_test_pca)
labels_pred_svm_lda = classifier_svm_lda.predict(X_test_lda)
labels_pred_rf_pca = classifier_rf_pca.predict(X_test_pca)
labels_pred_rf_lda = classifier_rf_lda.predict(X_test_lda) 
labels_pred_knn = classifier_knn.predict(X_test)

# Writing the results to a csv file
np.savetxt('results.csv', labels_pred_knn)


