#Q3
import numpy as np
from sklearn import datasets
import sklearn.discriminant_analysis
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

mean1 = [4,5,4,5,4,5,4,5,4,5]
mean2 = [-4,-5,-4,-5,-4,-5,-4,-5,-4,-5]

np.random.seed(42)
covpre1 = 5 + 40 * np.random.randn(10, 10)
cov1 = (np.dot(covpre1, covpre1.transpose()))
np.random.seed(42)
covpre2 = 20 + 20 * np.random.randn(10, 10)
cov2 = (np.dot(covpre2, covpre2.transpose()))

np.random.seed(42)
x1 = np.random.multivariate_normal(mean1,cov1, 1000)
np.random.seed(42)
x2 = np.random.multivariate_normal(mean2,cov2, 1000)

X = np.concatenate((x1,x2))
print(X.shape)
Xc = np.ones(1000)
Xc = np.concatenate((Xc, np.zeros(1000)))

XTrain, XTest , XcTrain, XcTest = train_test_split(X,Xc, test_size=0.2, stratify=Xc)

lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(XTrain,XcTrain)
prediction = lda.predict(XTest)
ogError = sum(abs(prediction - XcTest))
ogClassError = (ogError/XTest.shape[0]) * 100
classError = []
classError.append(ogClassError)
print(classError)

for i in range(5):
    subClassError = []
    for col in range(X.shape[1]):
        XNew = np.delete(X,col,1)
        XTrain, XTest , XcTrain, XcTest = train_test_split(XNew,Xc, test_size=0.2, stratify=Xc)
        lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
        lda.fit(XTrain,XcTrain)
        prediction = lda.predict(XTest)
        error = sum(abs(prediction - XcTest))
        subClassError.append(error/XTest.shape[0] * 100)
    accuracyLstSorted = np.argsort(subClassError)
    classError.append(subClassError[accuracyLstSorted[0]])
    X = np.delete(X,accuracyLstSorted[0],1)
    print(classError)

plt.plot(range(1,7), classError, label = 'Classification Error(%)')
plt.ylim([0,100])
plt.legend()
plt.show()

# Q4
# After Reconstruction the deconstructed data we can see that the mean square error increases on further reconstructions.
# This is because once we deconstruct the data we have less and less data to help us reconstruct. There is a lot of data loss
# which makes it difficult to reach the original data
# 
# While plotting the classification error graph of LDA after PCA we can see that, for the most part the error rate remains constant.
# The error hangs around at approximately 45% as there is a lot of overlapping between the 2 classes. 
# 
# In Q3 where we are applying LDA to classify after feature selection, we can see that the classification error goes down from 45% 
# to 40% after removing the least important dimentions. Bt removinng the least important dimensions we are reducing the overlapping
# and making it easier to classify while retaining the most most of the information.
# 
# In this case we can see that feature selection is comparitively better than using PCA as we are able to reduce the classification
# after removal of least important features