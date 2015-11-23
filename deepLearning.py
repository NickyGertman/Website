__author__ = 'nicky'

# inspired by: http://scikit-learn.org/dev/auto_examples/applications/face_recognition.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn import decomposition
from PIL import Image

X = []
Y = []

#Go through the sample images and assign them their x values (pixel array values) and their Y values (0 or 1)
#The images are already all cropped to be 50x50 px
for i in range (1,25):
    img = Image.open('TrainingImages/Waldo/waldo' + i.__str__() + '.jpg')
    img = img.convert('L')
    X.append(np.asarray(img, dtype=np.uint8)) #convert the image to a numpy array of pixels
    Y.append(1) #Waldo's faces get assigned a true valuation

for i in range (1, 25):
    img = Image.open('TrainingImages/Not_Waldo/notWaldo' + i.__str__()+ '.jpg')
    img = img.convert('L')
    X.append(np.asarray(img, dtype=np.uint8)) #convert the image to a numpy array of pixels
    Y.append(0) #non Waldo faces get assigned a false valuation


train, test = iter(cross_validation.StratifiedKFold(Y, 4)).next()

X_train, X_test = X[train], X[test]
Y_train, Y_test = Y[train], Y[test]

# .. dimension reduction ..
pca = decomposition.RandomizedPCA(whiten=True)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# .. classification ..
clf = svm.SVC(C=5., gamma=0.001)
clf.fit(X_train_pca, Y_train)

print 'Score on unseen data: '
print clf.score(X_test_pca, Y_test)



