__author__ = 'nicky'

# inspired by: http://scikit-learn.org/dev/auto_examples/applications/face_recognition.html
# and http://iamtrask.github.io/2015/07/12/basic-python-network/


import numpy as np
from sklearn import cross_validation
from sklearn import decomposition
from sklearn import svm
from PIL import Image


imageArray = np.array([])
isWaldoArray = np.array([])

#Build up the training list
#Go through the sample images and assign them their x values (pixel array values) and their Y values (0 or 1)
#The images are already all cropped to be 50x50 px
for i in range (1,26):
    img = Image.open('TrainingImages/Waldo/waldo' + i.__str__() + '.jpg')
    img = img.convert('L')
    imageArray = np.append(imageArray, np.asarray(img, dtype=np.uint8)) #convert the image to a numpy array of pixels
    isWaldoArray = np.append(isWaldoArray,1) #Waldo's faces get assigned a true valuation
    img = Image.open('TrainingImages/Not_Waldo/notWaldo' + i.__str__()+ '.jpg')
    img = img.convert('L')
    imageArray = np.append(imageArray, np.asarray(img, dtype=np.uint8)) #convert the image to a numpy array of pixels
    isWaldoArray = np.append(isWaldoArray,0) #non Waldo faces get assigned a false valuation

imageArray = np.reshape(imageArray, (isWaldoArray.shape[0], -1))

kFold = cross_validation.StratifiedKFold(isWaldoArray, 10)
for train_index, test_index in kFold:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = imageArray[train_index], imageArray[test_index]
        Y_train = isWaldoArray[train_index]
        Y_test =  isWaldoArray[test_index]
        print(X_test)
        print(X_train)

print(X_test.shape)
print(X_train.shape)
#train, test = iter(cross_validation.StratifiedKFold(isWaldoArray, 4)).next()

#X_train, X_test = imageArray[train], imageArray[test]
#Y_train, Y_test = isWaldoArray[train], isWaldoArray[test]

# .. dimension reduction ..

pca = decomposition.RandomizedPCA(whiten=True)
pca.fit(X_train)
print(pca)
print(len(X_test))
X_train_pca = pca.transform(X_train)
print(len(X_train))
print(X_test.shape)
print(X_train.shape)
X_test_pca = pca.transform(X_test)

# .. classification ..
clf = svm.SVC(C=5., gamma=0.00001)
clf.fit(X_train_pca, Y_train)

print 'Score on unseen data: '
print clf.score(X_test_pca, Y_test)

#np.random.seed(1)

#syn0 = 2*np.random.random((48,1)) - 1


#for iter in xrange(10000):
    # forward propagation
    #l0 = X
    #l1 = nonlin(np.dot(l0,syn0))

    # how much did we miss?
   # l1_error = Y - l1

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
  #  l1_delta = l1_error * nonlin(l1,True)

    # update weights
 #   syn0 += np.dot(l0.T,l1_delta)

#print "Output After Training:"
#print l1



