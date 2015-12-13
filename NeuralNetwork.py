__author__ = 'nicky'

# inspired by: http://scikit-learn.org/dev/auto_examples/applications/face_recognition.html
# and http://iamtrask.github.io/2015/07/12/basic-python-network/


import numpy as np
from sklearn import cross_validation
from sklearn import decomposition
from sklearn import svm, tree, neighbors, naive_bayes, ensemble, discriminant_analysis
from PIL import Image
from sklearn import neural_network, pipeline, linear_model


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

print 'Calculating using half of images as training...'
imageArray = np.reshape(imageArray, (isWaldoArray.shape[0], -1))

# training classification
#trainingClassifier = svm.SVC(gamma=0.0001)
#trainingClassifier = tree.DecisionTreeClassifier(max_depth = 5)
#trainingClassifier = neighbors.KNeighborsClassifier(3)
#trainingClassifier = naive_bayes.GaussianNB()
#trainingClassifier = ensemble.AdaBoostClassifier()
#trainingClassifier = ensemble.RandomForestClassifier()
logisticRegress = linear_model.LogisticRegression()
rbm = neural_network.BernoulliRBM(n_components=2)
trainingClassifier = pipeline.Pipeline(steps=[('rbm', rbm), ('logistic', logisticRegress)])
trainImages = imageArray[:45]       #An array of the images used to train the machine (first half of all the pictures)
trainResults = isWaldoArray[:45]    #Array of the correct results: 0 (not waldo) or 1 (waldo) (first half of all the results)

trainingClassifier.fit(trainImages, trainResults) #use built in fit function to classify these results to use later

testImages = imageArray[5:]        #An array of the images used to test the machine (second half of all the pictures)
expectedResults = isWaldoArray[5:] #An array of the correct results of the images being tested

testingOutput = trainingClassifier.predict(testImages)

totalPredictions = 5.0
correctPredictions = 0

for i in range(0, len(expectedResults)):
    if expectedResults[i] == testingOutput[i]:
        correctPredictions += 1

score = correctPredictions/totalPredictions * 100
print str(testingOutput)
print '\tAccuracy of Predictions: ' + str(score) + '%'

'''print 'Calculating K Fold Cross Validation...'
kFold = cross_validation.StratifiedKFold(isWaldoArray, 10)
for train_index, test_index in kFold:
        X_train, X_test = imageArray[train_index], imageArray[test_index]
        Y_train = isWaldoArray[train_index]
        Y_test = isWaldoArray[test_index]


# .. dimension reduction ..

pca = decomposition.RandomizedPCA(whiten=True)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# training classification
trainingClassifier = svm.SVC(C=5., gamma=0.00001)
trainingClassifier.fit(X_train_pca, Y_train)
res = trainingClassifier.predict(X_test_pca)
correctPredictions = 0


totalPredictions = 4.0
correctPredictions = 0

for i in range(0, len(res)):
    if res[i] == Y_test[i]:
        correctPredictions += 1

score = correctPredictions/totalPredictions * 100

print '\tAccuracy of Predictions: ' + str(score) + '%'
print 'Results' + str(res)
print 'Expected' + str(Y_test)'''





