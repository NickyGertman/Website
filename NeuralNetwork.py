__author__ = 'nicky'

# inspired by: http://scikit-learn.org/dev/auto_examples/applications/face_recognition.html
# and http://iamtrask.github.io/2015/07/12/basic-python-network/


import numpy as np
from PIL import Image
from sklearn import neural_network, pipeline, linear_model


imageArray = np.array([])
isWaldoArray = np.array([])

#Build up the training list
#Go through the sample images and assign them their x values (pixel array values) and their Y values (0 or 1)
#The images are already all cropped to be 50x50 px
for i in range (1,26):
    img = Image.open('TrainingImages/Waldo/waldo' + i.__str__() + '.jpg')
    #convert the images to a number format so that we can work with the data. This is done by converting the image to a numpy array
    imageArray = np.append(imageArray, np.asarray(img, dtype=np.uint8)) #convert the image to a numpy array of pixels
    isWaldoArray = np.append(isWaldoArray,1) #Waldo's faces get assigned a true valuation
    img = Image.open('TrainingImages/Not_Waldo/notWaldo' + i.__str__()+ '.jpg')
    #convert the images to a number format so that we can work with the data. This is done by converting the image to a numpy array
    imageArray = np.append(imageArray, np.asarray(img, dtype=np.uint8)) #convert the image to a numpy array of pixels
    isWaldoArray = np.append(isWaldoArray,0) #non Waldo faces get assigned a false valuation

print 'Calculating using 40 images as training...'
imageArray = np.reshape(imageArray, (isWaldoArray.shape[0], -1))

# training classification
logisticRegress = linear_model.LogisticRegression()     #selects features to use when fitting the model
rbm = neural_network.BernoulliRBM(n_components=2)       #http://scikit-learn.org/stable/modules/neural_networks.html#rbm has a good definition of this tool here. Basically, this is unsupervised pre-training.
trainingClassifier = pipeline.Pipeline(steps=[('rbm', rbm), ('logistic', logisticRegress)])     #more classification
trainImages = imageArray[:40]       #An array of the images used to train the machine (first 40 pictures)
trainResults = isWaldoArray[:40]    #Array of the correct results: 0 (not waldo) or 1 (waldo) (first 40 results)

trainingClassifier.fit(trainImages, trainResults) #use built in fit function to classify these results to use later

testImages = imageArray[40:]        #An array of the images used to test the machine (second half of all the pictures)
expectedResults = isWaldoArray[40:] #An array of the correct results of the images being tested

testingOutput = trainingClassifier.predict(testImages)

totalPredictions = 10.0
correctPredictions = 0

#go through each result and if the result is correct, increase the correctPredictions count
for i in range(0, len(expectedResults)):
    if expectedResults[i] == testingOutput[i]:
        correctPredictions += 1

score = correctPredictions/totalPredictions * 100
print str(testingOutput)
print '\tAccuracy of Predictions: ' + str(score) + '%'

