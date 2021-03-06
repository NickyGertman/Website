#Who is Waldo and why do I want to find him?
Where's Waldo is a children's picture game invented by Martin Handford. The game involves a large image filled with various characters and funny situations. The reader is supposed to search through the image in order to find Waldo, who can always be found wearing a red and white polo, glasses, and a bobble hat. The image usually contains a few characters who look an awful lot like Waldo, but are only meant to trick the reader. [1]
![WheresWaldo](https://cloud.githubusercontent.com/assets/15268123/11768783/54557c38-a18c-11e5-81dc-0d702c9d569b.jpg)
_Figure 1: Unmodified Where's Waldo puzzle_

#This project will compare two separate approaches to finding Waldo in the famous picture game, Where's Waldo. 

### First Approach
The first approach utilizes color channels to track down Waldo. The first step is to take an image's red channel and filter out all non-red colors. This will leave behind only the reds in the image as seen in Figure 2. The black areas indicate that the area was red in the original image.  ![Reds only](https://cloud.githubusercontent.com/assets/15268123/11768803/e82d49e0-a18c-11e5-8f84-43a03171c01a.jpg) _Figure 2: Red channel filter applied to Where's Waldo puzzle_

Then I created a white channel filter and applied it to the original image. I combined these two filtered results in order to identify areas that contained both red and white. This combination is shown in Figure 3. In this image, everything red and white is displayed in white and everything else appears black: 
![redsAndWhites](https://cloud.githubusercontent.com/assets/15268123/11768826/9975e0a4-a18d-11e5-94e9-af049eb52040.jpg) _Figure 3: Red and white channel of Where's Waldo puzzle_

Finally, I apply the combined red and white filter to the original image and circle the positive results as displayed in Figure 4. Clearly, this contains far too many false positives. However, it does correctly identify Waldo at the very least. [View this code here.](https://github.com/NickyGertman/Website/blob/master/RedChannel.py)
![results](https://cloud.githubusercontent.com/assets/15268123/11768890/6fb97e54-a18f-11e5-8158-47e45a37a585.jpg) _Figure 4: Final image with circled areas found to contain Waldo by the algorithm_

###Second Approach

The second approach involves machine learning. I will use a Neural Network to study the images of characters that are and are not Waldo to train the computer. In order to accomplish this, I first had to find Waldo by hand in 25 different puzzles. Then, I normalized the face data by cropping each of Waldo's faces to be 50x50 pixels. In order to give the computer an example of 'non-Waldo' faces, I used the same process to create 25 face images from the game that were not Waldo. Here are two example training images:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![Waldo](https://cloud.githubusercontent.com/assets/15268123/11772081/63b30722-a1cd-11e5-8eeb-1b21501d3b62.jpg)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![not Waldo](https://cloud.githubusercontent.com/assets/15268123/11772084/6e0dd346-a1cd-11e5-84fc-05cb607ada14.jpg)

The actual training process involves using these images as the X-values in a function, and assigning the Y-values as whether the image is Waldo or not. Waldo images are assigned a Y-value of 1 (for true) and non-Waldo images are assigned a Y-value of 0 (for false). Once all of these values are assigned, I use a subset of the images to train a classifier. The classifier has a fit function that learns from the model by using a training set of  X and Y values. The fit is then applied to future 'predictions' - when I test the function by giving it an image and having it detect whether the image is Waldo or not. 

Choosing the correct type of classifier proved to be the most difficult aspect of this approach. I tried various training classifiers from the scikit-learn python library in an effort to improve the accuracy of the predictions the machine was making. In order to compare the results between different classifiers, I created multiple versions of the learning algorithm,  The classifiers I used were: [K-Nearest Neighbors](https://github.com/NickyGertman/Website/blob/master/KNearestNeighbors), [General Neural Network](https://github.com/NickyGertman/Website/blob/master/NeuralNetwork.py), [Naive Bayes](https://github.com/NickyGertman/Website/blob/master/NaiveBayes), [Support Vector Machine](https://github.com/NickyGertman/Website/blob/master/PCAandSVM) (SVM), and finally [AdaBoost-SAMME](https://github.com/NickyGertman/Website/blob/master/AdaBoost-SAMME). Figure 5 demonstrates the portion of the neural network that I was working to find: the hidden layer.

![Neural Network](https://cloud.githubusercontent.com/assets/15268123/11772277/cb581f78-a1cf-11e5-9ecd-0bde8b4e2516.JPG) 

_Figure 5: Neural network flow-map \[2]_

Classifiers Used:

 * K-Nearest Neighbor classifies an image by comparing it to its "nearest neighbors" and assigning the classification that matches the majority of its k neighbors [3]. 
 * The general neural network classifier from scikit-learn is "known as unsupervised pre-training" and 
"tries to maximize the likelihood of the data using a particular graphical model."[4] 
 * Naive Bayes method is a "set of supervised learning algorithms based on applying Bayes’ theorem with the 'naive' assumption of independence between every pair of features".[5] 
 * SVM is a supervised learning approach that "builds a model that assigns new examples into one category or the other"[6]. 
 * AdaBoost-SAMME makes multiple passes over the training data. It will come up with a preliminary 'fit' function and then pass over the data again to test it out. It then changes its weights based on the incorrect predictions in order to make it more accurate. [7]

These different classification functions had interesting results for their predictions. The Naive Bayes approach was by far the worst. I ran this version multiple times with multiple sizes of partitions of my data, and on average, it was only 40% accurate (Note: random classification is 50% accurate, so this is very bad.)

Then, I ran the generic NN, K-Nearest Neighbors, and SVM classifiers. Surprisingly, these three different methods all had similar results to each other. I ran these versions multiple times with multiple sizes of partitions of my data, and on average, they were each about 50% accurate (still not good, but better than before).

Finally, I was able to test the AdaBoost-SAMME algorithm. I performed testing on this method multiple times and with multiple data sets and, on average, I got about 60% accuracy. This is by no means a great triumph in the world of Artificial Intelligence, but it was better than any of the other classifiers I had tried. 

The accuracy of the different classifiers is visualized in Figure 6. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![chart](https://cloud.githubusercontent.com/assets/15268123/11771697/d38f7d3c-a1c8-11e5-9c8a-64c5b5c34225.JPG)

 _Figure 6: Rate of correct Waldo predictions according to different classifiers_

One thing that probably would have improved the overall results of this research project would have been to increase the size of my data set. More images would have improved the training of the classifiers and would improve their classification results. I hope to be able to find more data in the future and possibly improve upon this work. 

This project was completed in a one semester Artificial Intelligence class. Although the experiment did not have the results I had been hoping for, I do feel that it was a very educational and interesting project. 

###Sources:

[1] https://en.wikipedia.org/wiki/Where's_Wally%3F

[2] http://www.texample.net/tikz/examples/neural-network/

[3] 	https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

[4] http://scikit-learn.org/stable/modules/neural_networks.html

[5] http://scikit-learn.org/stable/modules/naive_bayes.html

[6] https://en.wikipedia.org/wiki/Support_vector_machine#Support_Vector_Clustering_.28SVC.29

[7] http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
