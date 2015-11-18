# Where's Waldo: A Study in Image Recognition

#This project will compare two separate approaches to finding Waldo in the famous picture game, Where's Waldo. 

#The first approach is a human written algorithm: take an image's red channel to filter out all non-red colors. Then, look for horizontal red and white stripes. This will lead to a few results which we can then filter out by comparing the faces of the the characters to the face of Waldo. 

#The second approach involves machine learning. We will use a Neural Network to study the image of characters that are and are not Waldo to train the computer. We will normalize the data by taking images and making them all the same size. After the computer is trained, we will be able to have a moving window go over each image and show the parts that produce a positive result: Waldo.

#The purpose of this study is to improve the old-fashioned method of finding Waldo by hand. It is also a great learning experience for me because I have never done any sort of image manipulation/processing before.


