# Says-One-Neuron-To-Another
We have implemented this assignment in two parts.
1. We have done image classification with convolutional neural network.
  
  i. Dataset used is CIFAR-10 dataset. In the notebook we just tried extracting the dataset from tensorflow to see what data look like.    Other tasks related to data loading and extraction is written in "\image_classification\CNN_image_classification.data_utils.py".
  
  ii. "image_classification\CNN_image_classification\layers.py" and "image_classification\CNN_image_classification\layer_util.py" file is having all methods defined related to all the layer operations such as Forward pass, backward pass, pooling, non-linearity related functions like reLu and affine, batch normalization.
  
  iii. "image_classification\CNN_image_classification\solver.py" encapsulates all the logic necessary for training classification models.
  
  iv. "image_classification\CNN_image_classification\vis_utils.py" has all the functions defined related to visualization.
  
  v. "image_classification\CNN_image_classification\optim.py" file implements various first order update rules that are commonly used for training neural networks. 
  
  vi. The notebok "image_classification\Edge_detection.ipynb" is written for detecting edges of images that was our initial step. We took two sample images "index.jpeg" and "minion.jpg" to test the code.
  
  vii. The notebook "image_classification\ConvolutionalNetworks.ipnyb" has the code for image classification where we combined all the basic layers into one ConvNet and trained our dataset with 3-layer convNet and the functions for that are written in "image_classification\CNN_image_classification\classifiers\cnn.py"
  
  viii. We trained our dataset on windows machine on cpu and hence it took very long to get done.
  
  ix. Datasets are uploaded in the folder "image_classification\CNN_image_classification\datasets".
 
 References : https://en.wikipedia.org/wiki/Convolutional_neural_network

2. We did text classification with neural network. for spam filetering

1.We used the data set from the given wiki links that "ling spam data"(Ligustics data) set which has email spams with it
2. Extracted all the text into a data frame and labeled it based on the file name (file name for each mail text file name contains "spm" if it is spam. we extracted for testing and verification
3. Preprocessing. extarcted common words from spam data emails considered the common words with higher probability as features in the data frame
4. added every word as a column as for column values added 1 if the word exists in the text ot 0 if not.
5. used the this as numpy array of column values as input for our algorithm
6. Classified based on the weights of output as Spam /Not spam

Rferences :
https://www.researchgate.net/publication/281452141_Spam_Detection_using_a_Neural_Network_Classifier
https://en.wikipedia.org/wiki/Backpropagation

