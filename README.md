# Image-Classifier
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.
This classifier has been tested on a dataset containing 102 different species of flowers. The dataset can be found [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).
This is the final project for Udacity's AI Programming with Python Nanodegree.

## Prerequisites
This project has been written in Python 3.6.5. It should work with Python versions 3.* , Pytorch, PIL.
To work or view the project it is recommended to use Jupyter Notebooks. The classifier.pth file contains the trained classifier, therefore it can be directly used to classify a flower on the terminal also.

## Basic File Information

- **Image Classifier Project.ipynb:** Contains the Python code for the classifier developed on Jupyter Notebook.

- **classifier.pth:** It is the finished classfier trained on a dataset of 102 flower species using GPU. It can be directly used for predicition purposes or can be retrained.

- **cat_to_name.json:** This JSON file is used to map the flower numbers and flower names.

- **train.py:** Application to provide new hyperparameters, training dataset etc., to retrain the network and save it as classifier.pth

- **predict.py:** It is used to classify an input image on the basis of the trained network.

## Authors and Acknowledgement
- **Utkarsh Lath**
- **Udacity**

An interesting and insighful project which would not have been possible without Udacity and its mentors.

The project is licensed under MIT License.
