# proj-lfi-ml / Machine learning based lane following

The goal of the project is to implement a machine learning based lane following algorithm on a [Duckiebot](https://www.duckietown.org/).

We use a convolutional neural network (CNN) to compute the relative pose, heading and distance relative to the middle of the lane, of the Duckiebot. The input of the CNN is the live image stream of the Duckiebot. The Duckiebot is then controlled using a PID controller with the computed pose as the input.

A major part of the project is the creation of a framework that semi-automates the data acquisition and labeling process. New models can easily be trained and employed. The data is recorded using the localization system in Duckietown. 

This repo contains all the files which we used to complete the project as well as a step-by-step guide on how to run it on any Duckiebot. 

We split the project into three different parts:
* Data acquisition and labeling
* Training of the CNN (Convolutional Neural Network)
* Implementation on the Duckiebot


The readme files in each subset explains in detail how each task can be executed.

In the following is a short summary of the folders:


## [Data_sets](data_sets)
Already labeled data sets that can be used for model-training.

## [Data_acquisition](data_acquisition)
The whole data acquisition pipeline. From recording data with the localization system to recording data on the Duckiebot, post-processing and labeling the data.

## [Documentation](documentation)
Documentation of the project

## [Implementation](implementation)
Instructions on how to run a pre-trained model on a Duckiebot with our tuned controller.

## [Training](training)
Instructions and scripts on how to train a neural-network with labeled data.
