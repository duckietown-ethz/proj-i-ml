# proj-lfi-ml
Machine learning based lane following.

The goal of the project is to implement a machine learning based lane following algorithm on a [Duckiebot](https://www.duckietown.org/).
This repo contains all files which we used to complete the project. 

We split the project into three different part:
* Data acquisition and labling
* Training of the CNN
* Implementation on the Duckiebot

In the following is a short summery of the folders.
## [cnn](cnn)

## [cnn_node](https://github.com/wickipedia/cnn_node/tree/dd49d002a83657bd514b75f84b36686a2805b994)
ROS package which runs the convolutional neural network to estimate the pose.

## [data](data)
Labeled data

## [data_acquisition](data_acquisition)
How to acquire and process the data using the duckietown watchtower system

## [data_labeling](data_labeling)
How to label the data

## [demo](demo)
How to run the demo

## [documentation](documentation)
Documentation of the project

## [dt-car-interface](https://github.com/wickipedia/dt-car-interface/tree/b6247cecb72d954adf902c095f7cd4147235754a)
modified dt-car-interface

## [dt-core](https://github.com/wickipedia/dt-core/tree/777fdb3bb02716de814f5845889d64853c7ec702)
modified dt-core

