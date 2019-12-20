#  Data labeling

## Setup
This module adds labels to our previous collected data.
Before the module can be executed, we need data to work with which is saved in a specific way. 
Specifically, the processed trajectory files, created by the watchtowers are needed. 
These can be one or multiple split ‘.yaml’ files, which are named the same way as the folder they are in.
For example if the folder is called ‘autobot04_r2’, the trajectory .yaml files need to be called autobot04_r2_0.yaml, autobot04_r2_1.yaml.
What is also needed is a ‘image_timestamps.csv’ file. 
This contains the frames taken from the Duckiebot video stream bag files and their timestamp. 
