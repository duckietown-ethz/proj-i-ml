# Implementation

To run the machine learning based lane following three different docker images (`cnn_node`, `dt-core` and `dt-car-interface`) need to be built and run on the Duckiebot

## cnn_node
ROS package which subscribes to a image stream and publishes the control output (velocity and angular velocity) of the duckiebot.  

## dt-core
`dt-core` is a fork of the official duckietown docker image [dt-core](https://github.com/duckietown/dt-core). We added a new state to the FSM called CNN_LANE_FOLLOWING. 

[fsm_cnn](documentation/images/fsm_cnn)


## dt-car-interface
`dt-car-interface` is a fork from the official duckietown docker image [dt-car-interface](https://github.com/duckietown/dt-car-interface)


