# Implementation

To run the machine learning based lane following three different docker images 
* `cnn_node`
* `dt-core`
* `dt-car-interface`

need to be built and run on the Duckiebot

## cnn_node
ROS package for machine learning based lane following on the Duckiebot.

A node named cnn_node is initialized. The node subscribes to /camera_node/image/compressed. When the node receives a new image message the image is turned into grayscale and cropped. The cropped grayscale image is the input to the convolutional neural network (CNN). The ouput of the CNN is the relative pose, i.e., distance to middle lane and relative angle of the Duckiebot. The relative pose is used to computed the control signal (velocity and angular velocity) of the Duckiebot using a PID controller. Eventually, the control signal is published in the topic cnn_node/car_cmd. 

For further details see cnn_node.


## [dt-core](https://github.com/wickipedia/dt-core/tree/777fdb3bb02716de814f5845889d64853c7ec702)
`dt-core` is a fork of the official duckietown docker image [dt-core](https://github.com/duckietown/dt-core). We added a new state to the FSM called CNN_LANE_FOLLOWING. The state transition from NORMAL_JOYSTICK_CONTROL to CNN_LANE_FOLLOWING is toggeled if joystick_override_off_and_cnn_lane_following_on is true. The state transition from CNN_LANE_FOLLOWING to NORMAL_JOYSTICK_CONTROL is toggled if joystick_override_on_and_cnn_lane_following_of is true

![fsm_cnn](../documentation/images/fsm_cnn.png)


## [dt-car-interface](https://github.com/wickipedia/dt-car-interface/tree/b6247cecb72d954adf902c095f7cd4147235754a)
`dt-car-interface` is a fork from the official duckietown docker image [dt-car-interface](https://github.com/duckietown/dt-car-interface). We added a new mapping `CNN_LANE_FOLLOWING: "cnn"` and a new source_topic `cnn: "cnn_node/car_cmd"` to the `car_cmd_switch_node`


