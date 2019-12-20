# Data acquisition 

We recorded the data using the [duckietown localization system](https://docs.duckietown.org/daffy/opmanual_autolab/out/autolab_localization.html)

To record data follow the instruction [here](https://docs.duckietown.org/daffy/opmanual_autolab/out/localization_demo.html). We recommend to increase the buffer size of `rosbag record` with the `-b` tag (we used `-b 2048`) and to split the recording into smaller ones (we used 15 seconds split) using either the `--split` tag, e.g.,

```
$ rosbag record --split --duration=10 -b 2048 -a
```

or after the recording with the [split bash script](https://github.com/duckietown-ethz/proj-lfi-ml/blob/master/data_acquisition/utils/rosbag_split.sh) we provide. Increasing the buffer size avoids the dropping of messages during the recording with the localization system and splitting the rosbag will lower the memory usage of the post-processing and optimization. 

To automate the post-processing and optimization we provide two functions [post_process.sh](https://github.com/duckietown-ethz/proj-lfi-ml/blob/master/data_acquisition/utils/post_process.sh) and [optimize.sh](https://github.com/duckietown-ethz/proj-lfi-ml/blob/master/data_acquisition/utils/optimize.sh). For details see [Utilities](https://github.com/duckietown-ethz/proj-lfi-ml/blob/master/data_acquisition/README.md#utilities).


## [Data Labeling](data_labeling)
For details see [data_labeling/README.md](data_labeling/README.md)

## [Utilities](utils)

Utilities to automate the data acquisition

### [split bash script](https://github.com/duckietown-ethz/proj-lfi-ml/blob/master/data_acquisition/utils/rosbag_split.sh)
Split the recorded bag file into smaller ones. The lengths of the split can be defined in the bash file. Just change the variable `STEPS` to the desired time, e.g., `STEPS=15` for a split duration of 15 seconds.

### [post_process.sh](https://github.com/duckietown-ethz/proj-lfi-ml/blob/master/data_acquisition/utils/post_process.sh)
If the recorded data is split it's more convenient to use our post_process.sh script. The split data has to be stored in a single directory and the have the following naming convention *_<SEQUENCE NUMBER>.bag, e.g., recording1_0.bag, recording1_1.bag etc. Execute the following command.
```
$ bash post_process <Host IP Address> <path/to/rosbag> <sequence start> <sequence end>
```

### [optimize.sh](https://github.com/duckietown-ethz/proj-lfi-ml/blob/master/data_acquisition/utils/optimize.sh)
If the post_processed data is split it's more convenient to use our optimize.sh script. The split data has to be stored in a single directory and the have the following naming convention processed_*_<SEQUENCE NUMBER>.bag, e.g., processed_recording1_0.bag, processed_recording1_1.bag etc. Execute the following command.

```
$ bash optimize.sh <Host IP Address> <Host> <path/to/processed_rosbag> <sequence start> <sequence end>
```
You have to change the fork and map name in the bash file (variables FORK, MAP) to the desired fork and map.


### [process_image.py](https://github.com/duckietown-ethz/proj-lfi-ml/blob/master/data_acquisition/utils/process_image.py)
ROS node that converts the rostopic /<VEHICLE_NAME>/imageSparse/compressed to jpg images and stores the image name and and the corresponding timestamp in a csv file. To use the python script change the variable `path` to the desired location of the jpg images and the csv file and the variable `bot` to the VEHICLE_NAME. The images are stored in a subfolder called images/ and have the naming convention \<VEHICLE_NAME\>\_\<TIMESTAMP_SEC\>\_\<TIMESTAMP_NSEC\>.jpg. The csv file is named image_timestamps.csv and has the format \[\<image name\>, \<sequence number image\>, \<timestamp seconds\>, \<timestamp nanoseconds\>\].

Example:
* change variable `path = /home/oliolioli/recording/` and `bot = autobot05`
* Start a rosmaster
* Execute python script 
```
python process_image.py
```
* Images and csv file are stored as follows:
    + /home/oliolioli/recording/
        + timestamps.csv
        + images/
            + autobot05_111111_111111.jpg
            + autobot05_111111_222222.jpg
            + ...









