#!/bin/bash

# optimize processed bag files
# Arguments:
#    1: Host IP Address
#    2: Host Name
#    3: Source Folder of Processed Bag-Files
#    4: Start Number of Bag-Files
#    5: End Number of Bag-Files
# Output:
#    One yaml file for each processed bag file

if [[ $# -ne 5 ]] ; then
	echo "Not enough Arguments. Enter:"
	echo "optimize.sh [IP ADDR] [HOST] [SOURCE FOLDER] [SEQ_START] [SEQ END]"
	exit 1
fi

# Define variable
MASTER=$1     
HOST=$2      
SOURCE=$3
SEQ_START=$4
SEQ=$5

# Define Fork and Map
FORK='jasonhu5'
MAP='ethz_amod_lab_k31'

cd ${SOURCE}

echo ${MASTER} ${HOST} ${SOURCE}

# Optimize all processed bag files
for ((a=SEQ_START; a<=SEQ; a++)); do 
	echo $a
	BAGFILE=$(ls -l processed*_$a.bag | awk '{print $NF}')
	echo "${BAGFILE%.bag}"
	docker run --rm --net=host -it -e ATMSGS_BAG="/data/${BAGFILE}" -e OUTPUT_DIR=/data -e ROS_MASTER=${HOST} -e ROS_MASTER_IP=${MASTER} --name graph_optimizer -v ${SOURCE}:/data -e DUCKIETOWN_WORLD_FORK=${FORK} -e MAP_NAME=${MAP} duckietown/cslam-graphoptimizer:daffy-amd64

	mv autobot04.yaml autobot04_${a}.yaml
done

