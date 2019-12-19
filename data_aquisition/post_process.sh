#!/bin/bash

# Post process recorded bag file from watchtower system and Duckiebot
# Arguments:
#    1: IP address of host
#    2: Path to bag files
#    3: Sequence start
#    4: Sequence end


# Define variables
MASTER="http://$1:11311/"
SOURCE=$2
SEQ_START=$3
SEQ=$4

# Check if enough arguments
if [[ $# -ne 4 ]] ; then
	echo "Not enough arguments. Enter:"
	echo "<IP Addr> <PATH/TO/BAG> <SEQ START> <SEQ END>"
	exit 1
fi

cd ${SOURCE}
echo "${MASTER} ${SOURCE} ${SEQ}" 

# Post process bag files
for ((a=SEQ_START; a<=SEQ; a++)); do 
	echo $a
	BAGFILE=$(ls -l *_$a.bag | awk '{print $NF}')
	echo "${BAGFILE%.bag}"
	docker run --name post_processor -it --rm -e INPUT_BAG_PATH=/data/${BAGFILE%.bag} -e OUTPUT_BAG_PATH=/data/processed_${BAGFILE} -e ROS_MASTER_URI=${MASTER} --net=host -v ${SOURCE}:/data duckietown/post-processor:daffy-amd64
done
