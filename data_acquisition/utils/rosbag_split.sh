#!/bin/bash

# Split rosbag into several smaller bags
# Arguments: rosbag file name

if [[ $# -ne 1 ]] ; then
	echo "<ROS.BAG>"
	exit 1
fi

ROSBAG=$1
# Defines lenght of bag split in seconds
STEPS=10

START=$(rosbag info -y -k start ${ROSBAG}.bag)
DURATION=$(rosbag info -y -k duration ${ROSBAG}.bag)
T_TOSEC=t.to_sec\(\)

echo ${START}
echo ${DURATION}

# compute iteration number
NR=$(python -c "print(round(${DURATION}/${STEPS}))")

# Split bag files
for ((i=1; i<=NR; i++))
do
	LOW=$(python -c "print(${START}+(${i}-1)*${STEPS})")
	UP=$(python -c "print(${START}+${i}*${STEPS})")
	eval "rosbag filter ${ROSBAG}.bag ${ROSBAG}_$i.bag \"t.to_sec() >= ${LOW} and ${T_TOSEC} <= ${UP}\""
done

