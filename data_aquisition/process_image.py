#!/usr/bin/env python

import rospy
import math
import os
import numpy as np
import sys
from time import sleep
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from random import *
import csv

'''
Convert rosbag image stream to jpg images
'''

count = 0
# Autobot name
bot = 'autobot05'
# Path where you want to store files
path = "/media/elias/Samsung_T51/recordings/rec5_bright_curve_allposs/autobot05/"

# Callback function
def convertStream(data):
    cv_image = CvBridge().compressed_imgmsg_to_cv2(data, desired_encoding="passthrough")
    print(data.header)
    global count
    global path
    global bot
    name = path + "images/" + bot + "_" + str(data.header.stamp.secs) + "_" + str(data.header.stamp.nsecs)
    count += 1
    # Write csv file
    pose_file.writerow([name, data.header.seq, data.header.stamp.secs, data.header.stamp.nsecs])
    print("write image" + str(count))
    # Store images
    cv2.imwrite(name + '.jpg', cv_image)

# Open csv file
with open(path + 'image_timestamps.csv','w') as csvfile:
    topic = '/' + bot + '/imageSparse/compressed' 
    # Initialize Node
    rospy.init_node('bag2jpg', anonymous=True)
    pose_file = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # Initialize Subcriber
    rospy.Subscriber(topic, CompressedImage,convertStream)

    rospy.spin()


